
from pathlib import Path
import json
import shutil

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = str(ROOT / "data" / "processed_parquet" / "train")
TEST_PATH = str(ROOT / "data" / "processed_parquet" / "test")

MODEL_PATH = ROOT / "saved_models" / "stage2_balanced"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_PER_CLASS = 1000


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def prepare_attacks(df):
    """Keep attack records with valid, known attack categories."""
    return df.filter(
        (F.col("is_attack") == 1)
        & F.col("attack_type").isNotNull()
        & (F.trim(F.col("attack_type")) != "")
        & (~F.lower(F.trim(F.col("attack_type"))).isin(
            "normal", "unknown", "none"
        ))
    ).select("features", "attack_type")


def calculate_metrics(predictions, class_names):
    """Calculate accuracy, macro-F1, weighted-F1 and per-class metrics."""
    rows = (
        predictions.groupBy("label", "prediction")
        .count()
        .collect()
    )

    matrix = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in rows
    }

    total = sum(matrix.values())
    correct = sum(
        count
        for (actual, predicted), count in matrix.items()
        if actual == predicted
    )

    per_class = []
    f1_scores = []
    weighted_f1_sum = 0.0

    for index, name in enumerate(class_names):
        tp = matrix.get((index, index), 0)

        actual_count = sum(
            count
            for (actual, _), count in matrix.items()
            if actual == index
        )

        predicted_count = sum(
            count
            for (_, predicted), count in matrix.items()
            if predicted == index
        )

        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual_count if actual_count else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class.append({
            "attack_type": name,
            "support": actual_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        f1_scores.append(f1)
        weighted_f1_sum += f1 * actual_count

    return {
        "rows_evaluated": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        ),
        "weighted_f1": (
            weighted_f1_sum / total if total else 0.0
        ),
        "per_class": per_class,
        "confusion": matrix,
    }


def save_json(filename, data):
    path = RESULTS_DIR / filename

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Saved: {path}")


# --------------------------------------------------
# 3. START SPARK
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("BalancedStage2AttackProfiling")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


try:
    # --------------------------------------------------
    # 4. LOAD DATA
    # --------------------------------------------------

    print("\nLoading datasets...")

    train = prepare_attacks(spark.read.parquet(TRAIN_PATH))
    test = prepare_attacks(spark.read.parquet(TEST_PATH))

    fit_data, validation_data = train.randomSplit(
        [0.8, 0.2], seed=SEED
    )

    fit_data = fit_data.cache()
    validation_data = validation_data.cache()

    fit_count = fit_data.count()
    validation_count = validation_data.count()
    test_count = test.count()

    print(f"Original fit rows: {fit_count}")
    print(f"Validation rows: {validation_count}")
    print(f"Official test attack rows: {test_count}")

    # --------------------------------------------------
    # 5. BALANCE TRAINING DATA ONLY
    # --------------------------------------------------

    print("\nOriginal fit class distribution:")

    counts = {
        row["attack_type"]: row["count"]
        for row in fit_data.groupBy("attack_type").count().collect()
    }

    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    if not counts or any(count <= 0 for count in counts.values()):
        raise ValueError("One or more attack classes have no fit records.")

    print(
        f"\nCreating approximately {TARGET_PER_CLASS} "
        "training records per class..."
    )

    balanced_parts = []

    for name, count in counts.items():
        class_df = fit_data.filter(F.col("attack_type") == name)

        if count > TARGET_PER_CLASS:
            # Downsample majority classes without replacement.
            sampled = class_df.sample(
                withReplacement=False,
                fraction=TARGET_PER_CLASS / count,
                seed=SEED
            )
        else:
            # Oversample minority classes with replacement.
            sampled = class_df.sample(
                withReplacement=True,
                fraction=TARGET_PER_CLASS / count,
                seed=SEED
            )

        balanced_parts.append(sampled)

    balanced_fit = balanced_parts[0]

    for part in balanced_parts[1:]:
        balanced_fit = balanced_fit.unionByName(part)

    print("\nBalanced training class distribution (approximate):")

    balanced_fit.groupBy("attack_type").count().orderBy(
        F.desc("count")
    ).show(truncate=False)

    balanced_count = balanced_fit.count()
    print(f"Approximate balanced training rows: {balanced_count}")

    # --------------------------------------------------
    # 6. TRAIN LIGHTER RANDOM FOREST
    # --------------------------------------------------

    indexer = StringIndexer(
        inputCol="attack_type",
        outputCol="label",
        handleInvalid="skip",
        stringOrderType="frequencyDesc"
    )

    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=50,
        maxDepth=8,
        seed=SEED
    )

    pipeline = Pipeline(stages=[indexer, classifier])

    print("\nTraining balanced Random Forest...")

    model = pipeline.fit(balanced_fit)
    class_names = list(model.stages[0].labels)

    print("\nModel classes:")
    for name in class_names:
        print(f"  {name}")

    # --------------------------------------------------
    # 7. VALIDATION EVALUATION
    # --------------------------------------------------

    print("\nEvaluating validation data...")

    validation_predictions = model.transform(validation_data)

    validation_metrics = calculate_metrics(
        validation_predictions, class_names
    )

    print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"Validation macro-F1: {validation_metrics['macro_f1']:.4f}")
    print(f"Validation weighted-F1: {validation_metrics['weighted_f1']:.4f}")

    # --------------------------------------------------
    # 8. OFFICIAL TEST EVALUATION
    # --------------------------------------------------

    print("\nEvaluating official test data...")

    test_predictions = model.transform(test)
    test_metrics = calculate_metrics(test_predictions, class_names)

    print(f"Test records evaluated: {test_metrics['rows_evaluated']}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted-F1: {test_metrics['weighted_f1']:.4f}")

    print("\nPer-class test metrics:")

    for item in test_metrics["per_class"]:
        print(
            f"{item['attack_type']:18s} "
            f"Precision={item['precision']:.4f} "
            f"Recall={item['recall']:.4f} "
            f"F1={item['f1_score']:.4f} "
            f"Support={item['support']}"
        )

    # --------------------------------------------------
    # 9. SAVE MODEL AND RESULTS
    # --------------------------------------------------

    if MODEL_PATH.exists():
        shutil.rmtree(MODEL_PATH)

    model.write().overwrite().save(str(MODEL_PATH))
    print(f"\nSaved model: {MODEL_PATH}")

    save_json("stage2_balanced_validation_metrics.json", {
        "accuracy": validation_metrics["accuracy"],
        "macro_f1": validation_metrics["macro_f1"],
        "weighted_f1": validation_metrics["weighted_f1"],
        "rows_evaluated": validation_metrics["rows_evaluated"],
    })

    save_json("stage2_balanced_test_metrics.json", {
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "rows_evaluated": test_metrics["rows_evaluated"],
        "attack_classes": class_names,
        "training_method": "Approximate class-balanced sampling",
        "target_per_class": TARGET_PER_CLASS,
        "balanced_training_rows": balanced_count,
    })

    save_json(
        "stage2_balanced_per_class_metrics.json",
        test_metrics["per_class"]
    )

    save_json("stage2_balanced_confusion.json", {
        f"{actual},{predicted}": count
        for (actual, predicted), count in test_metrics["confusion"].items()
    })

    print("\nBalanced Stage 2 experiment finished successfully.")

finally:
    fit_data.unpersist() if "fit_data" in locals() else None
    validation_data.unpersist() if "validation_data" in locals() else None
    spark.stop()
