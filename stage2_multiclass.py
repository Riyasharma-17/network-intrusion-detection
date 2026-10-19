
import json
import csv
from pathlib import Path

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

MODEL_PATH = str(ROOT / "saved_models" / "stage2_multiclass")
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# --------------------------------------------------
# 2. START SPARK
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("NetworkIntrusionStage2")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def save_json(filename, content):
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4)
    print(f"Saved: {path}")


def prepare_attack_data(df):
    """Keep attack records with valid, known attack categories."""
    return df.filter(
        (F.col("is_attack") == 1)
        & F.col("attack_type").isNotNull()
        & (F.trim(F.col("attack_type")) != "")
        & (~F.lower(F.trim(F.col("attack_type"))).isin(
            "normal", "unknown", "none"
        ))
    ).select("features", "attack_type")


def confusion_data(predictions):
    """Collect only the small class-level confusion table."""
    return (
        predictions
        .groupBy("label", "prediction")
        .count()
        .collect()
    )


def calculate_metrics(predictions, class_names):
    """
    Calculate accuracy, macro-F1, weighted-F1,
    and per-class precision/recall/F1.
    """
    rows = confusion_data(predictions)
    matrix = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in rows
    }

    total = sum(matrix.values())
    correct = sum(
        count for (actual, predicted), count in matrix.items()
        if actual == predicted
    )
    accuracy = correct / total if total else 0.0

    per_class = []
    f1_values = []
    weighted_f1_sum = 0.0

    for index, name in enumerate(class_names):
        tp = matrix.get((index, index), 0)
        actual_count = sum(
            count for (actual, _), count in matrix.items()
            if actual == index
        )
        predicted_count = sum(
            count for (_, predicted), count in matrix.items()
            if predicted == index
        )

        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / actual_count if actual_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class.append({
            "attack_type": name,
            "support": actual_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        f1_values.append(f1)
        weighted_f1_sum += f1 * actual_count

    macro_f1 = (
        sum(f1_values) / len(f1_values) if f1_values else 0.0
    )
    weighted_f1 = weighted_f1_sum / total if total else 0.0

    return {
        "rows_evaluated": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion": matrix,
    }


# --------------------------------------------------
# 3. LOAD PROCESSED DATA
# --------------------------------------------------

print("\nLoading processed datasets...")

train_all = spark.read.parquet(TRAIN_PATH)
test_all = spark.read.parquet(TEST_PATH)

train_attacks = prepare_attack_data(train_all)
test_attacks = prepare_attack_data(test_all)

print(f"Training attack records: {train_attacks.count()}")
print(f"Official test attack records: {test_attacks.count()}")

print("\nTraining attack category distribution:")
train_attacks.groupBy("attack_type").count().orderBy(
    F.desc("count")
).show(truncate=False)


# --------------------------------------------------
# 4. SPLIT TRAINING DATA
#    Official test data remains untouched until
#    final evaluation.
# --------------------------------------------------

fit_data, validation_data = train_attacks.randomSplit(
    [0.8, 0.2], seed=SEED
)

fit_data = fit_data.cache()
validation_data = validation_data.cache()

print(f"Fit records: {fit_data.count()}")
print(f"Validation records: {validation_data.count()}")


# --------------------------------------------------
# 5. BUILD MULTICLASS CLASSIFIER
# --------------------------------------------------

label_indexer = StringIndexer(
    inputCol="attack_type",
    outputCol="label",
    handleInvalid="skip",
    stringOrderType="frequencyDesc"
)

classifier = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    numTrees=100,
    maxDepth=12,
    seed=SEED
)

pipeline = Pipeline(stages=[label_indexer, classifier])

print("\nTraining Stage 2 Random Forest...")
model = pipeline.fit(fit_data)

# Indexer is learned only from the fit subset.
class_names = list(model.stages[0].labels)

print("\nLearned attack classes:")
for name in class_names:
    print(f" - {name}")


# --------------------------------------------------
# 6. VALIDATION EVALUATION
# --------------------------------------------------

print("\nEvaluating on validation data...")

validation_predictions = model.transform(validation_data)
validation_metrics = calculate_metrics(
    validation_predictions, class_names
)

print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
print(f"Validation macro-F1: {validation_metrics['macro_f1']:.4f}")
print(f"Validation weighted-F1: {validation_metrics['weighted_f1']:.4f}")


# --------------------------------------------------
# 7. FINAL EVALUATION ON OFFICIAL TEST DATA
# --------------------------------------------------

print("\nEvaluating on official test attack records...")

test_predictions = model.transform(test_attacks)
test_metrics = calculate_metrics(test_predictions, class_names)

print(f"Test records evaluated: {test_metrics['rows_evaluated']}")
print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
print(f"Test weighted-F1: {test_metrics['weighted_f1']:.4f}")

print("\nPer-class results:")
for item in test_metrics["per_class"]:
    print(
        f"{item['attack_type']:18s} "
        f"Precision={item['precision']:.4f} "
        f"Recall={item['recall']:.4f} "
        f"F1={item['f1_score']:.4f} "
        f"Support={item['support']}"
    )


# --------------------------------------------------
# 8. SAVE MODEL AND RESULTS
# --------------------------------------------------

# Avoid accidentally overwriting a previous saved model.
if Path(MODEL_PATH).exists():
    import shutil
    shutil.rmtree(MODEL_PATH)

model.write().overwrite().save(MODEL_PATH)
print(f"\nSaved model: {MODEL_PATH}")

save_json("stage2_validation_metrics.json", {
    "accuracy": validation_metrics["accuracy"],
    "macro_f1": validation_metrics["macro_f1"],
    "weighted_f1": validation_metrics["weighted_f1"],
    "rows_evaluated": validation_metrics["rows_evaluated"],
})

save_json("stage2_test_metrics.json", {
    "accuracy": test_metrics["accuracy"],
    "macro_f1": test_metrics["macro_f1"],
    "weighted_f1": test_metrics["weighted_f1"],
    "rows_evaluated": test_metrics["rows_evaluated"],
    "attack_classes": class_names,
})

save_json("stage2_per_class_metrics.json", test_metrics["per_class"])

# Save a readable confusion matrix CSV.
matrix = test_metrics["confusion"]
csv_path = RESULTS_DIR / "stage2_confusion_matrix.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual / Predicted"] + class_names)

    for actual_index, actual_name in enumerate(class_names):
        writer.writerow([
            actual_name,
            *[
                matrix.get((actual_index, predicted_index), 0)
                for predicted_index in range(len(class_names))
            ]
        ])

print(f"Saved: {csv_path}")

fit_data.unpersist()
validation_data.unpersist()

spark.stop()
print("\nStage 2 finished.")
