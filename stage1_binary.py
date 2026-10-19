
"""Stage 1: Binary network intrusion detection using Spark MLlib."""

import json
import time
from pathlib import Path

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed_parquet" / "train"
TEST_PATH = ROOT / "data" / "processed_parquet" / "test"
MODEL_DIR = ROOT / "saved_models" / "stage1_binary"
RESULTS_DIR = ROOT / "results"
SEED = 42


def binary_metrics(predictions):
    """Calculate binary classification metrics with Spark aggregations."""
    counts = predictions.agg(
        F.sum(
            F.when(
                (F.col("label") == 1) & (F.col("prediction") == 1), 1
            ).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when(
                (F.col("label") == 0) & (F.col("prediction") == 0), 1
            ).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when(
                (F.col("label") == 0) & (F.col("prediction") == 1), 1
            ).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when(
                (F.col("label") == 1) & (F.col("prediction") == 0), 1
            ).otherwise(0)
        ).alias("fn"),
    ).first()

    tp, tn, fp, fn = (int(counts[k] or 0) for k in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision_attack": precision,
        "recall_attack": recall,
        "specificity": specificity,
        "f1_attack": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def main():
    spark = (
        SparkSession.builder
        .appName("Stage1BinaryIntrusionDetection")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        train = spark.read.parquet(str(TRAIN_PATH)).select(
            "features", F.col("is_attack").cast("double").alias("label")
        ).cache()
        test = spark.read.parquet(str(TEST_PATH)).select(
            "features", F.col("is_attack").cast("double").alias("label")
        ).cache()

        # Only split the training partition; keep the official test untouched.
        fit_data, validation_data = train.randomSplit([0.8, 0.2], seed=SEED)
        fit_data.cache()
        validation_data.cache()

        print(f"Fit rows: {fit_data.count():,}")
        print(f"Validation rows: {validation_data.count():,}")
        print(f"Official test rows: {test.count():,}")

        candidates = [
            LogisticRegression(
                featuresCol="features", labelCol="label",
                maxIter=60, regParam=0.01
            ),
            RandomForestClassifier(
                featuresCol="features", labelCol="label",
                numTrees=80, maxDepth=10, seed=SEED
            ),
            GBTClassifier(
                featuresCol="features", labelCol="label",
                maxIter=40, maxDepth=5, seed=SEED
            ),
        ]

        evaluator = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        results = []
        best_model = None
        best_score = -1.0

        for estimator in candidates:
            name = estimator.__class__.__name__
            print(f"\nTraining {name}...")
            started = time.perf_counter()
            model = estimator.fit(fit_data)
            training_seconds = time.perf_counter() - started

            predictions = model.transform(validation_data).cache()
            metrics = binary_metrics(predictions)
            pr_auc = evaluator.evaluate(predictions)

            row = {
                "model": name,
                "training_seconds": round(training_seconds, 2),
                "validation_pr_auc": pr_auc,
                **metrics,
            }
            results.append(row)

            print(
                f"{name}: PR-AUC={pr_auc:.4f}, "
                f"attack recall={metrics['recall_attack']:.4f}, "
                f"attack F1={metrics['f1_attack']:.4f}"
            )

            # Use PR-AUC for model selection; report recall alongside it.
            if pr_auc > best_score:
                best_score = pr_auc
                best_model = model

            predictions.unpersist()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

        with open(RESULTS_DIR / "stage1_model_comparison.json", "w") as f:
            json.dump(results, f, indent=2)

        # Evaluate the selected model on the untouched test set only now.
        print("\nEvaluating selected model on the official test set...")
        test_predictions = best_model.transform(test).cache()
        test_metrics = binary_metrics(test_predictions)
        test_pr_auc = evaluator.evaluate(test_predictions)

        test_report = {
            "selected_model": best_model.__class__.__name__,
            "selection_metric": "validation PR-AUC",
            "validation_pr_auc": best_score,
            "test_pr_auc": test_pr_auc,
            **test_metrics,
        }

        best_model.write().overwrite().save(str(MODEL_DIR))
        with open(RESULTS_DIR / "stage1_test_metrics.json", "w") as f:
            json.dump(test_report, f, indent=2)

        print("\n=== STAGE 1 TEST RESULTS ===")
        print(json.dumps(test_report, indent=2))
        print(f"\nModel saved to: {MODEL_DIR}")
        print(f"Results saved to: {RESULTS_DIR}")

        test_predictions.unpersist()
        train.unpersist()
        test.unpersist()
        fit_data.unpersist()
        validation_data.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
