
import json
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_json(filename):
    path = RESULTS / filename

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_confusion_matrix(filename):
    path = RESULTS / filename

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


def get_metric(data, *possible_names):
    """Find a metric even if its JSON key uses a different naming style."""
    if not isinstance(data, dict):
        return None

    normalized = {
        str(key).lower().replace("-", "_").replace(" ", "_"): value
        for key, value in data.items()
    }

    for name in possible_names:
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


def fmt(value, percent=False):
    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if percent:
        return f"{value * 100:.2f}%"

    return f"{value:.4f}"


def find_class_metrics(data):
    """Support list-based and dictionary-based per-class JSON files."""
    if isinstance(data, list):
        return {
            item["attack_type"]: item
            for item in data
            if isinstance(item, dict) and "attack_type" in item
        }

    if not isinstance(data, dict):
        return {}

    if data and all(isinstance(v, dict) for v in data.values()):
        if any(
            get_metric(v, "f1", "f1_score", "f1-score") is not None
            for v in data.values()
        ):
            return data

    for key in ("per_class", "class_metrics", "classes", "per_class_metrics"):
        nested = data.get(key)
        if isinstance(nested, (dict, list)):
            return find_class_metrics(nested)

    return {}


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_stage1():
    comparison = load_json("stage1_model_comparison.json")
    metrics = load_json("stage1_test_metrics.json")

    print_section("STAGE 1 — BINARY INTRUSION DETECTION")

    if comparison is None:
        print("Stage 1 model comparison file not found.")
    else:
        print("\nModel comparison:")
        print(json.dumps(comparison, indent=2))

    if metrics is None:
        print("\nStage 1 test metrics file not found.")
    else:
        print("\nOfficial test metrics:")
        print(json.dumps(metrics, indent=2))


def print_stage2_comparison():
    baseline = load_json("stage2_test_metrics.json")
    balanced = load_json("stage2_balanced_test_metrics.json")

    print_section("STAGE 2 — BASELINE VS BALANCED MODEL")

    if baseline is None:
        print("Missing results/stage2_test_metrics.json")
        return

    if balanced is None:
        print("Missing results/stage2_balanced_test_metrics.json")
        print("Run stage2_balanced.py first.")
        return

    metrics_to_compare = [
        ("Accuracy", ("accuracy", "test_accuracy"), True),
        ("Macro-F1", ("macro_f1", "f1_macro", "macro_f1_score"), True),
        ("Weighted-F1", ("weighted_f1", "f1_weighted", "weighted_f1_score"), True),
    ]

    print(f"\n{'Metric':<22}{'Baseline RF':>18}{'Balanced RF':>18}{'Change':>18}")
    print("-" * 76)

    for label, keys, is_ratio in metrics_to_compare:
        old = get_metric(baseline, *keys)
        new = get_metric(balanced, *keys)

        if old is None or new is None:
            print(f"{label:<22}{fmt(old, is_ratio):>18}{fmt(new, is_ratio):>18}{'N/A':>18}")
            continue

        change = (float(new) - float(old)) * (100 if is_ratio else 1)
        suffix = " pp" if is_ratio else ""
        print(
            f"{label:<22}{fmt(old, is_ratio):>18}"
            f"{fmt(new, is_ratio):>18}{f'{change:+.2f} pp' if is_ratio else f'{change:+.2f}':>18}"
        )

    old_macro = get_metric(
        baseline, "macro_f1", "f1_macro", "macro_f1_score"
    )
    new_macro = get_metric(
        balanced, "macro_f1", "f1_macro", "macro_f1_score"
    )

    if old_macro is not None and new_macro is not None:
        change = (float(new_macro) - float(old_macro)) * 100
        print("\nInterpretation:")
        if change > 0:
            print(
                f"Balancing improved macro-F1 by {change:.2f} percentage "
                "points."
            )
        elif change < 0:
            print(
                f"Balancing reduced macro-F1 by {abs(change):.2f} "
                "percentage points."
            )
        else:
            print("Macro-F1 remained unchanged.")

    print("\nWhat the trade-off means:")
    print("- Accuracy measures the fraction of correctly classified records.")
    print("- Macro-F1 gives each attack category equal weight.")
    print("- Weighted-F1 gives more weight to categories with more records.")
    print("- A balanced model may improve rare attacks while reducing accuracy.")


def print_per_class_comparison():
    baseline_data = load_json("stage2_per_class_metrics.json")
    balanced_data = load_json("stage2_balanced_per_class_metrics.json")

    print_section("PER-ATTACK F1 COMPARISON")

    if baseline_data is None or balanced_data is None:
        print("Could not find both per-class metrics JSON files.")
        return

    baseline_classes = find_class_metrics(baseline_data)
    balanced_classes = find_class_metrics(balanced_data)

    if not baseline_classes or not balanced_classes:
        print("Could not recognize the per-class JSON structure.")
        print("Baseline JSON structure:")
        print(json.dumps(baseline_data, indent=2)[:2500])
        print("\nBalanced JSON structure:")
        print(json.dumps(balanced_data, indent=2)[:2500])
        return

    class_names = sorted(set(baseline_classes) | set(balanced_classes))

    print(f"\n{'Attack category':<22}{'Baseline F1':>16}{'Balanced F1':>16}{'Change':>16}")
    print("-" * 70)

    for name in class_names:
        old = get_metric(
            baseline_classes.get(name), "f1", "f1_score", "f1-score"
        )
        new = get_metric(
            balanced_classes.get(name), "f1", "f1_score", "f1-score"
        )

        if old is None or new is None:
            print(f"{name:<22}{fmt(old, True):>16}{fmt(new, True):>16}{'N/A':>16}")
            continue

        change = (float(new) - float(old)) * 100

        print(
            f"{name:<22}{fmt(old, True):>16}{fmt(new, True):>16}"
            f"{f'{change:+.2f} pp':>16}"
        )


def print_confusion_matrix():
    print_section("BASELINE CONFUSION MATRIX")

    rows = load_confusion_matrix("stage2_confusion_matrix.csv")

    if not rows:
        print("Baseline confusion matrix CSV not found.")
        return

    for row in rows:
        print(" | ".join(row))


def main():
    print("\nLARGE-SCALE NETWORK INTRUSION DETECTION")
    print("Results Dashboard — Existing saved results only")

    print_stage1()
    print_stage2_comparison()
    print_per_class_comparison()
    print_confusion_matrix()

    print_section("DASHBOARD COMPLETE")
    print("No models were retrained. Results were read from the results/ folder.")


if __name__ == "__main__":
    main()
