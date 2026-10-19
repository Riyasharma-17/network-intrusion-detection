
import json
import csv
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Network Intrusion Detection Dashboard")
st.caption(
    "Large-Scale Network Intrusion Detection and Attack Profiling "
    "using PySpark | UNSW-NB15"
)


# --------------------------------------------------
# LOAD SAVED RESULTS — NO MODEL TRAINING
# --------------------------------------------------

@st.cache_data
def load_json(filename):
    path = RESULTS / filename
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_confusion_matrix(filename):
    path = RESULTS / filename
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if len(rows) < 2:
        return None

    headers = [item.strip() for item in rows[0]]
    data = rows[1:]

    if len(headers) != len(data[0]):
        return None

    df = pd.DataFrame(data, columns=headers)
    df = df.set_index(headers[0])

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def metric(data, *names):
    """Read a metric using common JSON key variations."""
    if not isinstance(data, dict):
        return None

    normalized = {
        str(key).lower().replace("-", "_").replace(" ", "_"): value
        for key, value in data.items()
    }

    for name in names:
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            try:
                return float(normalized[key])
            except (TypeError, ValueError):
                return None

    return None


def percentage(value):
    return f"{value * 100:.2f}%" if value is not None else "N/A"


def load_class_metrics(data):
    """Convert per-class JSON lists or dictionaries to a DataFrame."""
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get(
            "per_class",
            data.get("class_metrics", data.get("classes", [])),
        )

        if isinstance(records, dict):
            records = [
                {"attack_type": name, **values}
                for name, values in records.items()
                if isinstance(values, dict)
            ]
    else:
        return pd.DataFrame()

    if not isinstance(records, list):
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if "attack_type" not in df.columns:
        return pd.DataFrame()

    if "f1_score" not in df.columns and "f1" in df.columns:
        df["f1_score"] = df["f1"]

    for column in ["precision", "recall", "f1_score"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

stage1_comparison = load_json("stage1_model_comparison.json")
stage1_test = load_json("stage1_test_metrics.json")

stage2_baseline = load_json("stage2_test_metrics.json")
stage2_balanced = load_json("stage2_balanced_test_metrics.json")

baseline_classes = load_class_metrics(
    load_json("stage2_per_class_metrics.json")
)
balanced_classes = load_class_metrics(
    load_json("stage2_balanced_per_class_metrics.json")
)

confusion = load_confusion_matrix("stage2_confusion_matrix.csv")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Stage 1 — Binary Detection",
        "Stage 2 — Model Comparison",
        "Attack-wise Analysis",
        "Confusion Matrix",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Data source: locally saved experiment results")
st.sidebar.caption("Models are not retrained by this dashboard.")


# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

if page == "Project Overview":
    st.header("Project Overview")

    st.write(
        """
        This project uses Apache PySpark and machine learning to identify
        suspicious network traffic and classify detected attacks.

        **Stage 1:** Predict whether network traffic is normal or an attack.

        **Stage 2:** Classify attack traffic into one of nine attack categories.
        """
    )

    st.subheader("Two-stage detection pipeline")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Stage 1**\n\nBinary intrusion detection")

    with col2:
        st.info("**Stage 2**\n\nMulticlass attack profiling")

    with col3:
        st.info("**Evaluation**\n\nMetrics and error analysis")

    st.subheader("Stage 1 — Official test results")

    if isinstance(stage1_test, dict):
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            percentage(metric(stage1_test, "accuracy")),
        )
        c2.metric(
            "Attack Recall",
            percentage(metric(stage1_test, "recall_attack")),
        )
        c3.metric(
            "Attack Precision",
            percentage(metric(stage1_test, "precision_attack")),
        )
        c4.metric(
            "Test PR-AUC",
            f"{metric(stage1_test, 'test_pr_auc'):.4f}"
            if metric(stage1_test, "test_pr_auc") is not None
            else "N/A",
        )
    else:
        st.warning("Stage 1 test metrics file is missing.")

    st.subheader("Stage 2 — Baseline vs. balanced")

    if isinstance(stage2_baseline, dict) and isinstance(stage2_balanced, dict):
        base_f1 = metric(
            stage2_baseline, "macro_f1", "f1_macro", "macro_f1_score"
        )
        balanced_f1 = metric(
            stage2_balanced, "macro_f1", "f1_macro", "macro_f1_score"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Baseline Macro-F1",
            percentage(base_f1),
        )
        c2.metric(
            "Balanced Macro-F1",
            percentage(balanced_f1),
        )

        delta = (
            (balanced_f1 - base_f1) * 100
            if base_f1 is not None and balanced_f1 is not None
            else None
        )

        c3.metric(
            "Macro-F1 Change",
            f"{delta:+.2f} pp" if delta is not None else "N/A",
        )
    else:
        st.warning("Stage 2 comparison files are missing.")


# --------------------------------------------------
# STAGE 1 — BINARY DETECTION
# --------------------------------------------------

elif page == "Stage 1 — Binary Detection":
    st.header("Stage 1 — Binary Intrusion Detection")

    st.write(
        "Compare Logistic Regression, Random Forest and Gradient-Boosted Trees "
        "using their saved validation and evaluation results."
    )

    if isinstance(stage1_comparison, list) and stage1_comparison:
        df = pd.DataFrame(stage1_comparison)

        if "model" in df.columns:
            st.subheader("Validation PR-AUC")

            if "validation_pr_auc" in df.columns:
                chart = px.bar(
                    df,
                    x="model",
                    y="validation_pr_auc",
                    color="model",
                    text=df["validation_pr_auc"].map(lambda x: f"{x:.4f}"),
                    title="Model selection metric",
                    labels={
                        "model": "Model",
                        "validation_pr_auc": "Validation PR-AUC",
                    },
                )
                chart.update_layout(showlegend=False)
                chart.update_yaxes(range=[0, 1])
                st.plotly_chart(chart, use_container_width=True)

            metric_columns = [
                column
                for column in [
                    "accuracy",
                    "precision_attack",
                    "recall_attack",
                    "f1_attack",
                ]
                if column in df.columns
            ]

            if metric_columns:
                st.subheader("Model performance comparison")

                long_df = df.melt(
                    id_vars=["model"],
                    value_vars=metric_columns,
                    var_name="Metric",
                    value_name="Score",
                )

                chart = px.bar(
                    long_df,
                    x="model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Saved model evaluation metrics",
                    labels={"Score": "Score", "model": "Model"},
                )
                chart.update_yaxes(range=[0, 1])
                st.plotly_chart(chart, use_container_width=True)

            with st.expander("View complete model comparison data"):
                st.dataframe(df, use_container_width=True)

        else:
            st.warning("The model comparison JSON has no 'model' field.")
    else:
        st.warning("Stage 1 model comparison results are unavailable.")

    st.subheader("Selected model — Official test evaluation")

    if isinstance(stage1_test, dict):
        selected = stage1_test.get("selected_model", "Not specified")
        st.write(f"**Selected model:** {selected}")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", percentage(metric(stage1_test, "accuracy")))
        c2.metric(
            "Attack Precision",
            percentage(metric(stage1_test, "precision_attack")),
        )
        c3.metric(
            "Attack Recall",
            percentage(metric(stage1_test, "recall_attack")),
        )
        c4.metric(
            "Attack F1",
            percentage(metric(stage1_test, "f1_attack")),
        )

        with st.expander("View complete test metrics"):
            st.json(stage1_test)


# --------------------------------------------------
# STAGE 2 — BASELINE VS BALANCED
# --------------------------------------------------

elif page == "Stage 2 — Model Comparison":
    st.header("Stage 2 — Baseline vs. Balanced Random Forest")

    if not (
        isinstance(stage2_baseline, dict)
        and isinstance(stage2_balanced, dict)
    ):
        st.error("Could not load both Stage 2 metrics files.")
        st.stop()

    metric_definitions = [
        ("Accuracy", ("accuracy", "test_accuracy")),
        ("Macro-F1", ("macro_f1", "f1_macro", "macro_f1_score")),
        ("Weighted-F1", ("weighted_f1", "f1_weighted", "weighted_f1_score")),
    ]

    rows = []

    for label, keys in metric_definitions:
        baseline_value = metric(stage2_baseline, *keys)
        balanced_value = metric(stage2_balanced, *keys)

        rows.append({
            "Metric": label,
            "Baseline RF": baseline_value,
            "Balanced RF": balanced_value,
        })

    comparison_df = pd.DataFrame(rows)

    st.subheader("Key performance indicators")

    macro_row = comparison_df[comparison_df["Metric"] == "Macro-F1"].iloc[0]
    base_macro = macro_row["Baseline RF"]
    balanced_macro = macro_row["Balanced RF"]

    base_acc = metric(stage2_baseline, "accuracy")
    balanced_acc = metric(stage2_balanced, "accuracy")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Baseline Macro-F1",
        percentage(base_macro),
    )
    c2.metric(
        "Balanced Macro-F1",
        percentage(balanced_macro),
        delta=(
            f"{(balanced_macro - base_macro) * 100:+.2f} pp"
            if base_macro is not None and balanced_macro is not None
            else None
        ),
    )
    c3.metric(
        "Balanced Accuracy",
        percentage(balanced_acc),
        delta=(
            f"{(balanced_acc - base_acc) * 100:+.2f} pp"
            if base_acc is not None and balanced_acc is not None
            else None
        ),
    )

    st.subheader("Compare all three metrics")

    chart_df = comparison_df.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score",
    )

    chart = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=chart_df["Score"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
        ),
        title="Baseline vs. balanced model on the official test set",
    )
    chart.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(
        comparison_df.style.format(
            {"Baseline RF": "{:.2%}", "Balanced RF": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Balancing improved macro-F1 in this experiment, while accuracy and "
        "weighted-F1 decreased. Macro-F1 weights each attack category equally."
    )


# --------------------------------------------------
# ATTACK-WISE ANALYSIS
# --------------------------------------------------

elif page == "Attack-wise Analysis":
    st.header("Attack-wise Classification Analysis")

    if baseline_classes.empty or balanced_classes.empty:
        st.error("Could not load both per-attack metrics files.")
        st.stop()

    required = {"attack_type", "f1_score"}

    if not required.issubset(baseline_classes.columns) or not required.issubset(
        balanced_classes.columns
    ):
        st.error("The per-class JSON files do not contain attack_type and f1_score.")
        st.stop()

    merged = baseline_classes[["attack_type", "f1_score"]].rename(
        columns={"f1_score": "Baseline F1"}
    ).merge(
        balanced_classes[["attack_type", "f1_score"]].rename(
            columns={"f1_score": "Balanced F1"}
        ),
        on="attack_type",
        how="outer",
    )

    merged["Change (pp)"] = (
        merged["Balanced F1"] - merged["Baseline F1"]
    ) * 100

    merged = merged.sort_values("Change (pp)", ascending=True)

    st.write(
        "Compare per-category F1 scores to see which attack categories "
        "benefited from balancing."
    )

    chart_df = merged.melt(
        id_vars=["attack_type"],
        value_vars=["Baseline F1", "Balanced F1"],
        var_name="Model",
        value_name="F1 Score",
    )

    chart = px.bar(
        chart_df,
        x="F1 Score",
        y="attack_type",
        color="Model",
        barmode="group",
        orientation="h",
        title="F1 score by attack category",
        labels={
            "attack_type": "Attack category",
            "F1 Score": "F1 score",
        },
    )
    chart.update_xaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Change after balancing")

    change_chart = px.bar(
        merged,
        x="Change (pp)",
        y="attack_type",
        orientation="h",
        color="Change (pp)",
        color_continuous_scale="RdYlGn",
        title="F1 change by attack category (percentage points)",
        labels={
            "Change (pp)": "F1 change (percentage points)",
            "attack_type": "Attack category",
        },
    )
    change_chart.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(change_chart, use_container_width=True)

    display_df = merged.rename(columns={"attack_type": "Attack category"})
    st.dataframe(
        display_df.style.format({
            "Baseline F1": "{:.2%}",
            "Balanced F1": "{:.2%}",
            "Change (pp)": "{:+.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Positive change means the balanced model has a higher F1 score; "
        "negative change means its F1 score is lower."
    )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

elif page == "Confusion Matrix":
    st.header("Baseline Model — Confusion Matrix")

    if confusion is None:
        st.error("Could not load results/stage2_confusion_matrix.csv.")
        st.stop()

    st.write(
        "Rows represent actual attack categories; columns represent predicted "
        "categories. Larger off-diagonal values indicate more misclassifications."
    )

    numeric = confusion.apply(pd.to_numeric, errors="coerce").fillna(0)

    chart = px.imshow(
        numeric,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted category",
            "y": "Actual category",
            "color": "Record count",
        },
        title="Baseline Random Forest confusion matrix",
    )

    chart.update_layout(
        xaxis_title="Predicted category",
        yaxis_title="Actual category",
    )

    st.plotly_chart(chart, use_container_width=True)

    with st.expander("View confusion matrix as a table"):
        st.dataframe(confusion, use_container_width=True)

    st.caption(
        "This confusion matrix belongs to the baseline model. "
        "It is not the balanced model's confusion matrix."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Large-Scale Network Intrusion Detection using PySpark | "
    "Results loaded from local JSON and CSV files"
)
