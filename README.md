# Large-Scale Network Intrusion Detection using PySpark

A two-stage machine learning pipeline for detecting network intrusions and profiling cyberattacks using the **UNSW-NB15 dataset**, Apache Spark, and PySpark MLlib.

## Overview

Traditional intrusion detection systems can struggle with large volumes of network traffic and imbalanced attack categories. This project uses distributed data processing and machine learning to identify suspicious traffic and classify attack types.

## How It Works

**Stage 1 — Binary Intrusion Detection**
- Classifies network traffic as Normal or Attack.
- Compares Logistic Regression, Random Forest, and Gradient-Boosted Trees.
- Selects the model using validation PR-AUC.

**Stage 2 — Multiclass Attack Profiling**
- Classifies attack traffic into nine categories: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, and Worms.
- Compares a baseline Random Forest with a class-balanced training experiment.

**Interactive Dashboard**
- Visualizes model performance, attack-wise F1 scores, and the baseline confusion matrix.
- Loads saved results without retraining models.

## Tech Stack

- **Language:** Python
- **Big Data:** Apache Spark, PySpark
- **Machine Learning:** Spark MLlib
- **Dashboard:** Streamlit, Plotly
- **Data Analysis:** Pandas
- **Dataset:** UNSW-NB15

## Results

| Metric | Baseline Stage 2 | Balanced Stage 2 |
|---|---:|---:|
| Accuracy | 69.84% | 55.77% |
| Macro-F1 | 34.47% | **40.37%** |
| Weighted-F1 | 64.88% | 60.53% |

Balancing improved macro-F1 by **5.90 percentage points**, indicating better average performance across attack categories, while reducing overall accuracy.

Stage 1 selected the Gradient-Boosted Tree model using validation PR-AUC of **0.9738**. Its official test attack recall was **97.09%**, with a false-positive rate of **33.16%**.

## Project Structure

```text
network-intrusion-detection/
├── data/                   # Local datasets (not committed)
├── saved_models/           # Trained models (not committed)
├── results/                # Evaluation metrics and confusion matrix
├── preprocess.py           # Data preprocessing pipeline
├── stage1_binary.py        # Binary classification
├── stage2_multiclass.py    # Baseline attack classification
├── stage2_balanced.py      # Class-balanced experiment
├── show_results.py         # Terminal results summary
├── dashboard.py            # Interactive dashboard
├── requirements.txt
└── README.md
```

## Setup and Run

Requires Python 3.11, Java 17, and a compatible PySpark environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline scripts in order, using the locally configured dataset and environment:

```bash
python preprocess.py
python stage1_binary.py
python stage2_multiclass.py
python stage2_balanced.py
```

Display saved evaluation results:

```bash
python show_results.py
```

Launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dataset and trained model artifacts are excluded from Git. Configure the local dataset paths and environment before running the training pipeline.

## Key Takeaway

This project demonstrates a two-stage intrusion detection workflow using PySpark, model comparison, class-imbalance experimentation, and interactive evaluation. The results highlight the trade-off between overall accuracy and performance across rare attack categories.

## Author

**Riya Sharma**  
GitHub: [@Riyasharma-17](https://github.com/Riyasharma-17)
