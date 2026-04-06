# Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/aadarshkarki/fraud-detection)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

End-to-end machine learning pipeline for detecting fraudulent credit card transactions. Compares **unsupervised anomaly detection** (Isolation Forest, Local Outlier Factor) against **supervised classification** (XGBoost baseline and tuned) across five performance metrics.

---

## 📌 Problem Statement

Credit card fraud is a critical financial risk — fraudulent transactions are rare (< 0.2% of all transactions), making class imbalance the core technical challenge. This project investigates whether anomaly detection or supervised learning better handles this imbalance when detecting fraud with minimal false negatives.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions · 492 fraud cases (0.172%)
- **Features:** 28 PCA-transformed features (V1–V28) + Time + Amount
- **Target:** `Class` — 0 (legitimate) / 1 (fraud)

> The dataset is not included in this repository due to size. Download `creditcard.csv` from Kaggle and place it in the project root.

---

## 🧠 Methodology

### Pipeline Overview

```
Raw Data
   │
   ├── Feature Scaling (StandardScaler on Amount & Time)
   ├── EDA Visualisations (fraud distribution, PCA clustering)
   │
   ├── Train/Test Split (70/30, stratified)
   │
   ├── SMOTE (applied to training set only)
   │
   ├── Model Training
   │     ├── Isolation Forest       (unsupervised)
   │     ├── Local Outlier Factor   (unsupervised)
   │     ├── XGBoost Baseline       (supervised)
   │     └── XGBoost Tuned          (RandomizedSearchCV, 50 iterations, 3-fold CV)
   │
   └── Evaluation
         ├── Accuracy, Precision, Recall, F1, ROC AUC
         ├── ROC Curves
         ├── Confusion Matrices
         ├── Feature Importance (Top 10)
         └── Radar Chart (multi-metric model comparison)
```

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| SMOTE on training set only | Prevents data leakage — test set preserves real-world class distribution |
| Stratified train/test split | Ensures fraud cases appear proportionally in both sets |
| Contamination = 0.0017 | Matches actual fraud rate in dataset for anomaly models |
| F1 as tuning metric | Balances precision and recall — critical for fraud where both FP and FN carry cost |
| RandomizedSearchCV (50 iter) | Efficient hyperparameter search over large continuous parameter space |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Isolation Forest | ~0.9983 | ~0.27 | ~0.28 | ~0.27 | ~0.64 |
| Local Outlier Factor | ~0.9965 | ~0.08 | ~0.08 | ~0.08 | ~0.54 |
| XGBoost (Baseline) | ~0.9996 | ~0.94 | ~0.82 | ~0.87 | ~0.98 |
| **XGBoost (Tuned)** | **~0.9997** | **~0.95** | **~0.84** | **~0.90** | **~0.98** |

> Exact values vary per run. Results above are representative of typical outputs on the standard Kaggle dataset.

### Key Findings

- **Supervised learning significantly outperforms anomaly detection** on this dataset — XGBoost (Tuned) achieves ~0.90 F1 vs ~0.27 for Isolation Forest
- **Anomaly detection struggles** because fraud patterns in this dataset are not purely anomalous — they overlap with legitimate transactions in feature space
- **SMOTE + XGBoost** is a robust combination for heavily imbalanced tabular fraud data
- **Hyperparameter tuning** delivers measurable improvement over baseline XGBoost across all metrics

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
├── creditcard fraud.py   # Full ML pipeline
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## ▶️ Run Online

Click below to run the full pipeline in your browser — no setup required:

[![Run on Kaggle](https://img.shields.io/badge/Run%20on-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/aadarshkarki/fraud-detection)

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/aadarsh1282/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 4. Run the pipeline

```bash
python "creditcard fraud.py"
```

The script will generate all visualisations and print the final model comparison table to console.

---

## 📊 Visualisations Generated

1. **Fraud Transaction Distribution** — histogram of fraud cases by scaled time
2. **KMeans Clustering (PCA)** — 2D scatter plot of transaction clusters
3. **Per-Metric Bar Charts** — model comparison across all 5 metrics
4. **ROC Curves** — all 4 models overlaid
5. **Confusion Matrices** — individual and stacked component view
6. **Feature Importance** — top 10 features from XGBoost (Tuned)
7. **Radar Chart** — multi-metric model comparison

---

## 🛠️ Technologies

| Category | Tools |
|----------|-------|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Visualisation | matplotlib, seaborn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Hyperparameter Tuning | RandomizedSearchCV |
| Dimensionality Reduction | PCA |

---

## 👤 Author

**Aadarsh Karki**
- 📧 [aadarshk56@gmail.com](mailto:aadarshk56@gmail.com)
- 🐙 [github.com/aadarsh1282](https://github.com/aadarsh1282)
- 🌐 [aadarsh1282.github.io/Aadarsh-Website](https://aadarsh1282.github.io/Aadarsh-Website/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
