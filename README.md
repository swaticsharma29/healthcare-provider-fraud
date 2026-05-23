# Healthcare Provider Fraud Detection

A self-initiated end-to-end data science case study on detecting fraudulent Medicare providers using supervised machine learning in Python.

---

## Problem

Medicare fraud costs the US government billions annually. Fraudulent providers manipulate claims through tactics like billing for services never rendered, duplicate claim submission, upcoding procedures, and misrepresenting diagnoses. The goal of this project is to **predict whether a healthcare provider is potentially fraudulent** based on patterns in their inpatient, outpatient, and beneficiary claims data.

---

## Dataset

Source: [Kaggle — Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

Three linked datasets:

| Dataset | Description |
|---|---|
| Inpatient Claims | Claims from admitted patients — includes admission/discharge dates, diagnosis codes |
| Outpatient Claims | Claims from non-admitted patients visiting hospitals |
| Beneficiary Details | Patient KYC data — health conditions, demographics, region |

The datasets are joined on provider and beneficiary IDs to build a unified feature set per provider.

---

## Approach

### 1. Exploratory Data Analysis (`EDA_Frauds.ipynb`)
- Distribution analysis of claim amounts, diagnosis codes, and procedure codes
- Correlation between chronic conditions and fraud labels
- Visualising patterns in admission duration, reimbursement amounts, and physician involvement

### 2. Feature Engineering (`Self_Case_Study_1.ipynb`)
- Aggregated provider-level features: total claims, average reimbursement, number of unique patients, physicians, and diagnosis codes
- Derived features: claim duration, age from DOB/DOD, chronic condition flags
- Merged inpatient and outpatient data into a single provider-level feature matrix

### 3. Modelling
Binary classification: `PotentialFraud = Yes / No`

Models evaluated:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosted Trees (XGBoost)

Evaluation metrics: Precision, Recall, F1-score, ROC-AUC (prioritising Recall — missing a fraudulent provider is costly)

---

## Tech Stack

- **Python 3** — pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebook** — EDA and modelling
- **Git** — version control

---

## Key Findings

- Providers with a high number of unique physicians per claim and unusually short inpatient durations showed strong correlation with fraud labels
- Chronic condition combinations (renal disease + depression) appeared disproportionately in fraudulent claims
- Random Forest and Gradient Boosting significantly outperformed the Logistic Regression baseline on Recall

---

## Files

| File | Description |
|---|---|
| `EDA_Frauds.ipynb` | Exploratory data analysis and visualisations |
| `Self_Case_Study_1.ipynb` | Feature engineering, model training, and evaluation |
| `EDA_Frauds.pdf` | PDF export of EDA notebook |
| `Healthcare Provider Fraud Detection Analysis.pdf` | Full case study report |

---

## Why This Project

Fraud detection is a pattern-recognition problem at its core — the same principles (anomaly signals in structured data, imbalanced classification, feature aggregation across relational tables) apply directly to financial data, investment risk, and portfolio analytics. This case study was self-initiated to develop intuition for data-heavy, high-stakes classification problems.

---

*Part of a broader portfolio of ML case studies: [Applied AI Assignments & Case Studies](https://github.com/swaticsharma29/Applied-AI-Assignments-Case-Studies)*
