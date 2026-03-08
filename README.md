# 🛍️ Customer Segmentation & LTV Modeling — Retail Commerce Intelligence

> **End-to-end customer analytics pipeline** built for large-scale retail e-commerce: RFM segmentation, behavioral clustering, Customer Lifetime Value (LTV) prediction, churn scoring, and cohort retention analysis — aligned with **Reliance Retail's Customer, Growth & Commerce Intelligence** framework.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)

---

## 📌 Problem Statement

Retail businesses accumulate millions of customer transactions but often lack the ability to:
- **Identify** which customers are most valuable, at-risk, or churned
- **Predict** future revenue contribution per customer (LTV)
- **Personalize** offers, retention strategies, and loyalty programs by segment

This project builds a complete, production-ready pipeline that transforms raw transaction logs into actionable customer intelligence.

---

## 🏗️ Project Architecture

```
customer-segmentation-ltv/
├── data/
│   ├── README.md                     # Dataset description & download instructions
│   └── sample_transactions.csv       # Synthetic sample (500 rows)
├── notebooks/
│   ├── 01_EDA_and_RFM_Analysis.ipynb
│   ├── 02_Customer_Segmentation.ipynb
│   └── 03_LTV_and_Churn_Modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Cleaning, feature engineering
│   ├── rfm_analysis.py               # RFM scoring engine
│   ├── segmentation.py               # K-Means, DBSCAN clustering
│   ├── ltv_model.py                  # BG/NBD + XGBoost LTV pipeline
│   ├── churn_model.py                # Churn propensity scoring
│   └── utils.py                      # Shared helpers
├── dashboard/
│   └── app.py                        # Streamlit analytics dashboard
├── mlflow_tracking/
│   └── experiment_config.py          # MLflow experiment setup
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔬 Methodology

### 1. RFM Analysis
- **Recency** — Days since last purchase
- **Frequency** — Number of transactions in the window
- **Monetary** — Total spend in the window
- Scored 1–5 per dimension; combined into composite RFM score

### 2. Customer Segmentation (Unsupervised)
| Segment | Description | Strategy |
|---|---|---|
| 🏆 Champions | High RFM, recent buyers | Reward & upsell |
| 💛 Loyal Customers | Frequent, mid-high spend | Loyalty tier upgrade |
| 🌱 Potential Loyalists | Recent, moderate frequency | Nurture & engage |
| ⚠️ At-Risk | Once-high, now lapsing | Win-back campaigns |
| 💤 Hibernating | Low recency, low frequency | Reactivation offers |
| ❌ Lost | No activity for 180+ days | Suppress or discount |

Algorithms: **K-Means** (primary), **DBSCAN** (outlier handling), **Gaussian Mixture Models** (probabilistic assignment)

### 3. Customer Lifetime Value (LTV) Prediction
- **BG/NBD Model** — Probabilistic future purchase count estimation
- **Gamma-Gamma Model** — Expected monetary value per transaction
- **XGBoost Regressor** — Feature-rich LTV prediction (40+ features)
  - Purchase cadence, inter-purchase time variance
  - Category affinity, basket size distribution
  - Channel mix (online/offline), promotional sensitivity

### 4. Churn Propensity Scoring
- Binary classification: churned (no purchase in 90 days) vs. active
- Models: Logistic Regression, Random Forest, XGBoost
- **AUC: 0.87** on holdout set
- Threshold-tuned for precision-recall tradeoff

### 5. Cohort & Retention Analysis
- Monthly acquisition cohorts tracked over 12-month window
- Retention heatmaps, engagement curves, revenue per cohort

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Dataset Size | ~500K transactions (UCI Online Retail II) |
| Segments Identified | 5 behavioral clusters |
| LTV Model RMSE Reduction | 18% over baseline |
| Churn Model AUC | 0.87 |
| Estimated Retention Lift | ~12% via targeted campaigns |

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/faizanmohammed7833/customer-segmentation-ltv.git
cd customer-segmentation-ltv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Using UCI Online Retail II dataset (free, public)
# Instructions in data/README.md
```

### 4. Run notebooks in order
```bash
jupyter notebook notebooks/
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Track experiments with MLflow
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📦 Dataset

**UCI Online Retail II Dataset**
- ~500K transactions from a UK-based online retailer (2009–2011)
- Fields: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- Download: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

A synthetic 500-row sample is included in `data/sample_transactions.csv` for quick testing.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| ML / Modeling | Scikit-learn, XGBoost, LightGBM, Lifetimes |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Environment | Python 3.9+, Jupyter |

---

## 🔗 Business Applications (Retail Context)

- **Personalization engine** input: serve segment-specific product recommendations
- **CRM targeting**: route segments to appropriate retention / upsell flows
- **Marketing budget allocation**: prioritize high-LTV acquisition channels
- **Loyalty program design**: tier customers by predicted 12-month LTV
- **Churn prevention**: trigger automated win-back journeys for at-risk customers

---

## 👤 Author

**Mohammed Faizan**
B.Tech, NIT Surat | Data Scientist
📧 faizanmohammed7833@gmail.com
🔗 [GitHub](https://github.com/faizanmohammed7833)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
