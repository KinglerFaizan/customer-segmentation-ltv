"""
churn_model.py
--------------
Customer churn propensity scoring:
  - Binary classification: churned (no purchase in 90 days) vs. active
  - Logistic Regression, Random Forest, XGBoost
  - Threshold tuning for precision-recall tradeoff
  - SHAP explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ──────────────────────────────────────────────
# Label Generation
# ──────────────────────────────────────────────

def create_churn_labels(df: pd.DataFrame, churn_days: int = 90) -> pd.DataFrame:
    """
    Create binary churn label per customer.
    Churned = no purchase in the last `churn_days` days of the observation window.

    Args:
        df: Cleaned transaction DataFrame
        churn_days: Inactivity threshold in days

    Returns:
        DataFrame with CustomerID and Churned (0/1) columns
    """
    snapshot_date = df['InvoiceDate'].max()
    cutoff_date = snapshot_date - timedelta(days=churn_days)

    last_purchase = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    last_purchase.columns = ['CustomerID', 'LastPurchase']
    last_purchase['Churned'] = (last_purchase['LastPurchase'] < cutoff_date).astype(int)

    churn_rate = last_purchase['Churned'].mean()
    print(f"Churn Definition: no purchase in last {churn_days} days.")
    print(f"Churn Rate: {churn_rate:.1%} ({last_purchase['Churned'].sum():,} / {len(last_purchase):,} customers)")

    return last_purchase[['CustomerID', 'Churned']]


# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

CHURN_FEATURES = [
    'Recency', 'Frequency', 'Monetary',
    'total_orders', 'unique_products', 'unique_categories',
    'avg_order_value', 'std_order_value',
    'customer_age_days', 'purchase_span_days',
    'avg_days_between_orders', 'purchase_frequency_per_month',
    'product_diversity_ratio', 'is_multi_country',
]


def prepare_churn_dataset(features_df: pd.DataFrame, churn_labels: pd.DataFrame,
                           test_size: float = 0.2, random_state: int = 42):
    """Merge features with churn labels and split into train/test."""
    data = features_df.merge(churn_labels, on='CustomerID', how='inner')

    available_features = [c for c in CHURN_FEATURES if c in data.columns]
    X = data[available_features].fillna(0)
    y = data['Churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, available_features


def train_churn_models(X_train, X_test, y_train, y_test,
                        experiment_name: str = 'Churn_Modeling') -> dict:
    """
    Train and evaluate churn models with MLflow tracking.
    """
    mlflow.set_experiment(experiment_name)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'LogisticRegression': (
            LogisticRegression(C=1.0, max_iter=500, random_state=42),
            X_train_scaled, X_test_scaled
        ),
        'RandomForest': (
            RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            X_train, X_test
        ),
        'XGBoost': (
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
                random_state=42, verbosity=0, n_jobs=-1, eval_metric='auc'
            ),
            X_train, X_test
        ),
    }

    results = {}
    best_model = None
    best_auc = 0.0

    for name, (model, X_tr, X_te) in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_train)
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = model.predict(X_te)

            auc = roc_auc_score(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)

            print(f"\n{'='*40}")
            print(f"{name}")
            print(f"  AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))

            mlflow.log_metric('auc_roc', auc)
            mlflow.log_metric('avg_precision', ap)
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

            results[name] = {
                'model': model, 'auc': auc, 'ap': ap,
                'y_prob': y_prob, 'y_pred': y_pred,
            }

            if auc > best_auc:
                best_auc = auc
                best_model = (name, model)

    print(f"\n✅ Best churn model: {best_model[0]} (AUC: {best_auc:.4f})")
    return results, best_model, scaler


# ──────────────────────────────────────────────
# Threshold Tuning
# ──────────────────────────────────────────────

def tune_threshold(y_test, y_prob, target_recall: float = 0.75) -> float:
    """
    Find the probability threshold that achieves at least target_recall
    while maximizing precision.

    Returns:
        Optimal threshold float
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    valid = recall >= target_recall
    if not valid.any():
        return 0.5

    best_idx = np.argmax(precision[valid])
    best_threshold = thresholds[valid[:-1]][best_idx]

    print(f"Threshold tuned for Recall ≥ {target_recall:.0%}: "
          f"threshold={best_threshold:.3f}, precision={precision[valid][best_idx]:.3f}")
    return float(best_threshold)


# ──────────────────────────────────────────────
# SHAP Explainability
# ──────────────────────────────────────────────

def explain_with_shap(model, X_test, feature_names: list, max_display: int = 15):
    """Generate SHAP summary plot for model explainability."""
    if not SHAP_AVAILABLE:
        print("SHAP not installed. Run: pip install shap")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (Churned) SHAP values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig('../outputs/shap_churn.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("SHAP plot saved.")


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_roc_curves(results: dict, y_test):
    """Plot ROC curves for all trained churn models."""
    plt.figure(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Churn Model ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../outputs/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def score_all_customers(features_df: pd.DataFrame, model, scaler,
                         feature_cols: list, threshold: float = 0.5) -> pd.DataFrame:
    """
    Score all customers with churn probability and risk tier.
    """
    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)

    try:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        probs = model.predict_proba(X)[:, 1]

    features_df = features_df.copy()
    features_df['Churn_Probability'] = probs
    features_df['Churn_Predicted'] = (probs >= threshold).astype(int)
    features_df['Churn_Risk_Tier'] = pd.cut(
        probs,
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low', 'Medium', 'High', 'Critical']
    )

    print("\nChurn Risk Distribution:")
    print(features_df['Churn_Risk_Tier'].value_counts())
    return features_df


if __name__ == '__main__':
    print("Churn model module loaded. Run notebooks for full pipeline.")
