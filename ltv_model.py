"""
ltv_model.py
------------
Customer Lifetime Value (LTV) prediction using:
  1. BG/NBD + Gamma-Gamma probabilistic model (lifetimes library)
  2. XGBoost regression on engineered behavioral features
  3. MLflow experiment tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False
    print("Warning: 'lifetimes' not installed. BG/NBD model unavailable. Run: pip install lifetimes")


# ──────────────────────────────────────────────
# BG/NBD + Gamma-Gamma (Probabilistic LTV)
# ──────────────────────────────────────────────

def fit_bgnbd(df: pd.DataFrame, observation_period_end: str = None) -> dict:
    """
    Fit BG/NBD model to predict future purchase frequency.
    Fit Gamma-Gamma model to predict expected monetary value.

    Args:
        df: Cleaned transaction DataFrame with CustomerID, InvoiceDate, Revenue cols
        observation_period_end: ISO date string (default: max date in df)

    Returns:
        dict with fitted models and summary_data
    """
    if not LIFETIMES_AVAILABLE:
        raise ImportError("Install lifetimes: pip install lifetimes")

    if observation_period_end is None:
        observation_period_end = df['InvoiceDate'].max()

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col='CustomerID',
        datetime_col='InvoiceDate',
        monetary_value_col='Revenue',
        observation_period_end=observation_period_end,
        freq='D'
    )

    # Filter: only repeat buyers for Gamma-Gamma
    returning_customers = summary[summary['frequency'] > 0]

    # Fit BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    print(f"BG/NBD fitted. Log-likelihood: {bgf.log_likelihood_:.2f}")

    # Fit Gamma-Gamma on returning customers only
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])
    print(f"Gamma-Gamma fitted. Log-likelihood: {ggf.log_likelihood_:.2f}")

    # Predict 12-month LTV
    summary['predicted_purchases_12m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        365, summary['frequency'], summary['recency'], summary['T']
    )
    summary['expected_avg_profit'] = ggf.conditional_expected_average_profit(
        summary['frequency'], summary['monetary_value']
    )
    summary['LTV_12m'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=12,
        discount_rate=0.01
    )

    print(f"\nLTV 12-month Summary:")
    print(summary['LTV_12m'].describe().round(2))

    return {'bgf': bgf, 'ggf': ggf, 'summary': summary}


# ──────────────────────────────────────────────
# XGBoost LTV Regression
# ──────────────────────────────────────────────

FEATURE_COLS = [
    'total_orders', 'total_items', 'unique_products', 'unique_categories',
    'avg_order_value', 'std_order_value', 'avg_unit_price', 'max_unit_price',
    'customer_age_days', 'recency_days', 'purchase_span_days',
    'avg_days_between_orders', 'purchase_frequency_per_month',
    'revenue_per_item', 'product_diversity_ratio', 'is_multi_country',
    'log_total_revenue', 'log_avg_order_value', 'log_total_items',
]


def prepare_ltv_dataset(features_df: pd.DataFrame,
                         target_col: str = 'total_revenue',
                         test_size: float = 0.2,
                         random_state: int = 42):
    """
    Split features into train/test sets for LTV regression.
    Target: total revenue (proxy for LTV in observed window).
    """
    available_features = [c for c in FEATURE_COLS if c in features_df.columns]
    X = features_df[available_features].fillna(0)
    y = np.log1p(features_df[target_col])  # log-transform target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(available_features)}")
    return X_train, X_test, y_train, y_test, available_features


def evaluate_model(model, X_test, y_test, model_name: str = 'Model') -> dict:
    """Compute regression metrics on test set."""
    y_pred = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    }
    print(f"\n{model_name} — RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
    return metrics


def train_ltv_models(X_train, X_test, y_train, y_test,
                      experiment_name: str = 'LTV_Modeling') -> dict:
    """
    Train multiple LTV regression models with MLflow tracking.
    Returns best model and all results.
    """
    mlflow.set_experiment(experiment_name)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        ),
    }

    results = {}
    best_model = None
    best_rmse = float('inf')

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)

            # Log to MLflow
            mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

            results[name] = {'model': model, 'metrics': metrics}

            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model = (name, model)

    print(f"\n✅ Best model: {best_model[0]} (RMSE: {best_rmse:.4f})")
    return results, best_model


def plot_feature_importance(model, feature_names: list, top_n: int = 20, title: str = 'Feature Importance'):
    """Plot XGBoost / tree-based feature importance."""
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)[:top_n]

        plt.figure(figsize=(10, 6))
        imp.sort_values().plot(kind='barh', color='steelblue')
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('../outputs/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()


def predict_ltv_segments(features_df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    """
    Predict LTV for all customers and assign LTV tiers.
    """
    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)
    features_df = features_df.copy()
    features_df['Predicted_LTV_Log'] = model.predict(X)
    features_df['Predicted_LTV'] = np.expm1(features_df['Predicted_LTV_Log'])

    # Tier assignment
    ltv_quantiles = features_df['Predicted_LTV'].quantile([0.25, 0.50, 0.75])
    features_df['LTV_Tier'] = pd.cut(
        features_df['Predicted_LTV'],
        bins=[-np.inf, ltv_quantiles[0.25], ltv_quantiles[0.50], ltv_quantiles[0.75], np.inf],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )

    tier_summary = features_df.groupby('LTV_Tier')['Predicted_LTV'].agg(['count', 'mean', 'sum']).round(2)
    print("\nLTV Tier Summary:")
    print(tier_summary)
    return features_df


if __name__ == '__main__':
    print("LTV Model module loaded. Run notebooks for full pipeline.")
