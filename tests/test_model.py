import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import xgboost as xgb


# ──────────────────────────────────────────────
# Fixtures: reusable mock data
# ──────────────────────────────────────────────

@pytest.fixture
def sample_transactions():
    """Simulates a small slice of the 500K retail transaction dataset."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id":   np.random.randint(1, 50, n),
        "order_date":    pd.date_range("2023-01-01", periods=n, freq="D"),
        "order_value":   np.random.uniform(50, 5000, n),
        "quantity":      np.random.randint(1, 10, n),
        "category":      np.random.choice(["Electronics", "Apparel", "Grocery"], n),
    })


@pytest.fixture
def rfm_features(sample_transactions):
    """Builds RFM features from mock transactions."""
    snapshot_date = sample_transactions["order_date"].max() + pd.Timedelta(days=1)
    rfm = sample_transactions.groupby("customer_id").agg(
        Recency   = ("order_date",  lambda x: (snapshot_date - x.max()).days),
        Frequency = ("order_id" if "order_id" in sample_transactions else "order_value", "count"),
        Monetary  = ("order_value", "sum"),
    ).reset_index()
    return rfm


# ──────────────────────────────────────────────
# 1. Data integrity tests
# ──────────────────────────────────────────────

class TestDataIntegrity:

    def test_no_null_values(self, sample_transactions):
        assert sample_transactions.isnull().sum().sum() == 0, \
            "Raw transactions should have no nulls"

    def test_order_value_positive(self, sample_transactions):
        assert (sample_transactions["order_value"] > 0).all(), \
            "All order values must be positive"

    def test_rfm_columns_exist(self, rfm_features):
        for col in ["Recency", "Frequency", "Monetary"]:
            assert col in rfm_features.columns, f"Missing RFM column: {col}"

    def test_rfm_no_negatives(self, rfm_features):
        assert (rfm_features[["Recency", "Frequency", "Monetary"]] >= 0).all().all(), \
            "RFM values must be non-negative"


# ──────────────────────────────────────────────
# 2. Feature engineering tests
# ──────────────────────────────────────────────

class TestFeatureEngineering:

    def test_scaler_output_shape(self, rfm_features):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm_features[["Recency", "Frequency", "Monetary"]])
        assert scaled.shape == (len(rfm_features), 3), \
            "Scaler output shape mismatch"

    def test_scaler_mean_near_zero(self, rfm_features):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm_features[["Recency", "Frequency", "Monetary"]])
        means = scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-6), \
            "Scaled features should have mean ≈ 0"

    def test_category_encoding(self, sample_transactions):
        encoded = pd.get_dummies(sample_transactions["category"])
        assert encoded.shape[1] == 3, \
            "Expected 3 one-hot columns for 3 categories"


# ──────────────────────────────────────────────
# 3. Clustering (K-Means) tests
# ──────────────────────────────────────────────

class TestClustering:

    def test_kmeans_produces_5_segments(self, rfm_features):
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm_features[["Recency", "Frequency", "Monetary"]])
        model = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        assert len(set(labels)) == 5, \
            "K-Means should produce exactly 5 clusters"

    def test_silhouette_score_acceptable(self, rfm_features):
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm_features[["Recency", "Frequency", "Monetary"]])
        model = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        assert score > 0.1, \
            f"Silhouette score too low: {score:.3f} — check clustering quality"

    def test_all_customers_assigned(self, rfm_features):
        scaler = StandardScaler()
        X = scaler.fit_transform(rfm_features[["Recency", "Frequency", "Monetary"]])
        model = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        assert len(labels) == len(rfm_features), \
            "Every customer must be assigned a cluster"


# ──────────────────────────────────────────────
# 4. LTV model (XGBoost) tests
# ──────────────────────────────────────────────

class TestLTVModel:

    @pytest.fixture
    def ltv_data(self, rfm_features):
        """Mock feature matrix and LTV target."""
        X = rfm_features[["Recency", "Frequency", "Monetary"]].values
        y = rfm_features["Monetary"].values * np.random.uniform(0.8, 1.2, len(rfm_features))
        return X, y

    def test_model_trains_without_error(self, ltv_data):
        X, y = ltv_data
        model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        assert model is not None

    def test_predictions_are_positive(self, ltv_data):
        X, y = ltv_data
        model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds > 0).all(), \
            "LTV predictions should be positive"

    def test_prediction_shape_matches_input(self, ltv_data):
        X, y = ltv_data
        model = xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0], \
            "Number of predictions must match number of input rows"


# ──────────────────────────────────────────────
# 5. Churn model sanity tests
# ──────────────────────────────────────────────

class TestChurnModel:

    @pytest.fixture
    def churn_data(self, rfm_features):
        X = rfm_features[["Recency", "Frequency", "Monetary"]].values
        y = (rfm_features["Recency"] > rfm_features["Recency"].median()).astype(int).values
        return X, y

    def test_churn_labels_are_binary(self, churn_data):
        _, y = churn_data
        assert set(y).issubset({0, 1}), \
            "Churn labels must be binary (0 or 1)"

    def test_churn_model_predicts_probabilities(self, churn_data):
        from sklearn.ensemble import GradientBoostingClassifier
        X, y = churn_data
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all(), \
            "Churn probabilities must be between 0 and 1"
