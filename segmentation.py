"""
segmentation.py
---------------
Customer segmentation using RFM scores, K-Means, DBSCAN,
and rule-based segment labeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Segment labeling from RFM scores
# ──────────────────────────────────────────────

SEGMENT_MAP = {
    (5, 5): 'Champions',
    (5, 4): 'Champions',
    (4, 5): 'Champions',
    (4, 4): 'Loyal Customers',
    (3, 5): 'Loyal Customers',
    (5, 3): 'Potential Loyalists',
    (4, 3): 'Potential Loyalists',
    (5, 2): 'Potential Loyalists',
    (3, 3): 'Need Attention',
    (4, 2): 'Need Attention',
    (3, 2): 'About to Sleep',
    (2, 3): 'At Risk',
    (2, 4): 'At Risk',
    (2, 5): 'At Risk',
    (1, 5): "Can't Lose Them",
    (1, 4): "Can't Lose Them",
    (1, 3): 'Hibernating',
    (1, 2): 'Hibernating',
    (2, 2): 'Hibernating',
    (1, 1): 'Lost',
    (2, 1): 'Lost',
    (3, 1): 'Lost',
}


def label_segment_from_rfm(r_score: int, f_score: int) -> str:
    """Map (R_Score, F_Score) tuple to a named segment."""
    return SEGMENT_MAP.get((r_score, f_score), 'Need Attention')


def assign_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based segment labeling to RFM-scored DataFrame."""
    rfm = rfm.copy()
    rfm['Segment'] = rfm.apply(
        lambda row: label_segment_from_rfm(row['R_Score'], row['F_Score']), axis=1
    )
    return rfm


# ──────────────────────────────────────────────
# K-Means Clustering
# ──────────────────────────────────────────────

def find_optimal_k(rfm_scaled: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Use silhouette score to find the optimal number of clusters.
    Returns the k with the highest silhouette score.
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        scores[k] = silhouette_score(rfm_scaled, labels)

    optimal_k = max(scores, key=scores.get)
    print(f"Optimal K: {optimal_k} (Silhouette: {scores[optimal_k]:.4f})")
    return optimal_k


def run_kmeans(rfm: pd.DataFrame, n_clusters: int = 5,
               features: list = ['Recency', 'Frequency', 'Monetary']) -> pd.DataFrame:
    """
    Run K-Means clustering on RFM features.

    Returns:
        rfm with KMeans_Cluster column added
    """
    rfm = rfm.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['KMeans_Cluster'] = km.fit_predict(X_scaled)

    # Label clusters by their average monetary value (high → low)
    cluster_summary = rfm.groupby('KMeans_Cluster')[features].mean()
    cluster_summary['Rank'] = cluster_summary['Monetary'].rank(ascending=False).astype(int)
    rank_map = cluster_summary['Rank'].to_dict()
    rfm['Cluster_Rank'] = rfm['KMeans_Cluster'].map(rank_map)

    print(f"\nK-Means Cluster Summary ({n_clusters} clusters):")
    print(cluster_summary.sort_values('Monetary', ascending=False).to_string())

    return rfm, km, scaler


def run_dbscan(rfm: pd.DataFrame, eps: float = 0.5, min_samples: int = 5,
               features: list = ['Recency', 'Frequency', 'Monetary']) -> pd.DataFrame:
    """
    Run DBSCAN to identify outlier customers (noise points labeled as -1).
    Useful for finding very high-value or very anomalous customers.
    """
    rfm = rfm.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    db = DBSCAN(eps=eps, min_samples=min_samples)
    rfm['DBSCAN_Cluster'] = db.fit_predict(X_scaled)

    n_clusters = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'].values else 0)
    n_noise = (rfm['DBSCAN_Cluster'] == -1).sum()

    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise/outlier points")
    return rfm


def run_gmm(rfm: pd.DataFrame, n_components: int = 5,
            features: list = ['Recency', 'Frequency', 'Monetary']) -> pd.DataFrame:
    """
    Gaussian Mixture Model for soft / probabilistic cluster assignment.
    """
    rfm = rfm.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    rfm['GMM_Cluster'] = gmm.fit_predict(X_scaled)

    # Soft probabilities
    probs = gmm.predict_proba(X_scaled)
    rfm['GMM_Max_Prob'] = probs.max(axis=1)

    print(f"GMM BIC: {gmm.bic(X_scaled):.2f} | AIC: {gmm.aic(X_scaled):.2f}")
    return rfm


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_segment_distribution(rfm: pd.DataFrame, segment_col: str = 'Segment',
                               title: str = 'Customer Segment Distribution'):
    """Bar chart of customer counts per segment."""
    seg_counts = rfm[segment_col].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    seg_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title(f'{title} — Count')
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('# Customers')
    axes[0].tick_params(axis='x', rotation=45)

    # Revenue contribution
    if 'Monetary' in rfm.columns:
        seg_revenue = rfm.groupby(segment_col)['Monetary'].sum().sort_values(ascending=False)
        seg_revenue.plot(kind='bar', ax=axes[1], color='darkorange', edgecolor='white')
        axes[1].set_title(f'{title} — Revenue')
        axes[1].set_xlabel('Segment')
        axes[1].set_ylabel('Total Revenue (£)')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('../outputs/segment_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rfm_heatmap(rfm: pd.DataFrame):
    """RFM score heatmap: R vs F, colored by avg Monetary."""
    pivot = rfm.pivot_table(values='Monetary', index='R_Score', columns='F_Score', aggfunc='mean')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Average Monetary Value by R_Score × F_Score')
    plt.xlabel('Frequency Score')
    plt.ylabel('Recency Score')
    plt.tight_layout()
    plt.savefig('../outputs/rfm_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_clusters(rfm: pd.DataFrame, cluster_col: str = 'KMeans_Cluster',
                      features: list = ['Recency', 'Frequency', 'Monetary']):
    """2D PCA projection of customer clusters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(components[:, 0], components[:, 1],
                          c=rfm[cluster_col], cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'Customer Clusters — PCA Projection ({cluster_col})')
    plt.tight_layout()
    plt.savefig('../outputs/pca_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()


def summarize_segments(rfm: pd.DataFrame, segment_col: str = 'Segment') -> pd.DataFrame:
    """Summary table: count, avg RFM, total revenue per segment."""
    summary = rfm.groupby(segment_col).agg(
        Customer_Count=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
    ).round(2).sort_values('Total_Revenue', ascending=False)

    summary['Revenue_Share_%'] = (summary['Total_Revenue'] / summary['Total_Revenue'].sum() * 100).round(1)
    return summary


if __name__ == '__main__':
    from data_preprocessing import load_data, clean_transactions, compute_rfm, score_rfm

    df = load_data('../data/sample_transactions.csv')
    df = clean_transactions(df)
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = assign_rfm_segments(rfm)

    print("\nSegment Summary:")
    print(summarize_segments(rfm).to_string())
