"""
data_preprocessing.py
---------------------
Cleans raw retail transaction data and engineers features
for customer segmentation and LTV modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=['InvoiceDate'])
    logger.info(f"Loaded {len(df):,} rows from {filepath}")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data:
    - Remove cancellations (InvoiceNo starting with 'C')
    - Remove rows with null CustomerID
    - Remove negative Quantity or UnitPrice
    - Remove non-product stock codes
    """
    original_len = len(df)

    # Remove cancellations
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Remove null CustomerIDs
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    # Remove invalid quantities/prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Remove non-product codes (postage, manual entries, etc.)
    non_product_codes = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'DOT']
    df = df[~df['StockCode'].isin(non_product_codes)]

    # Compute line-level revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    logger.info(f"Cleaned: {original_len:,} → {len(df):,} rows ({original_len - len(df):,} removed)")
    return df.reset_index(drop=True)


def compute_rfm(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) metrics per customer.

    Args:
        df: Cleaned transaction DataFrame
        snapshot_date: Reference date for recency calculation (default: max date + 1 day)

    Returns:
        rfm: DataFrame with CustomerID, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()

    logger.info(f"RFM computed for {len(rfm):,} customers. Snapshot date: {snapshot_date.date()}")
    return rfm


def score_rfm(rfm: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """
    Assign RFM scores (1–5) per dimension and compute composite RFM score.
    Higher score = better customer.
    """
    rfm = rfm.copy()

    # Recency: lower is better → reverse ranking
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=quantiles, labels=range(quantiles, 0, -1), duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=quantiles, labels=range(1, quantiles + 1), duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=quantiles, labels=range(1, quantiles + 1), duplicates='drop')

    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    rfm['RFM_String'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    logger.info(f"RFM scores assigned. Score range: [{rfm['RFM_Score'].min()}, {rfm['RFM_Score'].max()}]")
    return rfm


def engineer_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 40+ behavioral and transactional features per customer for LTV modeling.
    """
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    agg = df.groupby('CustomerID').agg(
        total_revenue=('Revenue', 'sum'),
        total_orders=('InvoiceNo', 'nunique'),
        total_items=('Quantity', 'sum'),
        avg_order_value=('Revenue', lambda x: x.groupby(df.loc[x.index, 'InvoiceNo']).sum().mean()),
        std_order_value=('Revenue', lambda x: x.groupby(df.loc[x.index, 'InvoiceNo']).sum().std()),
        unique_products=('StockCode', 'nunique'),
        unique_categories=('StockCode', lambda x: x.str[:2].nunique()),
        first_purchase=('InvoiceDate', 'min'),
        last_purchase=('InvoiceDate', 'max'),
        avg_unit_price=('UnitPrice', 'mean'),
        max_unit_price=('UnitPrice', 'max'),
        total_countries=('Country', 'nunique'),
    ).reset_index()

    # Derived features
    agg['customer_age_days'] = (snapshot_date - agg['first_purchase']).dt.days
    agg['recency_days'] = (snapshot_date - agg['last_purchase']).dt.days
    agg['purchase_span_days'] = (agg['last_purchase'] - agg['first_purchase']).dt.days
    agg['avg_days_between_orders'] = agg['purchase_span_days'] / (agg['total_orders'] - 1).clip(lower=1)
    agg['purchase_frequency_per_month'] = agg['total_orders'] / (agg['customer_age_days'] / 30).clip(lower=0.1)
    agg['revenue_per_item'] = agg['total_revenue'] / agg['total_items'].clip(lower=1)
    agg['product_diversity_ratio'] = agg['unique_products'] / agg['total_items'].clip(lower=1)
    agg['is_multi_country'] = (agg['total_countries'] > 1).astype(int)
    agg['std_order_value'] = agg['std_order_value'].fillna(0)

    # Log-transform skewed features
    for col in ['total_revenue', 'avg_order_value', 'total_items']:
        agg[f'log_{col}'] = np.log1p(agg[col])

    logger.info(f"Engineered {len(agg.columns)} features for {len(agg):,} customers")
    return agg.drop(columns=['first_purchase', 'last_purchase'])


def generate_synthetic_sample(n_customers: int = 200, n_transactions: int = 500,
                               output_path: str = '../data/sample_transactions.csv'):
    """Generate a small synthetic dataset for testing purposes."""
    np.random.seed(42)

    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days

    records = []
    for _ in range(n_transactions):
        cust = np.random.choice(customer_ids)
        inv_date = start_date + timedelta(days=np.random.randint(0, date_range))
        qty = np.random.randint(1, 20)
        price = round(np.random.exponential(scale=5.0) + 0.5, 2)
        records.append({
            'InvoiceNo': f"INV{np.random.randint(10000, 99999)}",
            'StockCode': f"SKU{np.random.randint(100, 999)}",
            'Description': 'Sample Product',
            'Quantity': qty,
            'InvoiceDate': inv_date,
            'UnitPrice': price,
            'CustomerID': cust,
            'Country': np.random.choice(['India', 'UK', 'USA'], p=[0.8, 0.1, 0.1])
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic sample saved to {output_path}")
    return df


if __name__ == '__main__':
    generate_synthetic_sample()
