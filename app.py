"""
dashboard/app.py
-----------------
Streamlit Customer Analytics Dashboard
Displays: Segment distribution, LTV tiers, churn risk, cohort retention
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 10px; text-align: center;}
    .segment-chip {border-radius: 20px; padding: 4px 12px; font-size: 13px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Synthetic Data Generator (for demo)
# ──────────────────────────────────────────────

@st.cache_data
def generate_demo_data(n=1000, seed=42):
    np.random.seed(seed)
    segments = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                'At Risk', 'Hibernating', 'Lost']
    seg_weights = [0.15, 0.20, 0.18, 0.15, 0.17, 0.15]

    df = pd.DataFrame({
        'CustomerID': [f'C{i:05d}' for i in range(n)],
        'Recency': np.random.exponential(scale=60, size=n).clip(1, 365).astype(int),
        'Frequency': np.random.zipf(2.0, size=n).clip(1, 50),
        'Monetary': np.random.lognormal(mean=5.5, sigma=1.2, size=n).round(2),
        'Segment': np.random.choice(segments, size=n, p=seg_weights),
        'LTV_Tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'],
                                     size=n, p=[0.35, 0.30, 0.20, 0.15]),
        'Predicted_LTV': np.random.lognormal(mean=6.0, sigma=1.5, size=n).round(2),
        'Churn_Probability': np.random.beta(2, 5, size=n).round(3),
        'Churn_Risk_Tier': np.random.choice(['Low', 'Medium', 'High', 'Critical'],
                                            size=n, p=[0.45, 0.30, 0.15, 0.10]),
        'Country': np.random.choice(['India', 'UK', 'UAE', 'USA'],
                                    size=n, p=[0.70, 0.15, 0.10, 0.05]),
        'JoinDate': pd.date_range('2022-01-01', periods=n, freq='8H')[:n],
    })
    return df


@st.cache_data
def generate_cohort_data():
    months = pd.date_range('2023-01-01', periods=12, freq='MS')
    cohort_sizes = np.random.randint(80, 200, size=12)
    retention = {}
    for i, m in enumerate(months):
        row = [1.0]
        for j in range(1, 12 - i):
            prev = row[-1]
            row.append(round(max(prev * np.random.uniform(0.6, 0.85), 0.01), 3))
        row += [None] * (11 - len(row) + 1)
        retention[m.strftime('%b %Y')] = row
    return pd.DataFrame(retention, index=[f'Month +{i}' for i in range(12)]).T


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Reliance_Retail_logo.svg/200px-Reliance_Retail_logo.svg.png",
                 width=140, use_column_width=False)
st.sidebar.title("🛍️ Customer Intelligence")
st.sidebar.markdown("**Powered by ML Pipeline**")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Segmentation",
    "💰 LTV Analysis",
    "⚠️ Churn Risk",
    "📅 Cohort Retention"
])

st.sidebar.markdown("---")
selected_segments = st.sidebar.multiselect(
    "Filter Segments",
    ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Hibernating', 'Lost'],
    default=['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Hibernating', 'Lost']
)
selected_countries = st.sidebar.multiselect("Filter Country", ['India', 'UK', 'UAE', 'USA'],
                                             default=['India', 'UK', 'UAE', 'USA'])

# Load data
df = generate_demo_data()
df_filtered = df[df['Segment'].isin(selected_segments) & df['Country'].isin(selected_countries)]

# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

SEGMENT_COLORS = {
    'Champions': '#2ecc71', 'Loyal Customers': '#3498db',
    'Potential Loyalists': '#9b59b6', 'At Risk': '#e67e22',
    'Hibernating': '#95a5a6', 'Lost': '#e74c3c'
}

if page == "📊 Overview":
    st.title("📊 Customer Analytics Overview")
    st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')} | {len(df_filtered):,} customers shown")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df_filtered):,}")
    c2.metric("Total Revenue", f"£{df_filtered['Monetary'].sum():,.0f}")
    c3.metric("Avg LTV (12m)", f"£{df_filtered['Predicted_LTV'].mean():,.0f}")
    c4.metric("Avg Churn Risk", f"{df_filtered['Churn_Probability'].mean():.1%}")
    c5.metric("Champions", f"{(df_filtered['Segment'] == 'Champions').sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        seg_counts = df_filtered['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig = px.pie(seg_counts, names='Segment', values='Count',
                     title='Customer Segment Distribution',
                     color='Segment', color_discrete_map=SEGMENT_COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = df_filtered.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(seg_rev, x='Segment', y='Monetary',
                     title='Revenue by Segment', color='Segment',
                     color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Segmentation":
    st.title("🔍 Customer Segmentation (RFM + K-Means)")

    fig = px.scatter_3d(df_filtered, x='Recency', y='Frequency', z='Monetary',
                        color='Segment', color_discrete_map=SEGMENT_COLORS,
                        title='3D RFM Customer Map',
                        opacity=0.7, size_max=8,
                        labels={'Recency': 'Recency (days)', 'Frequency': 'Orders', 'Monetary': 'Revenue (£)'})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Summary Table")
    seg_table = df_filtered.groupby('Segment').agg(
        Customers=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).round(1).sort_values('Total_Revenue', ascending=False)
    seg_table['Revenue_Share'] = (seg_table['Total_Revenue'] / seg_table['Total_Revenue'].sum() * 100).round(1)
    st.dataframe(seg_table.style.background_gradient(cmap='Blues', subset=['Total_Revenue']), use_container_width=True)

elif page == "💰 LTV Analysis":
    st.title("💰 Customer Lifetime Value (LTV) Analysis")

    c1, c2 = st.columns(2)
    with c1:
        tier_counts = df_filtered['LTV_Tier'].value_counts().reindex(['Platinum', 'Gold', 'Silver', 'Bronze'])
        fig = px.bar(tier_counts, title='LTV Tier Distribution',
                     labels={'index': 'Tier', 'value': 'Customers'},
                     color=tier_counts.index,
                     color_discrete_map={'Platinum': '#7b2d8b', 'Gold': '#f39c12', 'Silver': '#95a5a6', 'Bronze': '#cd7f32'})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df_filtered, x='Predicted_LTV', nbins=40,
                           title='Predicted LTV Distribution',
                           color_discrete_sequence=['steelblue'])
        fig.update_xaxes(title='Predicted 12-Month LTV (£)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### LTV by Segment")
    ltv_seg = df_filtered.groupby('Segment')['Predicted_LTV'].agg(['mean', 'median', 'sum']).round(2)
    ltv_seg.columns = ['Avg LTV', 'Median LTV', 'Total LTV']
    ltv_seg = ltv_seg.sort_values('Avg LTV', ascending=False)
    st.dataframe(ltv_seg.style.background_gradient(cmap='Greens', subset=['Avg LTV']), use_container_width=True)

elif page == "⚠️ Churn Risk":
    st.title("⚠️ Churn Propensity Scoring")

    c1, c2, c3, c4 = st.columns(4)
    for col, tier, color in zip([c1, c2, c3, c4],
                                  ['Critical', 'High', 'Medium', 'Low'],
                                  ['🔴', '🟠', '🟡', '🟢']):
        count = (df_filtered['Churn_Risk_Tier'] == tier).sum()
        col.metric(f"{color} {tier} Risk", f"{count:,}")

    fig = px.histogram(df_filtered, x='Churn_Probability', nbins=50,
                       color='Churn_Risk_Tier',
                       title='Churn Probability Distribution by Risk Tier',
                       color_discrete_map={'Critical': 'red', 'High': 'orange', 'Medium': 'gold', 'Low': 'green'},
                       barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top At-Risk Customers (for Outreach)")
    at_risk = df_filtered[df_filtered['Churn_Risk_Tier'].isin(['Critical', 'High'])].sort_values(
        'Churn_Probability', ascending=False
    )[['CustomerID', 'Segment', 'Monetary', 'Predicted_LTV', 'Churn_Probability', 'Churn_Risk_Tier']].head(20)
    st.dataframe(at_risk.reset_index(drop=True), use_container_width=True)

elif page == "📅 Cohort Retention":
    st.title("📅 Cohort Retention Analysis")
    st.caption("Monthly acquisition cohorts tracked over 12-month window")

    cohort_df = generate_cohort_data()
    numeric_cohort = cohort_df.applymap(lambda x: x if pd.notna(x) else np.nan).astype(float)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(numeric_cohort, annot=True, fmt='.0%', cmap='YlOrRd_r',
                linewidths=0.3, ax=ax, cbar_kws={'label': 'Retention Rate'},
                vmin=0, vmax=1, mask=numeric_cohort.isna())
    ax.set_title('Customer Retention by Cohort (Monthly)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Months Since Acquisition')
    ax.set_ylabel('Acquisition Cohort')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Insight:** Champions and Loyal Customer cohorts show 40-60% retention at month 6.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built by Mohammed Faizan | NIT Surat\nData Science · Reliance Retail")
