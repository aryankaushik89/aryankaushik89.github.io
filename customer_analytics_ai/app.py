import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Customer Analytics AI Agent", layout="wide")
st.markdown("""
<style>
.big-font {font-size:2.2em;}
.metric-block {background:#f7fafd;padding:12px 18px;border-radius:12px;box-shadow:0 2px 6px #0001;margin-bottom:18px;}
hr {border:0;border-top:2px solid #eee;}
</style>
""", unsafe_allow_html=True)

st.title("Customer 360 Analytics & AI Insights")
st.markdown("""
Upload your customer data and instantly explore engagement, loyalty, churn, and revenue by segment.
This dashboard detects outliers, summarizes business opportunities, and recommends actions—all powered by advanced analytics and automation.
""")

# --- Load Data (Demo fallback) ---
st.header("1. Upload Customer Data")
demo_file = "streamflow_customers.csv"
df = None
uploaded = st.file_uploader("Upload CSV (or use demo data)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df):,} records from upload.")
    demo_mode = False
elif os.path.exists(demo_file):
    df = pd.read_csv(demo_file)
    st.info(f"Currently displaying demo data: `{demo_file}`. Upload your own for custom analysis.")
    demo_mode = True

if df is not None:
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Clustering if Needed ---
    if 'cluster' not in df.columns:
        st.info("No cluster labels found. Running KMeans clustering for you.")
        segment_features = [
            'subscription_type', 'tenure_months', 'avg_weekly_usage_hrs',
            'num_courses_completed', 'num_certificates', 'engagement_score',
            'loyalty_points', 'monthly_spend', 'add_ons_purchased', 'support_tickets_last_6mo'
        ]
        type_map = {'Free':0, 'Basic':1, 'Premium':2, 'Enterprise':3}
        X = df[segment_features].copy()
        X['subscription_type'] = X['subscription_type'].map(type_map)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        st.success("Clustering complete. 4 segments detected.")

    # --- Main KPI Overview ---
    st.header("2. Business Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{df['churned'].mean():.1%}")
    col3.metric("Premium %", f"{(df['subscription_type'] == 'Premium').mean():.1%}")
    col4.metric("Avg Loyalty", f"{df['loyalty_points'].mean():,.0f}")

    # --- Revenue & Churn by Segment ---
    st.subheader("Segment Revenue & Churn")
    seg_rev = df.groupby('cluster').agg(
        Customers=('customer_id', 'count'),
        Avg_Monthly_Spend=('monthly_spend', 'mean'),
        Total_Monthly_Revenue=('monthly_spend', 'sum'),
        Churn_Rate=('churned', 'mean')
    ).reset_index()
    seg_rev['Revenue_Share'] = seg_rev['Total_Monthly_Revenue'] / seg_rev['Total_Monthly_Revenue'].sum()
    fig = px.bar(
        seg_rev, x='cluster', y=['Avg_Monthly_Spend', 'Churn_Rate'],
        barmode='group', title="Avg Monthly Spend & Churn Rate by Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Cohort Trend Analysis ---
    st.subheader("Cohort Trends (Recent vs Earlier)")
    recent = df[df['tenure_months'] <= 6]
    earlier = df[(df['tenure_months'] > 6) & (df['tenure_months'] <= 12)]
    def cohort_stats(subset):
        return {
            'Churn Rate': f"{subset['churned'].mean():.1%}",
            'Engagement': f"{subset['engagement_score'].mean():.1f}",
            'Loyalty': f"{subset['loyalty_points'].mean():.0f}"
        }
    st.write("**Recent (≤6mo):**", cohort_stats(recent))
    st.write("**Earlier (6-12mo):**", cohort_stats(earlier))

    # --- Outlier Detection ---
    st.subheader("Outlier Detection Insights")
    churn_thresh = seg_rev['Churn_Rate'].mean() + 2*seg_rev['Churn_Rate'].std()
    rev_thresh = seg_rev['Avg_Monthly_Spend'].mean() + 2*seg_rev['Avg_Monthly_Spend'].std()
    anomalies = []
    for i, row in seg_rev.iterrows():
        if row['Churn_Rate'] > churn_thresh:
            anomalies.append(f"Cluster {row['cluster']} has **unusually high churn** ({row['Churn_Rate']:.1%}).")
        if row['Avg_Monthly_Spend'] > rev_thresh:
            anomalies.append(f"Cluster {row['cluster']} has **unusually high average monthly spend** (${row['Avg_Monthly_Spend']:.0f}).")
    if anomalies:
        st.warning('\n'.join(anomalies))
        fig = px.bar(
            seg_rev[seg_rev['cluster'].isin([int(a.split()[1]) for a in anomalies if "Cluster" in a])],
            x='cluster', y=['Churn_Rate', 'Avg_Monthly_Spend'],
            barmode='group', title="Anomalous Clusters: Churn & Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant anomalies detected in churn or revenue by cluster.")

    # --- Cluster Explorer ---
    st.subheader("Explore a Cluster in Detail")
    clust = st.selectbox("Select cluster", sorted(df['cluster'].unique()))
    cdf = df[df['cluster']==clust]
    st.dataframe(cdf.describe().T.style.format("{:.2f}"))
    st.dataframe(cdf.head(15))

    # --- Executive Insights & Next Steps (AI Agent) ---
    st.header("3. AI-Generated Insights & Next Steps")
    cluster_summary = []
    for i, row in seg_rev.iterrows():
        desc = (
            f"**Cluster {row['cluster']}** | Size: {row['Customers']}, Churn: {row['Churn_Rate']:.1%}, "
            f"Avg Revenue: ${row['Avg_Monthly_Spend']:.0f}, Loyalty: {df[df['cluster']==row['cluster']]['loyalty_points'].mean():.0f}."
        )
        if row['Churn_Rate'] > churn_thresh:
            desc += " 🚨 *This segment has unusually high churn—investigate retention drivers*."
        if row['Avg_Monthly_Spend'] > rev_thresh:
            desc += " 💰 *This segment spends the most—target with loyalty and upsell offers*."
        cluster_summary.append(desc)
    for line in cluster_summary:
        st.markdown(line)

    # --- Recommended Actions
    st.markdown("#### Recommended Actions")
    actions = []
    if any("high churn" in s for s in cluster_summary):
        actions.append("- Launch targeted retention campaigns for high-churn clusters.")
    if any("spends the most" in s for s in cluster_summary):
        actions.append("- Target top-revenue clusters with loyalty and cross-sell offers.")
    actions.append("- Regularly monitor for emerging anomalies or sudden shifts.")
    st.markdown("\n".join(actions))
else:
    st.info("Please upload your data or use the demo to get started.")

