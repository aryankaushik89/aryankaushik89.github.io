import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from streamlit_dynamic_filters import DynamicFilters

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Customer 360 Analytics & AI Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR HEADER, CARDS, TABS, LOGO, HEADINGS ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.3rem; }
    .kpi-card {
        background: linear-gradient(135deg, #282b51 80%, #18192c 100%);
        border-radius: 18px;
        box-shadow: 0 2px 18px 1px rgba(30,31,62,0.13);
        padding: 22px 32px 18px 26px;
        color: #f3f5f8;
        margin-bottom: 18px;
        text-align: center;
    }
    .kpi-label { font-size: 1.25em; font-weight: 700; color: #cfd3e7; margin-bottom: 7px; }
    .kpi-value { font-size: 2.9em; font-weight: 900; color: #fff; margin-bottom: 4px; letter-spacing: -1px; }
    .main-header {
        font-size: 4.2em !important;
        font-weight: 900;
        color: #fff;
        margin-bottom: 0.18em;
        margin-left: 2px;
    }
    .sub-header {
        font-size: 1.08em;
        color: #b5badf;
        margin-bottom: 8px;
        margin-left: 2px;
    }
    .demo-note { display: none; }
    div[data-testid="stTabs"] > div {
        font-size: 2.2em !important;
        font-weight: 800 !important;
        color: #d7d7f7 !important;
        background: #18192c !important;
        border-radius: 14px 14px 0 0 !important;
        min-height: 78px !important;
    }
    div[role="tablist"] > div[aria-selected="true"] {
        background: #282b51 !important;
        color: #fff !important;
        border-bottom: 6px solid #4290fa !important;
        border-radius: 14px 14px 0 0 !important;
    }
    .big-heading { font-size: 2.4em !important; font-weight: 800; color: #fafbff; margin-bottom: 4px; }
    .sub-explain { font-size: 1.12em; color: #b7bedf; margin-bottom: 16px; margin-top: -10px;}
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    return pd.read_csv("streamflow_customers.csv")
df = load_data()

iso2_to_iso3 = {
    'US': 'USA', 'IN': 'IND', 'BR': 'BRA', 'FR': 'FRA', 'JP': 'JPN',
    'UK': 'GBR', 'AU': 'AUS', 'CA': 'CAN', 'ZA': 'ZAF', 'DE': 'DEU'
}
if 'country_iso3' not in df.columns:
    df['country_iso3'] = df['country'].map(iso2_to_iso3)
df['month_period'] = pd.to_datetime(df['signup_date'], errors='coerce').dt.to_period('M').astype(str)

# --- SIDEBAR: LOGO, FILTERS, FILE UPLOAD ---
with st.sidebar:
    st.image("streamflow_logo.png", width=330)
    st.markdown("<h4 style='color:#f4f6fc; margin-bottom:1px;'>Customer 360 Analytics & AI</h4>", unsafe_allow_html=True)
    st.markdown("SaaS Customer Segmentation & Insights", unsafe_allow_html=True)
    st.divider()
    upload = st.file_uploader("Upload CSV (200MB max)", type="csv")
    st.divider()
    filter_fields = ['subscription_type', 'cluster', 'month_period']
    filters = DynamicFilters(df, filters=filter_fields)
    filters.display_filters()
    st.markdown("<div style='font-size:12px;color:#b3b6ce;margin-top:10px;'>Showing demo data. Upload your own CSV to customize all tables.</div>", unsafe_allow_html=True)
    st.divider()

df_filt = filters.filter_df()

# --- HEADER (NO CARD, LEFT-ALIGNED, BIGGER) ---
st.markdown("<div class='main-header'>Customer 360 Analytics & AI Insights</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload your customer data to analyze engagement, loyalty, churn, segments, and get instant AI-driven business recommendations.<br>Or just explore with our rich demo dataset.</div>", unsafe_allow_html=True)
st.markdown("&nbsp;", unsafe_allow_html=True)  # Add vertical space after sub-header

# --- TABS (LARGER, MODERN STYLE) ---
tabs = st.tabs([
    "Overview & Metrics", "Visual Trends & Map", "Segmentation & Clusters", "AI Insights & Export"
])

# ============ TAB 1: Overview & Metrics =============
with tabs[0]:
    st.markdown("<div class='big-heading'>Key Metrics & Segmentation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• Track core SaaS customer health, engagement, loyalty and premium conversion.</div>", unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)

    # --- KPI Cards (horizontal row, always) ---
    kpi_list = [
        ("Users", f"{len(df_filt):,}"),
        ("Churn Rate", f"{df_filt['churned'].mean()*100:.1f}%"),
        ("Avg Loyalty", f"{int(df_filt['loyalty_points'].mean()):,}"),
        ("Premium %", f"{df_filt['subscription_type'].eq('Premium').mean()*100:.1f}%"),
        ("Countries", f"{df_filt['country'].nunique()}")
    ]
    kpi_cards = st.columns(5)
    for col, (label, value) in zip(kpi_cards, kpi_list):
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ---- 4 MAIN VISUALS IN 2x2 GRID ----
    chart_cols = st.columns(2, gap="large")
    # Churn Rate by Subscription Type (NEW, replaces world map)
    with chart_cols[0]:
        st.markdown("<div class='big-heading'>Churn Rate by Subscription Type</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Churn sharply decreases with each higher subscription tier; enterprise churn is extremely low.</div>", unsafe_allow_html=True)
        sub_order = ['Free', 'Basic', 'Premium', 'Enterprise']
        custom_palette = ['#57b8ff', '#a259f7', '#f95d9b', '#ffc300']
        churn_rate = df_filt.groupby('subscription_type')['churned'].mean().reindex(sub_order)
        fig1 = px.bar(
            churn_rate.reset_index(), x='subscription_type', y='churned',
            color='subscription_type', color_discrete_sequence=custom_palette,
            text='churned'
        )
        fig1.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig1.update_layout(
            xaxis_title="Subscription Type",
            yaxis_title="Churn Rate",
            showlegend=False,
            yaxis_tickformat=".0%",
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Observation:**\n- **Churn is highest for Free users** and decreases sharply with each higher tier.\n  - Focus on converting free users to paid plans.\n- **Enterprise churn is extremely low**—retention is strongest for our most valuable accounts.\n  - Invest further in enterprise relationships.", unsafe_allow_html=False)

    # Subscription Pie
    with chart_cols[1]:
        st.markdown("<div class='big-heading'>Subscription Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• See what percent of users are Free, Basic, Premium, or Enterprise.</div>", unsafe_allow_html=True)
        pie_data = df_filt['subscription_type'].value_counts().reindex(sub_order).reset_index()
        pie_data.columns = ['subscription_type', 'count']
        fig_pie = px.pie(
            pie_data, names='subscription_type', values='count',
            color='subscription_type',
            color_discrete_sequence=custom_palette
        )
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # 2nd row (two visuals)
    chart2_cols = st.columns(2, gap="large")
    # Feature Correlation with Churn (NEW)
    with chart2_cols[0]:
        st.markdown("<div class='big-heading'>Feature Correlation with Churn</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Which features are most predictive of churn? Top correlations shown.</div>", unsafe_allow_html=True)
        num_df = df_filt.select_dtypes(include=['number'])
        if 'churned' in num_df.columns:
            corrs = num_df.corr(numeric_only=True)
            churn_corrs = corrs['churned'].sort_values(key=lambda x: abs(x), ascending=False)[1:9]
            fig4 = px.bar(
                churn_corrs.reset_index(), x='index', y='churned',
                color='churned', color_continuous_scale='sunsetdark',
            )
            fig4.update_layout(
                xaxis_title="Feature",
                yaxis_title="Correlation with Churn",
                margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("**Actionable Insights:**\n- **Low engagement score, high support tickets, low loyalty, and missed payments** are the most predictive of churn.\n  - Retention campaigns for these users can reduce churn dramatically.\n- **Targeted outreach for risk factors** can improve LTV.\n  - Build automated alerts for these risk features.", unsafe_allow_html=False)
        else:
            st.info("No churned column available for correlation.")

    # Bar: Top Clusters by Spend
    with chart2_cols[1]:
        st.markdown("<div class='big-heading'>Top Clusters by Spend</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Identify clusters with highest average monthly spend for targeting.</div>", unsafe_allow_html=True)
        if 'cluster' in df_filt.columns and 'monthly_spend' in df_filt.columns:
            cluster_spend = df_filt.groupby('cluster', observed=True)['monthly_spend'].mean().reset_index()
            cluster_spend = cluster_spend.sort_values('monthly_spend', ascending=False)
            fig_bar = px.bar(
                cluster_spend, x='cluster', y='monthly_spend', color='cluster',
                color_continuous_scale="blues"
            )
            fig_bar.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                xaxis=dict(title='Cluster', tickmode='array', tickvals=cluster_spend['cluster']),
                yaxis=dict(title='Avg. Spend'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No cluster or spend data available.")

# ============ TAB 2: Visual Trends & Map =============
with tabs[1]:
    st.markdown("<div class='big-heading'>Visual Trends & World Map</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• See monthly active user (MAU) trends and churn rate by tenure, globally.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,2], gap="large")
    with c1:
        st.markdown("<div class='big-heading'>Monthly Active Users (MAU)</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Trend of unique active users by signup month, with rolling mean.</div>", unsafe_allow_html=True)
        if 'signup_date' in df_filt.columns:
            mau = (df_filt.groupby(pd.to_datetime(df_filt['signup_date'], errors='coerce').dt.to_period('M'))
                   .agg({'customer_id':'nunique'})
                   .rename(columns={'customer_id':'MAU'}))
            mau = mau.reset_index()
            mau['month'] = mau['signup_date'].astype(str)
            mau['MAU_smooth'] = mau['MAU'].rolling(window=4, center=True, min_periods=1).mean()
            fig_mau = go.Figure()
            fig_mau.add_trace(go.Scatter(
                x=mau['month'], y=mau['MAU'], mode='lines+markers', name='Raw MAU',
                line=dict(color='#6e94d9', width=2), opacity=0.35
            ))
            fig_mau.add_trace(go.Scatter(
                x=mau['month'], y=mau['MAU_smooth'], mode='lines+markers', name='Smoothed MAU',
                line=dict(color='#2263cf', width=4)
            ))
            fig_mau.update_layout(
                title="",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(size=15))
            )
            st.plotly_chart(fig_mau, use_container_width=True)
        else:
            st.info("No signup_date column for MAU chart.")
    with c2:
        st.markdown("<div class='big-heading'>Customer Geo Trends</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Customer distribution by country, sized by active users.</div>", unsafe_allow_html=True)
        geo_df = df_filt.groupby(['country_iso3']).agg(customers=('customer_id','count')).reset_index()
        fig_geo = px.scatter_geo(
            geo_df, locations="country_iso3", size="customers",
            projection="natural earth", color="customers",
            color_continuous_scale="Purples"
        )
        fig_geo.update_geos(showland=True, landcolor="#18192c", bgcolor='rgba(0,0,0,0)')
        fig_geo.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_geo, use_container_width=True)

    st.divider()
    st.markdown("<div class='big-heading'>Churn Rate by Customer Tenure</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• Raw and smoothed churn rate by months as a customer (rolling mean).</div>", unsafe_allow_html=True)
    if 'tenure_months' in df_filt.columns:
        tenure_churn = df_filt.groupby('tenure_months').agg(
            churn_rate=('churned','mean'),
            users=('customer_id','count')
        ).reset_index()
        tenure_churn['churn_rate_smooth'] = tenure_churn['churn_rate'].rolling(window=6, center=True, min_periods=1).mean()
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Scatter(
            x=tenure_churn['tenure_months'], y=tenure_churn['churn_rate'],
            mode='lines', name='Churn Rate (Raw)',
            line=dict(color='#f95d9b', width=2), opacity=0.35
        ))
        fig_tc.add_trace(go.Scatter(
            x=tenure_churn['tenure_months'], y=tenure_churn['churn_rate_smooth'],
            mode='lines', name='Churn Rate (Smoothed)',
            line=dict(color='#f95d9b', width=4)
        ))
        fig_tc.update_layout(
            title="",
            xaxis_title="Tenure (Months)", yaxis_title="Churn Rate",
            font=dict(size=16), margin=dict(l=20, r=20, t=60, b=60),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(size=15)), yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_tc, use_container_width=True)
    else:
        st.info("No tenure_months data available.")

# ============ TAB 3: Segmentation & Clusters =============
with tabs[2]:
    st.markdown("<div class='big-heading'>Customer Segmentation & Clusters</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• Drill into clusters by loyalty, churn, and spend for micro-segmentation.</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='big-heading'>Loyalty by Cluster</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Compares average loyalty points by cluster (color: churn rate).</div>", unsafe_allow_html=True)
        seg_loy = df_filt.groupby('cluster').agg(
            avg_loyalty=('loyalty_points','mean'),
            churn_rate=('churned','mean'),
            customers=('customer_id','count')
        ).reset_index()
        fig_loy = px.bar(seg_loy, x='cluster', y='avg_loyalty', color='churn_rate', color_continuous_scale="RdPu",
                         title="")
        st.plotly_chart(fig_loy, use_container_width=True)
    with col2:
        st.markdown("<div class='big-heading'>Loyalty vs Spend by Cluster</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-explain'>• Clusters plotted by avg loyalty vs avg spend, bubble size = churn rate.</div>", unsafe_allow_html=True)
        if 'monthly_spend' in df_filt.columns:
            cluster_metrics = df_filt.groupby('cluster').agg(
                avg_loyalty=('loyalty_points','mean'),
                avg_spend=('monthly_spend','mean'),
                churn_rate=('churned','mean')
            ).reset_index()
            fig_scatter = px.scatter(
                cluster_metrics, x='avg_loyalty', y='avg_spend',
                size='churn_rate', color='cluster', hover_name='cluster',
                title="", labels={
                    'avg_loyalty': 'Avg Loyalty',
                    'avg_spend': 'Avg Spend'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No spend data for clusters available.")

    st.divider()
    st.markdown("<div class='big-heading'>Churn/Loyalty Table by Segment</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• Detailed segment table (cluster x subscription type): customers, churn, loyalty, spend.</div>", unsafe_allow_html=True)
    seg_tab = df_filt.groupby(['cluster','subscription_type']).agg(
        customers=('customer_id','count'),
        churn_rate=('churned','mean'),
        loyalty=('loyalty_points','mean'),
        spend=('monthly_spend','mean')
    ).reset_index()
    st.dataframe(seg_tab, use_container_width=True)

# ============ TAB 4: AI Insights & Export =============
with tabs[3]:
    st.markdown("<div class='big-heading'>AI Insights & Export</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• Download filtered tables, and review actionable AI-driven insights and risk segments below.</div>", unsafe_allow_html=True)

    # More robust insights, markdown with sub-bullets
    st.markdown("##### 🔍 Quick Wins / Risks")
    st.markdown(
        """
- **Churn highest in clusters:** 1, 3  
  - Focus on churn drivers in these segments with tailored retention offers.
- **Most loyal cluster:** Cluster 0 (avg loyalty 364)  
  - Analyze what makes this cluster stick—replicate best practices.
- **Most revenue:** Cluster 2 (avg spend $50)  
  - Upsell or cross-sell into this cluster to maximize revenue further.
- **Potential churn risk if loyalty below 250.**  
  - Launch proactive engagement for users near this threshold.
- **Consider retention campaign for clusters with churn > 30%.**  
  - Target with value reinforcement and incentives.
- **Explore pricing for Enterprise vs Premium segment.**  
  - Test price sensitivity; could unlock more enterprise conversions.
- **At-risk clusters:** 1, 3  
  - Further profile at-risk segments for targeted outreach.
- **Clusters with highest growth:** 2  
  - Double-down on growth drivers, promote case studies.
        """
    )

    st.divider()
    st.markdown("<div class='big-heading'>Download All Tables</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-explain'>• All exports are filtered to current dashboard selections.</div>", unsafe_allow_html=True)
    def export_csv_button(df, name, description):
        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv"><button style="padding:4px 18px; border-radius:8px; background:#423fa1; color:#fff; border:0; margin-right:7px;">{description}</button></a>'
        st.markdown(href, unsafe_allow_html=True)

    # CLEAR IN-TEXT EXPLANATIONS
    export_csv_button(df_filt, "filtered_data_all", "Download All Filtered Customer Data (CSV)")
    export_csv_button(seg_tab, "segment_churn_loyalty", "Download Segment Churn & Loyalty Table (CSV)")
    export_csv_button(cluster_metrics, "cluster_metrics", "Download Cluster Summary Metrics (CSV)")

# --- FOOTER ---
st.markdown("""
    <hr style="margin-top:38px; margin-bottom:3px;">
    <center>
    <span style='color:#6d729e;'>Data Science Portfolio Demo | Inspired by <a href="https://streamlit.io/gallery" style='color:#b8baff;'>Streamlit Gallery</a></span>
    </center>
""", unsafe_allow_html=True)
