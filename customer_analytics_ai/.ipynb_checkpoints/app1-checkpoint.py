import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime

# -- PAGE SETUP --
st.set_page_config(
    page_title="Customer 360 Analytics & AI Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CARD STYLES ---
def card_style(css_id, bg="#23243a", pad="30px"):
    st.markdown(f"""
        <style>
        #{css_id} {{
            background: linear-gradient(135deg, #282b51 80%, #18192c 100%);
            border-radius: 22px;
            box-shadow: 0 2px 18px 1px rgba(30,31,62,0.13);
            padding: {pad};
            color: #f3f5f8;
            margin-bottom: 18px;
        }}
        </style>
    """, unsafe_allow_html=True)

def export_csv_button(df, name="export"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv"><button style="padding:4px 18px; border-radius:8px; background:#423fa1; color:#fff; border:0;">Download CSV</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    uploaded = st.session_state.get('user_csv', None)
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("streamflow_customers.csv")  # <-- Use your own path/demo
    return df

# --- COUNTRY CODES ---
iso2_to_iso3 = {
    'US': 'USA', 'IN': 'IND', 'BR': 'BRA', 'FR': 'FRA', 'JP': 'JPN',
    'UK': 'GBR', 'AU': 'AUS', 'CA': 'CAN', 'ZA': 'ZAF', 'DE': 'DEU'
}

# --- SIDEBAR ---
with st.sidebar:
    st.image("streamflow_logo.png", width=94)
    st.markdown("<h4 style='color:#f4f6fc; margin-bottom:2px;'>Customer 360 Analytics & AI</h4>", unsafe_allow_html=True)
    st.markdown("SaaS Customer Segmentation & Insights", unsafe_allow_html=True)
    st.divider()
    user_file = st.file_uploader("Upload CSV (200MB max)", type="csv", label_visibility="visible")
    if user_file:
        st.session_state['user_csv'] = user_file
    st.divider()
    st.markdown("<b>Key KPIs</b>", unsafe_allow_html=True)

# --- LOAD DATA ---
df = load_data()
if 'country_iso3' not in df.columns:
    df['country_iso3'] = df['country'].map(iso2_to_iso3)

# --- Filter State ---
with st.sidebar:
    kpi_cols = [
        ("👥 Customers", f"{len(df):,}"),
        ("💔 Churn Rate", f"{df['churned'].mean()*100:.1f}%"),
        ("🏆 Avg Loyalty", f"{int(df['loyalty_points'].mean()):,}"),
        ("💎 Premium %", f"{df['subscription_type'].eq('Premium').mean()*100:.1f}%"),
        ("🌍 Countries", f"{df['country'].nunique()}")
    ]
    for icon, value in kpi_cols:
        st.markdown(f"<div style='font-size:1.15em; margin-bottom:2px;'>{icon}</div><div style='font-size:2em; font-weight:bold; margin-bottom:15px; color:#ffe; '>{value}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<b>Filter Customers</b>", unsafe_allow_html=True)
    subs = sorted(df['subscription_type'].unique().tolist())
    clusters = sorted(df['cluster'].unique().tolist())
    countries = sorted(df['country'].unique().tolist())
    filter_sub = st.multiselect("Subscription", subs, default=subs)
    filter_clu = st.multiselect("Cluster", clusters, default=clusters)
    filter_cou = st.multiselect("Country", countries, default=countries)
    st.markdown("<div style='font-size:12px;color:#b3b6ce;margin-top:12px;'>Currently showing demo data.<br>You can upload your own CSV above for custom analysis.</div>", unsafe_allow_html=True)

# --- FILTERED DATA ---
df_filt = df[
    df['subscription_type'].isin(filter_sub)
    & df['cluster'].isin(filter_clu)
    & df['country'].isin(filter_cou)
]

# --- HEADER CARD ---
st.markdown("""
<div id="headercard">
    <h2 style="margin-bottom:2px; color:#fff;">Customer 360 Analytics & AI Insights</h2>
    <span style="font-size:1.15em; color:#b3b6ce;">
    Upload your customer data to analyze engagement, loyalty, churn, segments, and get instant AI-driven business recommendations.<br>
    Or just explore with our rich demo dataset.<br>
    </span>
    <span style="font-size:0.96em;color:#b3e0f9;">Showing demo data. You can upload your own CSV on the left for custom analysis.</span>
</div>
""", unsafe_allow_html=True)
card_style("headercard")

# --- TABS ---
tabs = st.tabs([
    "Overview & Metrics", "Visual Trends & Map", "Segmentation & Clusters", "AI Insights & Export"
])

# =================== TAB 1: Overview & Metrics ===================
with tabs[0]:
    st.markdown("<div id='kpirow' style='display:flex; gap:20px; margin-bottom:18px;'>", unsafe_allow_html=True)
    for icon, value in kpi_cols:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#282b51 65%,#23243A 100%);
                    border-radius:18px; box-shadow:0 2px 8px 2px #21234533;
                    padding:18px 26px 12px 20px; min-width:130px; max-width:185px; flex:1;'>
            <div style='font-size:1.03em; margin-bottom:3px;'>{icon}</div>
            <div style='font-size:1.6em; font-weight:700; color:#fff; margin-bottom:0;'>{value}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    grid1, grid2, grid3 = st.columns([1, 1, 1.15], gap="large")
    with grid1:
        st.markdown("#### 🌍 Customer Map")
        country_counts = df_filt['country_iso3'].value_counts().reset_index()
        country_counts.columns = ['country_iso3', 'customers']
        fig_map = px.choropleth(
            country_counts, locations='country_iso3',
            color='customers',
            color_continuous_scale="blues",
            projection="natural earth",
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(bgcolor='rgba(0,0,0,0)'),
            coloraxis_colorbar=dict(title="Customers"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        export_csv_button(country_counts, name="customer_by_country")

    with grid2:
        st.markdown("#### 📊 Subscription Breakdown")
        pie_data = df_filt['subscription_type'].value_counts().reset_index()
        pie_data.columns = ['subscription_type', 'count']
        fig_pie = px.pie(
            pie_data, names='subscription_type', values='count',
            color='subscription_type',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_pie.update_traces(textinfo='percent+label', pull=[0.06]*len(pie_data))
        fig_pie.update_layout(
            legend=dict(orientation='v', y=1, x=1.05, bgcolor='rgba(0,0,0,0)', font=dict(color='#fff')),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        export_csv_button(pie_data, name="subscriptions")

    with grid3:
        st.markdown("#### 🏅 Top Clusters by Spend")
        if 'cluster' in df_filt.columns and 'monthly_spend' in df_filt.columns:
            cluster_spend = df_filt.groupby('cluster', observed=True)['monthly_spend'].mean().reset_index()
            cluster_spend = cluster_spend.sort_values('monthly_spend', ascending=False)
            fig_bar = px.bar(
                cluster_spend, x='cluster', y='monthly_spend', color='cluster',
                color_continuous_scale="blues"
            )
            fig_bar.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                xaxis=dict(title='Cluster', tickmode='array', tickvals=cluster_spend['cluster']),
                yaxis=dict(title='Avg. Spend'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            export_csv_button(cluster_spend, name="cluster_spend")
        else:
            st.info("No cluster or spend data available.")

    st.markdown("### 🤖 AI Insights & Recommendations")
    main_takeaways = [
        f"**Churn Rate:** {df_filt['churned'].mean()*100:.1f}% (Target: <11%)",
        f"**Most Premium Users:** {pie_data.sort_values('count', ascending=False)['subscription_type'].iloc[0]}",
        f"**Largest Country:** {country_counts.sort_values('customers', ascending=False)['country_iso3'].iloc[0]} ({country_counts['customers'].max()} customers)",
        f"**Top Cluster by Spend:** {int(cluster_spend['cluster'].iloc[0])} (${cluster_spend['monthly_spend'].max():.2f})" if 'cluster_spend' in locals() and len(cluster_spend) > 0 else "",
        "Consider loyalty/retention campaigns in top-churn clusters.",
    ]
    st.markdown("<ul style='font-size:1.11em;'>" + ''.join([f"<li>{item}</li>" for item in main_takeaways if item]) + "</ul>", unsafe_allow_html=True)
    st.info("All data/insights above respond live to filter changes.")

# =================== TAB 2: Visual Trends & Map ===================
with tabs[1]:
    st.markdown("<h4 style='margin-bottom:8px;color:#e3e8f5;'>📈 Visual Trends & World Map</h4>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,2], gap="large")
    with c1:
        # --- Monthly Active Users Over Time ---
        if 'signup_date' in df_filt.columns:
            mau = (df_filt.groupby(pd.to_datetime(df_filt['signup_date']).dt.to_period('M'))
                   .agg({'customer_id':'nunique'})
                   .rename(columns={'customer_id':'MAU'}))
            mau = mau.reset_index()
            mau['month'] = mau['signup_date'].astype(str)
            fig_mau = px.line(mau, x='month', y='MAU', title="Monthly Active Users (MAU)", markers=True)
            fig_mau.update_traces(line_color='#53c6ff')
            st.plotly_chart(fig_mau, use_container_width=True)
            export_csv_button(mau, name="monthly_active_users")
        else:
            st.info("No signup_date column for MAU chart.")

    with c2:
        # --- Global Map of Customers (Bubble Size) ---
        geo_df = df_filt.groupby(['country_iso3']).agg(customers=('customer_id','count')).reset_index()
        fig_geo = px.scatter_geo(
            geo_df, locations="country_iso3", size="customers",
            projection="natural earth", color="customers",
            color_continuous_scale="Purples", template="plotly_dark"
        )
        fig_geo.update_geos(showland=True, landcolor="#18192c", bgcolor='rgba(0,0,0,0)')
        fig_geo.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_geo, use_container_width=True)
        export_csv_button(geo_df, name="geo_customer_bubble")

    st.divider()

    # --- Churn by Tenure ---
    st.markdown("#### Churn Rate by Customer Tenure")
    if 'tenure_months' in df_filt.columns:
        tenure_churn = df_filt.groupby('tenure_months').agg(
            churn_rate=('churned','mean'),
            users=('customer_id','count')
        ).reset_index()
        fig_tc = px.line(
            tenure_churn, x='tenure_months', y='churn_rate',
            markers=True, title="Churn by Tenure (Months)"
        )
        fig_tc.update_traces(line_color='#ff66a0')
        st.plotly_chart(fig_tc, use_container_width=True)
        export_csv_button(tenure_churn, name="churn_by_tenure")
    else:
        st.info("No tenure_months data available.")

# =================== TAB 3: Segmentation & Clusters ===================
with tabs[2]:
    st.markdown("<h4 style='margin-bottom:8px;color:#e3e8f5;'>👥 Customer Segmentation & Clusters</h4>", unsafe_allow_html=True)
    # Loyalty by Segment
    col1, col2 = st.columns(2, gap="large")
    with col1:
        seg_loy = df_filt.groupby('cluster').agg(
            avg_loyalty=('loyalty_points','mean'),
            churn_rate=('churned','mean'),
            customers=('customer_id','count')
        ).reset_index()
        fig_loy = px.bar(seg_loy, x='cluster', y='avg_loyalty', color='churn_rate', color_continuous_scale="RdPu",
                         title="Loyalty by Cluster")
        fig_loy.update_layout(margin=dict(l=0, r=0, t=24, b=0), showlegend=True)
        st.plotly_chart(fig_loy, use_container_width=True)
        export_csv_button(seg_loy, name="loyalty_by_cluster")
    with col2:
        # Radar chart/spider for clusters
        radar = pd.DataFrame()
        for metric in ['loyalty_points', 'monthly_spend', 'churned']:
            vals = df_filt.groupby('cluster')[metric].mean().reset_index()
            if radar.empty:
                radar = vals
            else:
                radar = radar.merge(vals, on="cluster")
        radar = radar.rename(columns={
            "loyalty_points": "Loyalty", "monthly_spend": "Spend", "churned": "Churn"
        })
        radar_vals = [radar[m] for m in ["Loyalty","Spend","Churn"]]
        radar_fig = go.Figure()
        for idx, row in radar.iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=[row["Loyalty"], row["Spend"], row["Churn"]],
                theta=["Loyalty", "Spend", "Churn"], fill='toself', name=f'Cluster {int(row["cluster"])}'
            ))
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Cluster Radar Profiles")
        st.plotly_chart(radar_fig, use_container_width=True)
        export_csv_button(radar, name="cluster_radar")

    st.divider()
    # Churn by Segment Table
    st.markdown("#### Churn/Loyalty Table by Segment")
    seg_tab = df_filt.groupby(['cluster','subscription_type']).agg(
        customers=('customer_id','count'),
        churn_rate=('churned','mean'),
        loyalty=('loyalty_points','mean'),
        spend=('monthly_spend','mean')
    ).reset_index()
    st.dataframe(seg_tab, use_container_width=True)
    export_csv_button(seg_tab, name="segment_table")

# =================== TAB 4: AI Insights & Export ===================
with tabs[3]:
    st.markdown("<h4 style='margin-bottom:8px;color:#e3e8f5;'>🤖 AI Insights & Export</h4>", unsafe_allow_html=True)
    st.write("Below are auto-generated insights, recommendations, and download options:")

    # Deep dive insights
    st.markdown("##### 🔍 Quick Wins / Risks")
    st.markdown("- Churn highest in clusters: " +
        ', '.join(str(c) for c in seg_loy[seg_loy['churn_rate'] > 0.25]['cluster'].tolist()))
    st.markdown("- Most loyal: Cluster {} (avg loyalty {:.0f})".format(
        seg_loy.loc[seg_loy['avg_loyalty'].idxmax(),'cluster'],
        seg_loy['avg_loyalty'].max()
    ))
    st.markdown("- Most revenue: Cluster {} (avg spend ${:.0f})".format(
        radar.loc[radar['Spend'].idxmax(),'cluster'],
        radar['Spend'].max()
    ))
    st.markdown("- At-risk countries: " + ', '.join(
        country_counts[country_counts['customers'] < 1000]['country_iso3']
    ))

    st.divider()
    st.markdown("#### 📁 Download All Tables")
    export_csv_button(df_filt, name="filtered_data_all")
    export_csv_button(seg_tab, name="segment_churn_loyalty")
    export_csv_button(radar, name="cluster_radar_all")
    export_csv_button(mau if 'mau' in locals() else pd.DataFrame(), name="MAU_by_month")
    st.info("All exports are filtered to current dashboard selections.")

# --- FOOTER ---
st.markdown("""
    <hr style="margin-top:38px; margin-bottom:3px;">
    <center>
    <span style='color:#6d729e;'>Data Science Portfolio Demo | Inspired by <a href="https://streamlit.io/gallery" style='color:#b8baff;'>Streamlit Gallery</a></span>
    </center>
""", unsafe_allow_html=True)

