Customer Churn Analysis and Automated AI Insights Dashboard
-----------------------------------------------------------

Project by Aryan Kaushik

-----------------------------------------------------------
1. Overview and Project Goals

The Customer Analytics AI project is an end-to-end demonstration of how to approach SaaS customer data from simulation and analysis through to production-quality, executive-level dashboards.

This project achieves two things:
- First; it simulates realistic SaaS customer data and deeply analyzes churn, retention, loyalty, and segment health using a full data science workflow in Python.
- Second; it turns these insights into a modern, filterable, interactive dashboard that any business user, product manager, or executive could use to drive real action.

This project is valuable to others because it shows how to:
- Generate high-quality synthetic data for SaaS scenarios; perfect for those without real data
- Run an entire analysis; from EDA to churn segmentation and advanced visualization
- Build a professional dashboard; ready for presentation or online demo, with all business logic and “AI insights” built-in

All code, analysis, and the dashboard are open source. You can adapt the methods and the dashboard for your own SaaS, B2B, or product analytics needs.

-----------------------------------------------------------
2. Folder and File Structure

customer_analytics_ai/
    app.py                        # Streamlit dashboard
    customer_churn_project.ipynb  # Full data science analysis notebook
    streamflow_customers.csv      # Synthetic customer dataset
    streamflow_logo.png           # Branding/logo for dashboard
    requirements.txt              # All libraries needed
    readme.txt                    # This file
    (screenshots/, ai_agent_report.txt, etc.)

-----------------------------------------------------------
3. The Data Science Churn Analysis (customer_churn_project.ipynb)

The notebook demonstrates the full SaaS customer analytics workflow, including:

- Synthetic Data Generation;
    The notebook programmatically generates a dataset of over 23,000 unique SaaS customers. It assigns each customer a country, subscription type (Free, Basic, Premium, Enterprise), monthly spend, engagement score, loyalty points, and uses randomization and logic to simulate churn and tenure.
    Each customer is given realistic dates for signup and churn, as well as fields for KMeans clustering.

- Exploratory Data Analysis (EDA);
    The notebook explores the overall health of the user base; visualizing subscription distribution, spend, loyalty, engagement, churn rates, and tenure.
    It computes KPIs like churn rate, average loyalty, premium conversion, and user segmentation.

- Segmentation and Clustering;
    Customers are grouped into clusters using KMeans, based on behavioral and value-based attributes. The notebook analyzes cluster-level loyalty, churn, and spend to identify high-value and at-risk segments.

- Churn and Retention Modeling;
    The notebook calculates churn rates by subscription type, by tenure, by cluster, and by country. It visualizes retention with survival curves, boxplots for engagement by churn status, and bar charts for feature importance.
    It uses feature correlation analysis to identify what most predicts customer loss.

- Business Insights and Recommendations;
    Each major analysis section ends with specific written observations; for example, “Enterprise churn is lowest; focus onboarding on Free users,” or “Engagement and missed payments are top churn predictors.”
    The analysis is designed for presentation or handoff to product/business teams.

The notebook also provides a clean summary of actionable insights for each chart, and the key recommendations are exported to `ai_agent_report.txt` for use in the dashboard.

-----------------------------------------------------------
4. Automated AI Insights Dashboard (app.py)

This project includes a fully interactive, modern SaaS analytics dashboard built in Streamlit with Plotly. The dashboard is designed for business users and executives as well as technical teams.

Features include:

- Responsive Sidebar;
    Branded with logo; modern dropdown, checkable filters for Subscription Type, Cluster, and Month. CSV upload option for your own data.

- Live KPIs;
    Key stats like total users, churn rate, loyalty, premium percent, and number of countries; all respond live to filter changes.

- Executive Visuals;
    4 main visualizations in the “Overview & Metrics” tab:
        1. Churn Rate by Subscription Type; highlights where customer loss is highest
        2. Subscription Breakdown; pie chart showing mix of Free, Basic, Premium, Enterprise users
        3. Feature Correlation with Churn; shows which numeric variables most predict churn
        4. Top Clusters by Spend; bar chart of clusters with highest revenue

- Each chart has an “Overall Observation” box written for a non-technical audience; these summaries do not change with filters and reflect the full data.

- Visual Trends Tab;
    Includes Monthly Active Users (MAU) over time with smoothing, world map of customer locations, and churn rate by customer tenure with rolling mean.

- Segmentation & Clusters Tab;
    Shows loyalty and spend by cluster, and a churn/loyalty table by segment.

- AI Insights Tab;
    Provides multi-point, filter-responsive business insights. Includes download buttons for filtered datasets and segment tables.

- About Tab;
    Explains the project, data, and how to use the dashboard.

- All visuals and tables (except “Overall Observations”) update instantly with any sidebar filter; perfect for scenario analysis.

- The dashboard is ready for deployment to Streamlit Community Cloud or any cloud provider; just push to GitHub and connect.

-----------------------------------------------------------
5. How to Use This Project

To use the churn analysis notebook:
    - Open `customer_churn_project.ipynb` in Jupyter or VS Code.
    - Run all cells to generate the data, visualize trends, and review key churn and retention insights.

To use the dashboard locally:
    - Install all dependencies from requirements.txt (see below).
    - Run: `streamlit run app.py`
    - The dashboard will open in your browser at localhost:8501.
    - You can filter the data, upload your own CSV, and download filtered tables.

To deploy the dashboard online (Streamlit Community Cloud recommended):
    - Push all files to your GitHub repository.
    - Go to https://share.streamlit.io and connect your repo.
    - Select `app.py` as the entry point.
    - The dashboard will be instantly available at your personal Streamlit URL.
    - Add the public link to your portfolio and project descriptions.

To use your own data:
    - Prepare a CSV with the following required columns: customer_id, subscription_type, cluster, churned, signup_date, monthly_spend, loyalty_points.
    - Recommended columns: country, tenure_months, engagement or usage fields.
    - Upload your CSV in the dashboard sidebar.
    - All visuals, tables, and AI insights will update with your data.

-----------------------------------------------------------
6. Business Value and Use Cases

This project is designed to:
    - Help SaaS, product, and data teams understand and reduce customer churn
    - Provide non-technical stakeholders with instant, self-serve analytics and executive-level insights
    - Demonstrate end-to-end, production-quality data science for resumes, portfolios, or client demos
    - Serve as a template for anyone building a customer 360 dashboard or automated business insights tool

All code, data, and text are open source and ready for adaptation.

-----------------------------------------------------------
7. Project Credits

Dashboard, analysis, and automation by Aryan Kaushik.

For questions or feedback, see:
    https://github.com/aryankaushik89/aryankaushik89.github.io/tree/main/customer_analytics_ai

-----------------------------------------------------------
