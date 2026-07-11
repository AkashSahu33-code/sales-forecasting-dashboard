import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# ------------------ LOAD DATA (cached) ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales.csv', parse_dates=['Order Date', 'Ship Date'])
    monthly = pd.read_csv('monthly_sales.csv', index_col=0, parse_dates=True)
    comparison = pd.read_csv('model_comparison.csv')
    segment_forecasts = pd.read_csv('segment_forecasts.csv', parse_dates=['Date'])
    anomalies = pd.read_csv('anomaly_data.csv', index_col=0, parse_dates=True)
    clusters = pd.read_csv('cluster_data.csv')
    return df, monthly, comparison, segment_forecasts, anomalies, clusters

@st.cache_resource
def load_model():
    with open('sarima_model.pkl', 'rb') as f:
        return pickle.load(f)

df, monthly, comparison, segment_forecasts, anomalies, clusters = load_data()
sarima_model = load_model()

# ------------------ SIDEBAR NAV ------------------
page = st.sidebar.radio("Navigate", 
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# ================== PAGE 1: SALES OVERVIEW ==================
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby(df['Order Date'].dt.year)['Sales'].sum()
        fig, ax = plt.subplots()
        yearly.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        fig, ax = plt.subplots()
        monthly.plot(ax=ax, color='darkorange')
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    st.subheader("Sales by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        region_filter = st.multiselect("Filter by Region", df['Region'].unique(), default=list(df['Region'].unique()))
    with c2:
        category_filter = st.multiselect("Filter by Category", df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_filter) & df['Category'].isin(category_filter)]
    pivot = filtered.groupby(['Region', 'Category'])['Sales'].sum().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind='bar', ax=ax)
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

# ================== PAGE 2: FORECAST EXPLORER ==================
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")

    dimension = st.selectbox("Select dimension", ["Overall"] + sorted(segment_forecasts['Segment'].unique().tolist()))
    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    if dimension == "Overall":
        forecast_result = sarima_model.get_forecast(steps=horizon)
        pred = forecast_result.predicted_mean
        ci = forecast_result.conf_int()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly.index[-12:], monthly.iloc[-12:], label='Recent Actual')
        ax.plot(pred.index, pred.values, label='Forecast', marker='o', color='green')
        ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.write("**Forecast values:**")
        st.dataframe(pred.rename("Forecasted Sales"))
    else:
        seg_data = segment_forecasts[segment_forecasts['Segment'] == dimension].head(horizon)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(seg_data['Date'], seg_data['Forecast'], marker='o', color='green', label=f'{dimension} Forecast')
        ax.legend()
        st.pyplot(fig)
        st.dataframe(seg_data)

    st.subheader("Model Performance (SARIMA — Best Model)")
    sarima_row = comparison[comparison['Model'] == 'SARIMA']
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{sarima_row['MAE'].values[0]:,.2f}")
    m2.metric("RMSE", f"{sarima_row['RMSE'].values[0]:,.2f}")
    m3.metric("MAPE", f"{sarima_row['MAPE'].values[0]:.2f}%")

    st.subheader("All Models Comparison")
    st.dataframe(comparison)

# ================== PAGE 3: ANOMALY REPORT ==================
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(anomalies.index, anomalies['Sales'], label='Weekly Sales', color='steelblue')
    flagged = anomalies[anomalies['IF_Anomaly'] | anomalies['Zscore_Anomaly']]
    ax.scatter(flagged.index, flagged['Sales'], color='red', s=80, zorder=5, label='Anomaly')
    ax.legend()
    ax.set_title("Detected Anomalies in Weekly Sales")
    st.pyplot(fig)

    st.subheader("Detected Anomaly Dates")
    display_cols = ['Sales', 'IF_Anomaly', 'Zscore_Anomaly']
    st.dataframe(flagged[display_cols].sort_values('Sales', ascending=False))

# ================== PAGE 4: PRODUCT DEMAND SEGMENTS ==================
elif page == "Product Demand Segments":
    st.title("📦 Product Demand Segments")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_map = {0: 'steelblue', 1: 'green', 2: 'orange', 3: 'red'}
    for cluster_id in sorted(clusters['Cluster'].unique()):
        subset = clusters[clusters['Cluster'] == cluster_id]
        ax.scatter(subset['PCA1'], subset['PCA2'], color=colors_map.get(cluster_id, 'gray'),
                   s=150, label=subset['Cluster_Label'].iloc[0])
        for _, row in subset.iterrows():
            ax.annotate(row['Sub-Category'], (row['PCA1'], row['PCA2']), fontsize=8)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Categories by Demand Cluster")
    st.dataframe(clusters[['Sub-Category', 'Cluster_Label', 'Total_Sales', 'Growth_Rate_Pct', 'Volatility']].sort_values('Cluster_Label'))
