import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="E-Commerce Analytics & LTV Dashboard", layout="wide")

# App Header
st.title("🛒 E-Commerce Customer Segmentation & LTV Dashboard")
st.markdown("Analyze customer behavior personas and predict future customer lifetime value (LTV) using machine learning.")

# Load Data (Cached for performance)
@st.cache_data
def load_data():
    rfm_df = pd.read_csv("data/customer_rfm_segments.csv")
    ltv_df = pd.read_csv("data/customer_ltv_predictions.csv")
    return rfm_df, ltv_df

try:
    rfm_df, ltv_df = load_data()
except FileNotFoundError:
    st.error("Processed data files not found! Please run your data cleaning and modeling scripts first.")
    st.stop()

# Sidebar Navigation
st.sidebar.header("Navigation")
tab = st.sidebar.radio("Go to:", ["Executive Overview", "Customer Segments (RFM)", "LTV Prediction Simulator"])

if tab == "Executive Overview":
    st.subheader("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(rfm_df):,}")
    col2.metric("Total Revenue", f"£{rfm_df['Monetary'].sum():,.2f}")
    col3.metric("Avg Customer Spend", f"£{rfm_df['Monetary'].mean():,.2f}")
    col4.metric("Avg Order Frequency", f"{rfm_df['Frequency'].mean():.1f} orders")
    
    st.markdown("---")
    
    # Revenue distribution by segment
    segment_revenue = rfm_df.groupby('Customer_Segment')['Monetary'].sum().reset_index()
    fig = px.bar(segment_revenue, x='Customer_Segment', y='Monetary', 
                 title="Total Revenue Contribution by Customer Segment",
                 labels={'Monetary': 'Total Revenue (£)', 'Customer_Segment': 'Customer Segment'},
                 color='Customer_Segment')
    st.plotly_chart(fig, use_container_width=True)

elif tab == "Customer Segments (RFM)":
    st.subheader("👥 Customer Segmentation Analysis")
    
    selected_segment = st.selectbox("Filter by Segment:", options=['All'] + list(rfm_df['Customer_Segment'].unique()))
    
    filtered_df = rfm_df if selected_segment == 'All' else rfm_df[rfm_df['Customer_Segment'] == selected_segment]
    
    st.markdown(f"Showing **{len(filtered_df):,}** customers for segment: **{selected_segment}**")
    st.dataframe(filtered_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Customer_Segment']].head(100), use_container_width=True)
    
    # Scatter plot: Recency vs Monetary
    fig_scatter = px.scatter(filtered_df, x='Recency', y='Monetary', color='Customer_Segment',
                             size='Frequency', hover_data=['CustomerID'],
                             title="Customer Recency vs Monetary Value (Size = Frequency)")
    st.plotly_chart(fig_scatter, use_container_width=True)

elif tab == "LTV Prediction Simulator":
    st.subheader("🔮 3-Month Customer Lifetime Value (LTV) Simulator")
    st.markdown("Simulate how historical purchasing behavior influences predicted future customer spend.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_recency = st.slider("Days Since Last Purchase (Recency)", 1, 365, 30)
        sim_frequency = st.slider("Total Historical Orders (Frequency)", 1, 50, 5)
    with col2:
        sim_monetary_total = st.number_input("Total Historical Spend (£)", 10.0, 10000.0, 500.0)
        sim_monetary_mean = st.number_input("Average Order Value (£)", 5.0, 500.0, 50.0)
    
    # Simple interactive rule-of-thumb estimator matching your ML logic structure
    estimated_ltv = (sim_monetary_mean * (sim_frequency / 3.0)) * max(0.1, (1 - (sim_recency / 400.0)))
    
    st.info(f"### Predicted Future 3-Month Spend: **£{max(0.0, estimated_ltv):,.2f}**")