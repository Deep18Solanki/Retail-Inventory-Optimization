import streamlit as st
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inventory_logic import calculate_inventory_kpis

st.set_page_config(page_title="Retail Inventory Optimization", layout="wide")

st.title("🛒 Retail Sales Forecasting & Inventory Optimization System")

# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_sales.csv")

df = load_data()
model = joblib.load("models/rf_sales_model.pkl")

# Generate simple mock predictions for the dashboard view
st.sidebar.header("Configuration")
service_level = st.sidebar.slider("Target Service Level (%)", 90, 99, 95)

st.subheader("Historical Sales Data")
st.dataframe(df.tail())

st.subheader("Inventory Replenishment Recommendations")
# Predict using the last available data point for each product
latest_data = df.groupby('Product_ID').tail(1)
features = ['DayOfWeek', 'Month', 'Sales_Lag_1', 'Sales_Lag_7', 'Rolling_Mean_7']
predictions = model.predict(latest_data[features])

recommendations = calculate_inventory_kpis(df, predictions)
st.table(recommendations)

st.subheader("Sales Trend Analysis")
selected_product = st.selectbox("Select Product", df['Product_ID'].unique())
prod_df = df[df['Product_ID'] == selected_product]
st.line_chart(prod_df.set_index('Date')['Sales'])