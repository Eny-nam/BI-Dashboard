import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Product Analytics — SME BI Dashboard",
                   layout="wide",
                   page_icon="📊")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

def compute_derived(df):
    if "Total_Revenue" not in df.columns and "Units_Sold" in df.columns and "Unit_Price" in df.columns:
        df["Total_Revenue"] = df["Units_Sold"] * df["Unit_Price"]
    if "Total_Cost" not in df.columns and "Units_Produced" in df.columns and "Cost_Per_Unit" in df.columns:
        df["Total_Cost"] = df["Units_Produced"] * df["Cost_Per_Unit"]
    if "Profit" not in df.columns and "Total_Revenue" in df.columns and "Total_Cost" in df.columns:
        df["Profit"] = df["Total_Revenue"] - df["Total_Cost"]
    df["Production_Efficiency"] = df.apply(lambda r: (r["Units_Sold"]/r["Units_Produced"]) if r["Units_Produced"]>0 else np.nan, axis=1)
    df["Margin_per_unit"] = df["Unit_Price"] - df["Cost_Per_Unit"] if ("Unit_Price" in df.columns and "Cost_Per_Unit" in df.columns) else np.nan
    return df

def download_button(df, filename="dataset.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.download_button("Download filtered dataset (CSV)", towrite, file_name=filename, mime="text/csv")

# -------------------------
# Load Data
# -------------------------
st.sidebar.header("Load data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel file", type=["csv","xlsx"])
if uploaded_file is None:
    st.sidebar.warning("Please upload a CSV or Excel file to continue.")
    st.stop()
else:
    df = load_data(uploaded_file)

df.columns = df.columns.str.strip()
df = compute_derived(df)

required_cols = ["Product_Name","Units_Produced","Units_Sold","Unit_Price","Cost_Per_Unit","Total_Revenue","Total_Cost","Profit"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.sidebar.error(f"Dataset missing required columns: {missing}")
    st.stop()

# -------------------------
# Filters
# -------------------------
st.sidebar.header("Filters")
product_filter = st.sidebar.multiselect("Product(s)", options=sorted(df["Product_Name"].unique()), default=sorted(df["Product_Name"].unique()))

region_col = "Region/Market" if "Region/Market" in df.columns else ("Region" if "Region" in df.columns else None)
region_filter = st.sidebar.multiselect("Region(s)", options=sorted(df[region_col].unique()), default=sorted(df[region_col].unique())) if region_col else None

filtered = df[df["Product_Name"].isin(product_filter)]
if region_col and region_filter:
    filtered = filtered[filtered[region_col].isin(region_filter)]

# -------------------------
# Top 3 Recommendations
# -------------------------
def generate_recommendations(df):
    recs = []
    summary = df.groupby("Product_Name").agg({"Units_Sold":"sum","Units_Produced":"sum","Profit":"sum","Total_Revenue":"sum"}).reset_index()
    summary["Sell_Rate"] = summary["Units_Sold"]/summary["Units_Produced"]
    summary["MarginPct"] = summary.apply(lambda r: ((r["Profit"]/r["Total_Revenue"])*100) if r["Total_Revenue"]>0 else 0, axis=1)

    # Produce more
    high = summary[(summary["Units_Sold"]>summary["Units_Sold"].median()) & (summary["Profit"]>0)].sort_values("Profit", ascending=False)
    if not high.empty: recs.append(f"Produce more: **{high.iloc[0]['Product_Name']}** — high demand & profit.")

    # Reduce production
    low = summary[summary["Sell_Rate"]<0.6].sort_values("Sell_Rate")
    if not low.empty: recs.append(f"Reduce production: **{low.iloc[0]['Product_Name']}** — low sell-through rate.")

    # Increase price
    candidate = summary[(summary["Units_Sold"]>summary["Units_Sold"].median()) & (summary["MarginPct"]<5)]
    if not candidate.empty: recs.append(f"Consider increasing price: **{candidate.iloc[0]['Product_Name']}** — strong demand, low margin.")

    if not recs: recs = ["No strong recommendations detected."]
    return recs[:3]

st.header("Top 3 Actionable Recommendations")
top_recs = generate_recommendations(filtered)
for r in top_recs:
    st.markdown("• " + r)
st.markdown("---")

# -------------------------
# KPIs
# -------------------------
st.subheader("Key KPIs")
k1,k2,k3,k4 = st.columns(4)
total_rev = filtered["Total_Revenue"].sum()
total_cost = filtered["Total_Cost"].sum()
total_profit = filtered["Profit"].sum()
avg_eff = filtered["Production_Efficiency"].mean()

k1.metric("Total Revenue", f"GHS {total_rev:,.2f}", delta=f"{total_profit:,.2f}")
k2.metric("Total Cost", f"GHS {total_cost:,.2f}")
k3.metric("Total Profit", f"GHS {total_profit:,.2f}")
k4.metric("Avg Production Efficiency", f"{avg_eff:.2%}" if not np.isnan(avg_eff) else "N/A")

# -------------------------
# Charts, ML, etc.
# -------------------------
# (All your original charting, ML, and model sections remain exactly as in your current code)
