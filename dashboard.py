import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix 
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="SME Production & Sales Dashboard", layout="wide", page_icon="📊")

DEFAULT_DATA_PATH = "/mnt/data/plastic_rubber_sme_dataset_extended.csv"

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path, engine="openpyxl")
    return df

def compute_derived(df):
    if "Total_Revenue" not in df.columns and all(x in df.columns for x in ["Units_Sold","Unit_Price"]):
        df["Total_Revenue"] = df["Units_Sold"] * df["Unit_Price"]
    if "Total_Cost" not in df.columns and all(x in df.columns for x in ["Units_Produced","Cost_Per_Unit"]):
        df["Total_Cost"] = df["Units_Produced"] * df["Cost_Per_Unit"]
    if "Profit" not in df.columns and all(x in df.columns for x in ["Total_Revenue","Total_Cost"]):
        df["Profit"] = df["Total_Revenue"] - df["Total_Cost"]
    df["Production_Efficiency"] = df.apply(lambda r: r["Units_Sold"]/r["Units_Produced"] if r["Units_Produced"]>0 else np.nan, axis=1)
    df["Margin_per_unit"] = df["Unit_Price"] - df["Cost_Per_Unit"] if all(x in df.columns for x in ["Unit_Price","Cost_Per_Unit"]) else np.nan
    return df

def download_button(df, filename="dataset.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.download_button("Download filtered dataset", towrite, file_name=filename, mime="text/csv")

# -------------------------
# Load Data
# -------------------------
st.sidebar.header("Load data")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])
use_default = st.sidebar.checkbox("Use default dataset", value=True)

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, engine="openpyxl")
elif use_default and os.path.exists(DEFAULT_DATA_PATH):
    df = load_data(DEFAULT_DATA_PATH)
else:
    st.stop()

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

st.sidebar.header("Download Data")
download_button(filtered, filename="filtered_product_data.csv")

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
# Summary Stats
# -------------------------
st.subheader("Summary Statistics by Product")
prod_summary = filtered.groupby("Product_Name").agg(
    Units_Produced=("Units_Produced","sum"),
    Units_Sold=("Units_Sold","sum"),
    Total_Revenue=("Total_Revenue","sum"),
    Total_Cost=("Total_Cost","sum"),
    Profit=("Profit","sum"),
    Production_Efficiency=("Production_Efficiency","mean")
).round(2)
st.dataframe(prod_summary)

if region_col:
    st.subheader(f"Summary by {region_col}")
    region_summary = filtered.groupby(region_col).agg(
        Units_Produced=("Units_Produced","sum"),
        Units_Sold=("Units_Sold","sum"),
        Total_Revenue=("Total_Revenue","sum"),
        Total_Cost=("Total_Cost","sum"),
        Profit=("Profit","sum"),
        Production_Efficiency=("Production_Efficiency","mean")
    ).round(2)
    st.dataframe(region_summary)

# -------------------------
# Top Recommendations
# -------------------------
st.subheader("Top 3 Actionable Recommendations")
profit_threshold_input = st.number_input("Profit threshold for recommendations (0 = median)", value=0.0, format="%.2f")
profit_thr = filtered["Profit"].median() if profit_threshold_input==0 else profit_threshold_input

def generate_recommendations(df, threshold):
    recs = []
    summary = df.groupby("Product_Name").agg({"Units_Sold":"sum","Units_Produced":"sum","Profit":"sum","Total_Revenue":"sum"}).reset_index()
    summary["Sell_Rate"] = summary["Units_Sold"]/summary["Units_Produced"]
    summary["MarginPct"] = summary.apply(lambda r: ((r["Profit"]/r["Total_Revenue"])*100) if r["Total_Revenue"]>0 else 0, axis=1)

    high = summary[(summary["Units_Sold"]>summary["Units_Sold"].median()) & (summary["Profit"]>threshold)].sort_values("Profit", ascending=False)
    if not high.empty: recs.append(f"Produce more: **{high.iloc[0]['Product_Name']}** — high demand & profit.")
    
    low = summary[summary["Sell_Rate"]<0.6].sort_values("Sell_Rate")
    if not low.empty: recs.append(f"Reduce production: **{low.iloc[0]['Product_Name']}** — low sell-through rate.")
    
    candidate = summary[(summary["Units_Sold"]>summary["Units_Sold"].median()) & (summary["MarginPct"]<5)]
    if not candidate.empty: recs.append(f"Consider increasing price: **{candidate.iloc[0]['Product_Name']}** — strong demand, low margin.")

    return recs[:3] if recs else ["No strong recommendations detected."]

top_recs = generate_recommendations(filtered, profit_thr)
for r in top_recs:
    st.markdown("• " + r)

# -------------------------
# Charts
# -------------------------
st.subheader("Charts for Decision-making")
col1,col2 = st.columns((2,1))

with col1:
    fig1 = px.bar(filtered.groupby("Product_Name")[["Units_Produced","Units_Sold"]].sum().reset_index(),
                  x="Product_Name", y=["Units_Produced","Units_Sold"], barmode="group", 
                  title="Production vs Sales per Product")
    st.plotly_chart(fig1, use_container_width=True)

    profit_df = filtered.groupby("Product_Name")["Profit"].sum().reset_index().sort_values("Profit", ascending=False)
    fig2 = px.bar(profit_df, x="Product_Name", y="Profit", color="Profit", color_continuous_scale=px.colors.sequential.Teal)
    fig2.add_hline(y=profit_thr, line_dash="dash", line_color="red",
                   annotation_text=f"Profit Threshold = GHS {profit_thr:,.2f}", annotation_position="top right")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    top_rev = filtered.groupby("Product_Name")[["Total_Revenue","Total_Cost"]].sum().reset_index().sort_values("Total_Revenue", ascending=False).head(10)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=top_rev["Product_Name"], y=top_rev["Total_Revenue"], name="Revenue", marker_color='green'))
    fig3.add_trace(go.Bar(x=top_rev["Product_Name"], y=top_rev["Total_Cost"], name="Cost", marker_color='red'))
    fig3.update_layout(barmode='group', title="Revenue vs Cost (Top 10 Products)")
    st.plotly_chart(fig3, use_container_width=True)

if region_col:
    region_profit = filtered.groupby([region_col,"Product_Name"])["Profit"].sum().reset_index()
    fig_hm = px.density_heatmap(region_profit, x="Product_Name", y=region_col, z="Profit", color_continuous_scale="Viridis",
                                title="Profit by Product × Region")
    st.plotly_chart(fig_hm, use_container_width=True)

# -------------------------
# Product Pattern Analysis
# -------------------------
st.subheader("Product Performance Patterns")
prod_list = st.multiselect("Select Product(s) to analyze", options=sorted(filtered["Product_Name"].unique()), 
                           default=sorted(filtered["Product_Name"].unique())[:3])

pattern_recs = []
for product in prod_list:
    prod_data = filtered[filtered["Product_Name"]==product]
    total_profit = prod_data["Profit"].sum()
    total_cost = prod_data["Total_Cost"].sum()
    units_prod = prod_data["Units_Produced"].sum()
    units_sold = prod_data["Units_Sold"].sum()
    
    if total_profit < profit_thr:
        pattern_recs.append(f"⚠️ {product}: Profit GHS {total_profit:,.2f} below threshold → review pricing or costs.")
    if units_prod > units_sold * 1.2:
        pattern_recs.append(f"📦 {product}: Overproduction detected (Produced {units_prod}, Sold {units_sold}) → reduce production.")
    if total_cost > prod_data["Total_Revenue"].sum():
        pattern_recs.append(f"💰 {product}: Cost GHS {total_cost:,.2f} exceeds revenue → review inputs or pricing.")

if pattern_recs:
    for rec in pattern_recs:
        st.markdown(rec)
else:
    st.markdown("✅ No concerning patterns detected. Performance is stable.")

# -------------------------
# Stacking ML Model (Hidden)
# -------------------------
ml_features = ["Units_Produced","Units_Sold","Unit_Price","Cost_Per_Unit","Production_Efficiency"]

df_ml = filtered.dropna(subset=ml_features+["Profit"]).copy()
df_ml["High_Profit"] = (df_ml["Profit"]>=profit_thr).astype(int)

X = pd.get_dummies(df_ml[ml_features], drop_first=True)
y = df_ml["High_Profit"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

estimators = [
    ("lr", LogisticRegression(max_iter=500)),
    ("svm", SVC(probability=True)),
    ("mlp", MLPClassifier(max_iter=500))
]
stack_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
pipeline = Pipeline([("scaler", StandardScaler()), ("stack", stack_clf)])
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate Stacked Model (Hidden)
# -------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps['stack'], "predict_proba") else None

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test))>1 else None
}

# -------------------------
# Display Metrics in Dashboard
# -------------------------
st.subheader("Model Evaluation Metrics")
st.table(pd.DataFrame(metrics, index=[0]).T.rename(columns={0:"Value"}))


df_ml["High_Profit_Pred"] = pipeline.predict(X)
high_profit_products = df_ml[df_ml["High_Profit_Pred"]==1]["Product_Name"].unique()

st.subheader("Predicted High-Profit Products")
if len(high_profit_products) > 0:
    for p in high_profit_products:
        st.markdown(f"✅ **{p}** predicted as high-profit product.")
else:
    st.markdown("No products predicted as high-profit at this threshold.")

st.caption("Enhanced SME Dashboard: KPIs, charts, product patterns, dynamic recommendations, and hidden stacking ML model.")
