# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from io import BytesIO

st.set_page_config(page_title="SME BI Dashboard", layout="wide")

# ---------- Helper functions ----------
@st.cache_data
def load_default_data(path="MANUFACTURING SMES -BIG DATA.xlsx"):
    if os.path.exists(path):
        return pd.read_excel(path)
    return None

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def safe_mean(series):
    return round(series.dropna().mean(), 2) if not series.dropna().empty else None

# ---------- Load data (default file) ----------
default_df = load_default_data()
st.sidebar.title("SME BI - Controls")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (Excel/CSV)", type=["xlsx", "xls", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Couldn't read uploaded file: {e}")
        st.stop()
elif default_df is not None:
    df = default_df.copy()
    st.sidebar.info("Using default demo dataset.")
else:
    st.sidebar.warning("No data file found. Please upload your dataset.")
    st.stop()

# Ensure column name trimming
df.columns = [c.strip() for c in df.columns]

# ---------- Basic checks for expected columns ----------
expected_cols = [
    "SME_ID","Region","Firm Size","Annual Revenue","Years in Operation","Decision Making Approach",
    "BI Tool Usage","Big Data Analytics Usage","Type of BI Tool","Data Storage Type",
    "Digital Infrastructure Score","Manager IT Skill Level","Data Security Concern","Big Data Barriers",
    "Decision Efficiency","Operational Performance","Profit Growth Rate","Customer Satisfaction Score"
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.sidebar.error(f"Warning: Dataset missing expected columns. Missing: {', '.join(missing)}")
    st.info("Proceeding with available columns. Some visuals may be empty if columns are missing.")

# ---------- Sidebar filters ----------
st.sidebar.markdown("---")
regions = df["Region"].dropna().unique().tolist() if "Region" in df.columns else []
firm_sizes = df["Firm Size"].dropna().unique().tolist() if "Firm Size" in df.columns else []
bi_usage_vals = df["BI Tool Usage"].dropna().unique().tolist() if "BI Tool Usage" in df.columns else []

sel_region = st.sidebar.multiselect("Filter by Region", options=regions, default=regions if regions else None)
sel_firm_size = st.sidebar.multiselect("Filter by Firm Size", options=firm_sizes, default=firm_sizes if firm_sizes else None)
sel_bi = st.sidebar.multiselect("Filter by BI Tool Usage", options=bi_usage_vals, default=bi_usage_vals if bi_usage_vals else None)

# Apply filters
filtered = df.copy()
if sel_region:
    if "Region" in filtered.columns:
        filtered = filtered[filtered["Region"].isin(sel_region)]
if sel_firm_size:
    if "Firm Size" in filtered.columns:
        filtered = filtered[filtered["Firm Size"].isin(sel_firm_size)]
if sel_bi:
    if "BI Tool Usage" in filtered.columns:
        filtered = filtered[filtered["BI Tool Usage"].isin(sel_bi)]

# ---------- Top header ----------
st.title("SME Business Intelligence Dashboard")
st.markdown("Light-mode interactive dashboard — upload your SME dataset to generate insights. (Demo uses sample dataset)")

# ---------- KPI cards ----------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

total_revenue = filtered["Annual Revenue"].sum() if "Annual Revenue" in filtered.columns else 0
avg_profit_growth = safe_mean(filtered["Profit Growth Rate"]) if "Profit Growth Rate" in filtered.columns else None
avg_decision_eff = safe_mean(filtered["Decision Efficiency"]) if "Decision Efficiency" in filtered.columns else None
avg_cust_satisfaction = safe_mean(filtered["Customer Satisfaction Score"]) if "Customer Satisfaction Score" in filtered.columns else None
num_smes = filtered.shape[0]

kpi_col1.metric("Total Revenue (₵)", f"{total_revenue:,.0f}")
kpi_col2.metric("Average Profit Growth (%)", f"{avg_profit_growth if avg_profit_growth is not None else 'N/A'}")
kpi_col3.metric("Avg Decision Efficiency", f"{avg_decision_eff if avg_decision_eff is not None else 'N/A'}")
kpi_col4.metric("Avg Customer Satisfaction", f"{avg_cust_satisfaction if avg_cust_satisfaction is not None else 'N/A'}")

st.markdown(f"**SMEs in current filter:** {num_smes}")

st.markdown("---")

# ---------- Charts row 1 ----------
chart_col1, chart_col2 = st.columns((2,1))

with chart_col1:
    st.subheader("Annual Revenue by Region")
    if "Region" in filtered.columns and "Annual Revenue" in filtered.columns:
        rev_region = filtered.groupby("Region", as_index=False)["Annual Revenue"].sum().sort_values("Annual Revenue", ascending=False)
        fig_rev = px.bar(rev_region, x="Region", y="Annual Revenue", color="Annual Revenue",
                         color_continuous_scale="Blues", labels={"Annual Revenue":"Revenue (₵)"})
        fig_rev.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=420)
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("Annual Revenue or Region column missing.")

with chart_col2:
    st.subheader("BI Tool Usage Share")
    if "BI Tool Usage" in filtered.columns:
        bi_counts = filtered["BI Tool Usage"].value_counts().reset_index()
        bi_counts.columns = ["BI Tool Usage","count"]
        fig_pie = px.pie(bi_counts, names="BI Tool Usage", values="count", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=420)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("BI Tool Usage column missing.")

st.markdown("---")

# ---------- Charts row 2 ----------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Profit Growth by BI Tool Usage")
    if "Profit Growth Rate" in filtered.columns and "BI Tool Usage" in filtered.columns:
        fig_box = px.box(filtered, x="BI Tool Usage", y="Profit Growth Rate", points="all",
                         labels={"Profit Growth Rate":"Profit Growth (%)"})
        fig_box.update_layout(height=420, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Profit Growth Rate or BI Tool Usage missing.")

with row2_col2:
    st.subheader("Digital Infrastructure vs Operational Performance")
    if "Digital Infrastructure Score" in filtered.columns and "Operational Performance" in filtered.columns:
        fig_scatter = px.scatter(filtered, x="Digital Infrastructure Score", y="Operational Performance",
                                 size="Annual Revenue" if "Annual Revenue" in filtered.columns else None,
                                 color="BI Tool Usage" if "BI Tool Usage" in filtered.columns else None,
                                 labels={"Digital Infrastructure Score":"Digital Infra Score", "Operational Performance":"Operational Perf"},
                                 hover_data=["SME_ID"] if "SME_ID" in filtered.columns else None)
        fig_scatter.update_layout(height=420, margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Digital Infrastructure or Operational Performance missing.")

st.markdown("---")

# ---------- Insights text box ----------
st.subheader("Quick Insights")
insights = []
if avg_decision_eff:
    insights.append(f"- Average Decision Efficiency in filter: **{avg_decision_eff}**")
if avg_profit_growth:
    insights.append(f"- Average Profit Growth: **{avg_profit_growth}%**")
if "Digital Infrastructure Score" in filtered.columns and "Operational Performance" in filtered.columns:
    corr = filtered[["Digital Infrastructure Score","Operational Performance"]].corr().iloc[0,1]
    insights.append(f"- Correlation (DigitalInfra vs OperationalPerf): **{round(corr,2)}**")
if insights:
    for i in insights:
        st.markdown(i)
else:
    st.markdown("No quick insights available for selected filters.")

st.markdown("---")

# ---------- Data and download ----------
st.subheader("Filtered Data Preview")
st.dataframe(filtered.head(200))

csv_bytes = to_csv_bytes(filtered)
st.download_button(label="Download filtered data as CSV", data=csv_bytes, file_name="filtered_smes.csv", mime="text/csv")

st.markdown("Developed by Regina Glavee-Geo — Demo dashboard showing BI insights for manufacturing SMEs.")
