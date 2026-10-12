
import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import load_data, cluster_data

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏙️",
    layout="wide",
)

st.title("🏙️ Real Estate Buyer Segmentation & Investment Profiling")
st.caption("Machine-learning buyer intelligence dashboard | Unified Mentor Project")


@st.cache_data
def get_results():
    clients, properties, data = load_data()
    clustered, metrics = cluster_data(data)
    return clients, properties, clustered, metrics


try:
    clients, properties, data, metrics = get_results()
except Exception as e:
    st.error("Project data load nahi ho paaya.")
    st.exception(e)
    st.stop()


# Ensure every dashboard column exists
required_defaults = {
    "client_id": "",
    "client_type": "Unknown",
    "country": "Unknown",
    "region": "Unknown",
    "acquisition_purpose": "Unknown",
    "loan_applied": "Unknown",
    "satisfaction_score": 0,
    "age": 0,
    "sold_properties": 0,
    "avg_purchase_price": 0,
    "total_purchase_value": 0,
    "cluster": 0,
}

for col, default in required_defaults.items():
    if col not in data.columns:
        data[col] = default

for col in ["sold_properties", "avg_purchase_price",
            "total_purchase_value", "satisfaction_score", "age"]:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

for col in ["country", "region", "acquisition_purpose",
            "client_type", "loan_applied"]:
    data[col] = data[col].fillna("Unknown").astype(str)


# Sidebar filters
st.sidebar.header("Filter buyers")


def choices(col):
    return sorted(data[col].dropna().astype(str).unique().tolist())


country = st.sidebar.multiselect("Country", choices("country"))
region = st.sidebar.multiselect("Region", choices("region"))
purpose = st.sidebar.multiselect("Acquisition purpose", choices("acquisition_purpose"))
ctype = st.sidebar.multiselect("Client type", choices("client_type"))

filtered = data.copy()

for col, vals in [
    ("country", country),
    ("region", region),
    ("acquisition_purpose", purpose),
    ("client_type", ctype),
]:
    if vals:
        filtered = filtered[filtered[col].astype(str).isin(vals)]


# KPI cards
st.subheader("Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Clients in View", f"{len(filtered):,}")

investment_count = (
    filtered["acquisition_purpose"].str.lower() == "investment"
).sum()
c2.metric("Investment Buyers", f"{investment_count:,}")

loan_count = (
    filtered["loan_applied"].str.lower().isin(["yes", "true", "1"])
).sum()
c3.metric("Loan Applicants", f"{loan_count:,}")

c4.metric("Sold Properties", f"{int(filtered['sold_properties'].sum()):,}")


# Segmentation charts
st.subheader("Buyer Segmentation Overview")

left, right = st.columns(2)

with left:
    counts = (
        filtered["cluster"].value_counts().sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Buyers")
    )
    counts["Cluster"] = counts["Cluster"].map(lambda x: f"C{int(x) + 1}")
    fig = px.bar(
        counts, x="Cluster", y="Buyers",
        title="Buyer Count by Cluster", text="Buyers"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    purpose_counts = (
        filtered.groupby(["cluster", "acquisition_purpose"])
        .size().reset_index(name="Buyers")
    )
    purpose_counts["Cluster"] = purpose_counts["cluster"].map(
        lambda x: f"C{int(x) + 1}"
    )
    fig = px.bar(
        purpose_counts, x="Cluster", y="Buyers",
        color="acquisition_purpose", barmode="group",
        title="Acquisition Purpose by Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)


# Purchase profile
st.subheader("Investor Behaviour")

left, right = st.columns(2)

with left:
    price_profile = (
        filtered.groupby("cluster", as_index=False)
        .agg(
            avg_purchase_value=("avg_purchase_price", "mean"),
            total_purchase_value=("total_purchase_value", "sum"),
        )
    )
    price_profile["Cluster"] = price_profile["cluster"].map(
        lambda x: f"C{int(x) + 1}"
    )
    fig = px.bar(
        price_profile, x="Cluster", y="avg_purchase_value",
        title="Average Purchase Value by Cluster",
        labels={"avg_purchase_value": "Average Purchase Value"}
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.histogram(
        filtered, x="satisfaction_score",
        color="acquisition_purpose", nbins=5,
        title="Satisfaction Score Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


# Geographic analysis
st.subheader("Geographic Buyer Analysis")

geo = (
    filtered.groupby(["country", "region"], as_index=False)
    .agg(
        buyers=("client_id", "nunique"),
        investment_buyers=(
            "acquisition_purpose",
            lambda s: (s.astype(str).str.lower() == "investment").sum()
        ),
    )
    .sort_values("buyers", ascending=False)
    .head(20)
)

if not geo.empty:
    fig = px.bar(
        geo, x="region", y="buyers", color="country",
        title="Top Regions by Buyer Count",
        hover_data=["investment_buyers"]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Selected filters ke liye geographic data available nahi hai.")


# Cluster profiles
st.subheader("Cluster Insights")

profile = (
    filtered.groupby("cluster")
    .agg(
        Buyers=("client_id", "nunique"),
        Mean_Age=("age", "mean"),
        Mean_Satisfaction=("satisfaction_score", "mean"),
        Investment_Buyers=(
            "acquisition_purpose",
            lambda s: (s.astype(str).str.lower() == "investment").sum()
        ),
        Loan_Applicants=(
            "loan_applied",
            lambda s: s.astype(str).str.lower().isin(["yes", "true", "1"]).sum()
        ),
        Properties_Sold=("sold_properties", "sum"),
        Avg_Purchase_Value=("avg_purchase_price", "mean"),
    )
    .reset_index()
)

profile["Cluster"] = profile["cluster"].map(lambda x: f"C{int(x) + 1}")

st.dataframe(
    profile.drop(columns="cluster").round(2),
    use_container_width=True,
    hide_index=True,
)

k_score = metrics.get("kmeans_silhouette_sample", 0.0)
h_score = metrics.get("hierarchical_silhouette_sample", 0.0)

st.caption(
    f"Clustering diagnostics — K-Means silhouette: {k_score:.3f}; "
    f"Hierarchical silhouette: {h_score:.3f}. "
    "These scores are diagnostics and do not prove commercial optimality."
)


# Client records
st.subheader("Filtered Client Records")

display_cols = [
    "client_id", "client_type", "country", "region",
    "acquisition_purpose", "loan_applied", "satisfaction_score",
    "sold_properties", "total_purchase_value", "cluster"
]

records = filtered[display_cols].copy()
records["cluster"] = records["cluster"].map(lambda x: f"C{int(x) + 1}")

st.dataframe(records, use_container_width=True, hide_index=True)

st.download_button(
    "Download Filtered Buyer Data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_buyer_segments.csv",
    mime="text/csv",
)
