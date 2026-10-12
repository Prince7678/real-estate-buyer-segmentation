
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


defaults = {
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

for col, default in defaults.items():
    if col not in data.columns:
        data[col] = default

for col in [
    "satisfaction_score", "age", "sold_properties",
    "avg_purchase_price", "total_purchase_value"
]:
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

for col in [
    "country", "region", "acquisition_purpose",
    "client_type", "loan_applied"
]:
    data[col] = data[col].fillna("Unknown").astype(str)

st.sidebar.header("Filter buyers")


def choices(col):
    return sorted(data[col].dropna().astype(str).unique().tolist())


country = st.sidebar.multiselect("Country", choices("country"))
region = st.sidebar.multiselect("Region", choices("region"))
purpose = st.sidebar.multiselect(
    "Acquisition purpose", choices("acquisition_purpose")
)
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


st.subheader("Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Clients in View", f"{len(filtered):,}")
c2.metric(
    "Investment Buyers",
    int((filtered["acquisition_purpose"].str.lower() == "investment").sum()),
)
c3.metric(
    "Loan Applicants",
    int(filtered["loan_applied"].str.lower().isin(["yes", "true", "1"]).sum()),
)
c4.metric("Sold Properties", f"{int(filtered['sold_properties'].sum()):,}")


st.subheader("Buyer Segmentation Overview")
left, right = st.columns(2)

with left:
    counts = (
        filtered["cluster"].value_counts().sort_index()
        .rename_axis("Cluster").reset_index(name="Buyers")
    )
    counts["Cluster"] = counts["Cluster"].map(lambda x: f"C{int(x)+1}")
    st.plotly_chart(
        px.bar(counts, x="Cluster", y="Buyers", text="Buyers",
               title="Buyer Count by Cluster"),
        use_container_width=True,
    )

with right:
    purpose_counts = (
        filtered.groupby(["cluster", "acquisition_purpose"])
        .size().reset_index(name="Buyers")
    )
    purpose_counts["Cluster"] = purpose_counts["cluster"].map(
        lambda x: f"C{int(x)+1}"
    )
    st.plotly_chart(
        px.bar(
            purpose_counts, x="Cluster", y="Buyers",
            color="acquisition_purpose", barmode="group",
            title="Acquisition Purpose by Cluster",
        ),
        use_container_width=True,
    )


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
        lambda x: f"C{int(x)+1}"
    )
    st.plotly_chart(
        px.bar(
            price_profile, x="Cluster", y="avg_purchase_value",
            title="Average Purchase Value by Cluster",
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        px.histogram(
            filtered, x="satisfaction_score",
            color="acquisition_purpose", nbins=5,
            title="Satisfaction Score Distribution",
        ),
        use_container_width=True,
    )


st.subheader("Geographic Buyer Analysis")
geo = (
    filtered.groupby(["country", "region"], as_index=False)
    .agg(
        buyers=("client_id", "nunique"),
        investment_buyers=(
            "acquisition_purpose",
            lambda s: (s.astype(str).str.lower() == "investment").sum(),
        ),
    )
    .sort_values("buyers", ascending=False)
    .head(20)
)

if not geo.empty:
    st.plotly_chart(
        px.bar(
            geo, x="region", y="buyers", color="country",
            title="Top Regions by Buyer Count",
            hover_data=["investment_buyers"],
        ),
        use_container_width=True,
    )
else:
    st.info("Selected filters ke liye geographic data nahi hai.")


st.subheader("Cluster Insights")
profile = (
    filtered.groupby("cluster")
    .agg(
        Buyers=("client_id", "nunique"),
        Mean_Age=("age", "mean"),
        Mean_Satisfaction=("satisfaction_score", "mean"),
        Investment_Buyers=(
            "acquisition_purpose",
            lambda s: (s.astype(str).str.lower() == "investment").sum(),
        ),
        Loan_Applicants=(
            "loan_applied",
            lambda s: s.astype(str).str.lower().isin(["yes", "true", "1"]).sum(),
        ),
        Properties_Sold=("sold_properties", "sum"),
        Avg_Purchase_Value=("avg_purchase_price", "mean"),
    )
    .reset_index()
)

profile["Cluster"] = profile["cluster"].map(lambda x: f"C{int(x)+1}")
st.dataframe(
    profile.drop(columns="cluster").round(2),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"K-Means silhouette: {metrics.get('kmeans_silhouette_sample', 0):.3f} | "
    f"Hierarchical silhouette: "
    f"{metrics.get('hierarchical_silhouette_sample', 0):.3f}"
)


st.subheader("Filtered Client Records")
display_cols = [
    "client_id", "client_type", "country", "region",
    "acquisition_purpose", "loan_applied", "satisfaction_score",
    "sold_properties", "total_purchase_value", "cluster",
]
records = filtered[display_cols].copy()
records["cluster"] = records["cluster"].map(lambda x: f"C{int(x)+1}")

st.dataframe(records, use_container_width=True, hide_index=True)

st.download_button(
    "Download Filtered Buyer Data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_buyer_segments.csv",
    mime="text/csv",
)
