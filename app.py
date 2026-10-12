import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from analysis import load_data, cluster_data

st.set_page_config(page_title='Real Estate Buyer Intelligence', page_icon='🏙️', layout='wide')
st.title('🏙️ Real Estate Buyer Segmentation & Investment Profiling')
st.caption('Machine-learning buyer intelligence dashboard | Unified Mentor project')

@st.cache_data
def get_results():
    clients, properties, data = load_data()
    clustered, metrics = cluster_data(data)
    return clients, properties, clustered, metrics

try:
    clients, properties, data, metrics = get_results()
except Exception as e:
    st.error(f'Could not load project data: {e}')
    st.stop()

st.sidebar.header('Filter buyers')
def choices(col): return sorted(data[col].dropna().astype(str).unique().tolist())
country = st.sidebar.multiselect('Country', choices('country'), default=[])
region = st.sidebar.multiselect('Region', choices('region'), default=[])
purpose = st.sidebar.multiselect('Acquisition purpose', choices('acquisition_purpose'), default=[])
ctype = st.sidebar.multiselect('Client type', choices('client_type'), default=[])
filtered = data.copy()
for col, vals in [('country',country),('region',region),('acquisition_purpose',purpose),('client_type',ctype)]:
    if vals: filtered = filtered[filtered[col].astype(str).isin(vals)]

c1,c2,c3,c4 = st.columns(4)
c1.metric('Clients in view', f'{len(filtered):,}')
c2.metric('Investment-purpose buyers', f'{(filtered.acquisition_purpose.astype(str).str.lower() == "investment").sum():,}')
c3.metric('Loan applications', f'{(filtered.loan_applied.astype(str).str.lower() == "yes").sum():,}')
c4.metric('Sold property records', f'{int(filtered.sold_properties.sum()):,}')

st.subheader('Buyer segmentation overview')
left,right = st.columns(2)
with left:
    counts = filtered['cluster'].value_counts().sort_index().rename_axis('Cluster').reset_index(name='Buyers')
    counts['Cluster'] = counts['Cluster'].map(lambda x: f'C{int(x)+1}')
    st.plotly_chart(px.bar(counts, x='Cluster', y='Buyers', title='Buyer count by cluster', text='Buyers'), use_container_width=True)
with right:
    purpose_counts = filtered.groupby(['cluster','acquisition_purpose']).size().reset_index(name='Buyers')
    purpose_counts['Cluster'] = purpose_counts['cluster'].map(lambda x: f'C{int(x)+1}')
    st.plotly_chart(px.bar(purpose_counts, x='Cluster', y='Buyers', color='acquisition_purpose', barmode='group', title='Acquisition purpose by cluster'), use_container_width=True)

st.subheader('Investor behaviour')
left,right = st.columns(2)
with left:
    price_profile = filtered.groupby('cluster', as_index=False).agg(avg_purchase_value=('avg_purchase_price','mean'), total_purchase_value=('total_purchase_value','sum'))
    price_profile['Cluster'] = price_profile['cluster'].map(lambda x: f'C{int(x)+1}')
    st.plotly_chart(px.bar(price_profile, x='Cluster', y='avg_purchase_value', title='Average purchase value by cluster', labels={'avg_purchase_value':'Average purchase value'}), use_container_width=True)
with right:
    st.plotly_chart(px.histogram(filtered, x='satisfaction_score', color='acquisition_purpose', nbins=5, title='Satisfaction score distribution'), use_container_width=True)

st.subheader('Geographic buyer analysis')
geo = filtered.groupby(['country','region'], as_index=False).agg(buyers=('client_id','nunique'), investment_buyers=('acquisition_purpose', lambda s: (s.astype(str).str.lower()=='investment').sum()))
geo = geo.sort_values('buyers', ascending=False).head(20)
st.plotly_chart(px.bar(geo, x='region', y='buyers', color='country', title='Top regions by buyer count', hover_data=['investment_buyers']), use_container_width=True)

st.subheader('Cluster insights')
profile = filtered.groupby('cluster').agg(Buyers=('client_id','nunique'), Mean_age=('age','mean'), Mean_satisfaction=('satisfaction_score','mean'), Investment_buyers=('acquisition_purpose',lambda s:(s.astype(str).str.lower()=='investment').sum()), Loan_applicants=('loan_applied',lambda s:(s.astype(str).str.lower()=='yes').sum()), Properties_sold=('sold_properties','sum'), Avg_purchase_value=('avg_purchase_price','mean')).reset_index()
profile['Cluster'] = profile['cluster'].map(lambda x: f'C{int(x)+1}')
st.dataframe(profile.drop(columns='cluster').round(2), use_container_width=True, hide_index=True)
st.caption(f"Clustering validation (sampled): K-Means silhouette = {metrics['kmeans_silhouette_sample']:.3f}; Hierarchical silhouette = {metrics['hierarchical_silhouette_sample']:.3f}. These scores are diagnostics, not proof that clusters are commercially optimal.")
st.subheader('Filtered client records')
st.dataframe(filtered[['client_id','client_type','country','region','acquisition_purpose','loan_applied','satisfaction_score','sold_properties','total_purchase_value','cluster']].assign(cluster=lambda d:d.cluster.map(lambda x:f'C{int(x)+1}')), use_container_width=True, hide_index=True)
st.download_button('Download filtered buyer data (CSV)', filtered.to_csv(index=False).encode('utf-8'), 'filtered_buyer_segments.csv', 'text/csv')
