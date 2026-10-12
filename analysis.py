from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

BASE = Path(__file__).parent
DATA = BASE / 'data'

def load_data():
    clients = pd.read_csv(DATA / 'clients.csv')
    props = pd.read_csv(DATA / 'properties.csv')
    # Normalize dates with mixed formats and currency values.
    clients['date_of_birth'] = pd.to_datetime(clients['date_of_birth'], errors='coerce', dayfirst=True)
    clients['age'] = ((pd.Timestamp('2026-10-12') - clients['date_of_birth']).dt.days / 365.25).round()
    props['transaction_date'] = pd.to_datetime(props['transaction_date'], errors='coerce', dayfirst=True)
    props['sale_price'] = pd.to_numeric(props['sale_price'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    props['floor_area_sqft'] = pd.to_numeric(props['floor_area_sqft'], errors='coerce')
    sold = props[props['listing_status'].astype(str).str.lower().eq('sold')].copy()
    agg = sold.groupby('client_ref', dropna=True).agg(
        sold_properties=('listing_id', 'count'),
        total_purchase_value=('sale_price', 'sum'),
        avg_purchase_price=('sale_price', 'mean'),
        avg_area_sqft=('floor_area_sqft', 'mean'),
        property_types=('unit_category', 'nunique'),
    ).reset_index().rename(columns={'client_ref':'client_id'})
    data = clients.merge(agg, on='client_id', how='left')
    data['sold_properties'] = data['sold_properties'].fillna(0)
    data['total_purchase_value'] = data['total_purchase_value'].fillna(0)
    data['avg_purchase_price'] = data['avg_purchase_price'].fillna(0)
    data['avg_area_sqft'] = data['avg_area_sqft'].fillna(0)
    data['property_types'] = data['property_types'].fillna(0)
    return clients, props, data

def build_features(data):
    numeric = ['age','satisfaction_score','sold_properties','total_purchase_value','avg_purchase_price','avg_area_sqft','property_types']
    categorical = ['client_type','gender','country','region','acquisition_purpose','loan_applied','referral_channel']
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num',num_pipe,numeric),('cat',cat_pipe,categorical)])
    X = prep.fit_transform(data[numeric+categorical])
    return X, prep, numeric, categorical

def cluster_data(data, k=4):
    X, prep, numeric, categorical = build_features(data)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    data = data.copy()
    data['cluster'] = labels
    # Hierarchical validation on a manageable reproducible sample.
    n = min(500, X.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    Xs = X[idx].toarray() if hasattr(X, 'toarray') else X[idx]
    hlabels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xs)
    hsil = float(silhouette_score(Xs, hlabels)) if len(set(hlabels)) > 1 else float('nan')
    # Silhouette on a reproducible subset to keep runtime reasonable.
    sample_n = min(700, X.shape[0])
    sil = float(silhouette_score(X, labels, sample_size=sample_n, random_state=42)) if len(set(labels)) > 1 else float('nan')
    return data, {'kmeans_silhouette_sample':sil, 'hierarchical_silhouette_sample':hsil, 'k':k, 'records':len(data)}

if __name__ == '__main__':
    _, props, data = load_data()
    clustered, metrics = cluster_data(data)
    print('Clients:', len(data), '| Properties:', len(props))
    print('Metrics:', metrics)
    print('\nCluster sizes:\n', clustered['cluster'].value_counts().sort_index())
    print('\nCluster profile:\n', clustered.groupby('cluster')[['age','satisfaction_score','sold_properties','total_purchase_value','avg_purchase_price']].mean().round(2))
    clustered.to_csv(BASE/'clustered_clients.csv', index=False)
