
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score


BASE = Path(__file__).resolve().parent


def find_csv(filename):
    paths = [
        BASE / filename,
        BASE / "data" / filename,
    ]

    for path in paths:
        if path.exists():
            return path

    matches = list(BASE.glob(Path(filename).stem + "*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"{filename} nahi mili. CSV ko repository ke main folder mein upload karo."
    )


def load_data():
    clients = pd.read_csv(find_csv("clients.csv"))
    properties = pd.read_csv(find_csv("properties.csv"))

    clients.columns = clients.columns.str.strip().str.lower()
    properties.columns = properties.columns.str.strip().str.lower()

    if "client_id" not in clients.columns:
        raise ValueError("clients.csv mein client_id column missing hai.")

    if "client_ref" not in properties.columns:
        raise ValueError("properties.csv mein client_ref column missing hai.")

    clients["client_id"] = clients["client_id"].astype("string").str.strip()
    properties["client_ref"] = (
        properties["client_ref"].astype("string").str.strip()
    )

    # Calculate age from date of birth
    if "date_of_birth" in clients.columns:
        dob = pd.to_datetime(
            clients["date_of_birth"],
            errors="coerce",
            dayfirst=True,
        )
        today = pd.Timestamp.today().normalize()

        age = today.year - dob.dt.year
        birthday_not_reached = (
            (today.month < dob.dt.month)
            | (
                (today.month == dob.dt.month)
                & (today.day < dob.dt.day)
            )
        )
        clients["age"] = (age - birthday_not_reached.astype(float)).where(
            dob.notna()
        )

    # Convert property prices into numbers
    if "sale_price" in properties.columns:
        properties["sale_price"] = pd.to_numeric(
            properties["sale_price"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip(),
            errors="coerce",
        )
    else:
        properties["sale_price"] = np.nan

    # Identify sold property records
    if "listing_status" in properties.columns:
        sold_mask = (
            properties["listing_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("sold")
        )
    else:
        sold_mask = pd.Series(False, index=properties.index)

    sold = properties.loc[
        sold_mask & properties["client_ref"].notna()
    ].copy()

    # Summarise transactions for each client
    if len(sold) > 0:
        summary = (
            sold.groupby("client_ref")
            .agg(
                sold_properties=("client_ref", "size"),
                avg_purchase_price=("sale_price", "mean"),
                total_purchase_value=("sale_price", "sum"),
            )
            .reset_index()
            .rename(columns={"client_ref": "client_id"})
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "client_id",
                "sold_properties",
                "avg_purchase_price",
                "total_purchase_value",
            ]
        )

    # Match properties to clients using client_ref = client_id
    data = clients.merge(summary, on="client_id", how="left")

    data["sold_properties"] = (
        data["sold_properties"].fillna(0).astype(int)
    )
    data["avg_purchase_price"] = (
        data["avg_purchase_price"].fillna(0)
    )
    data["total_purchase_value"] = (
        data["total_purchase_value"].fillna(0)
    )

    # Ensure columns needed by the dashboard exist
    defaults = {
        "client_type": "Unknown",
        "country": "Unknown",
        "region": "Unknown",
        "acquisition_purpose": "Unknown",
        "loan_applied": "Unknown",
        "satisfaction_score": np.nan,
        "age": np.nan,
    }

    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    data["satisfaction_score"] = pd.to_numeric(
        data["satisfaction_score"], errors="coerce"
    )
    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    return clients, properties, data


def cluster_data(data):
    df = data.copy()

    numeric_features = [
        "age",
        "satisfaction_score",
        "sold_properties",
        "avg_purchase_price",
        "total_purchase_value",
    ]

    categorical_features = [
        "client_type",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
    ]

    # Ensure modelling columns are present
    for column in numeric_features:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in categorical_features:
        if column not in df.columns:
            df[column] = "Unknown"
        df[column] = df[column].fillna("Unknown").astype(str)

    # Avoid all-missing numeric columns
    for column in numeric_features:
        if df[column].isna().all():
            df[column] = 0

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    n = len(df)

    if n == 0:
        df["cluster"] = pd.Series(dtype=int)
        return df, {
            "kmeans_silhouette_sample": 0.0,
            "hierarchical_silhouette_sample": 0.0,
            "n_clusters": 0,
            "samples": 0,
        }

    if n < 3:
        df["cluster"] = 0
        return df, {
            "kmeans_silhouette_sample": 0.0,
            "hierarchical_silhouette_sample": 0.0,
            "n_clusters": 1,
            "samples": n,
        }

    X = preprocessor.fit_transform(df)

    if hasattr(X, "toarray"):
        X = X.toarray()

    # Use fewer clusters for small datasets
    n_clusters = min(4, n - 1)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    df["cluster"] = model.fit_predict(X)

    # Calculate silhouette scores on at most 500 records
    rng = np.random.RandomState(42)
    sample_size = min(500, n)
    indices = rng.choice(n, size=sample_size, replace=False)
    X_sample = X[indices]

    def calculate_silhouette(features, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
            return 0.0
        try:
            return float(silhouette_score(features, labels))
        except Exception:
            return 0.0

    km_labels = df["cluster"].to_numpy()[indices]
    kmeans_score = calculate_silhouette(X_sample, km_labels)

    try:
        hierarchical_labels = AgglomerativeClustering(
            n_clusters=n_clusters
        ).fit_predict(X_sample)

        hierarchical_score = calculate_silhouette(
            X_sample, hierarchical_labels
        )
    except Exception:
        hierarchical_score = 0.0

    metrics = {
        "kmeans_silhouette_sample": kmeans_score,
        "hierarchical_silhouette_sample": hierarchical_score,
        "n_clusters": n_clusters,
        "samples": n,
    }

    return df, metrics
