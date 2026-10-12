
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
    candidates = [
        BASE / filename,
        BASE / "data" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = list(BASE.glob(Path(filename).stem + "*.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"{filename} not found. Upload it to the repository root."
    )


def load_data():
    clients = pd.read_csv(find_csv("clients.csv"))
    properties = pd.read_csv(find_csv("properties.csv"))

    clients.columns = clients.columns.str.strip().str.lower()
    properties.columns = properties.columns.str.strip().str.lower()

    if "client_id" not in clients.columns:
        raise ValueError("clients.csv must contain client_id")

    if "client_ref" not in properties.columns:
        raise ValueError("properties.csv must contain client_ref")

    clients["client_id"] = clients["client_id"].astype("string").str.strip()
    properties["client_ref"] = (
        properties["client_ref"].astype("string").str.strip()
    )

    # Calculate client age
    if "date_of_birth" in clients.columns:
        dob = pd.to_datetime(
            clients["date_of_birth"], errors="coerce", dayfirst=True
        )
        today = pd.Timestamp.today().normalize()
        clients["age"] = (
            today.year
            - dob.dt.year
            - (
                (today.month < dob.dt.month)
                | (
                    (today.month == dob.dt.month)
                    & (today.day < dob.dt.day)
                )
            ).astype("Int64")
        )
        clients["age"] = pd.to_numeric(clients["age"], errors="coerce")

    # Prepare property transaction fields
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

    # Count and value sold properties for each client
    sold = properties.loc[sold_mask].copy()

    if sold.empty:
        property_summary = pd.DataFrame(
            columns=[
                "client_id",
                "sold_properties",
                "avg_purchase_price",
                "total_purchase_value",
            ]
        )
    else:
        property_summary = (
            sold.groupby("client_ref", dropna=True)
            .agg(
                sold_properties=("client_ref", "size"),
                avg_purchase_price=("sale_price", "mean"),
                total_purchase_value=("sale_price", "sum"),
            )
            .reset_index()
            .rename(columns={"client_ref": "client_id"})
        )

    # Join client details with their sold-property summary
    data = clients.merge(property_summary, on="client_id", how="left")

    data["sold_properties"] = (
        data["sold_properties"].fillna(0).astype(int)
    )
    data["avg_purchase_price"] = data["avg_purchase_price"].fillna(0)
    data["total_purchase_value"] = data["total_purchase_value"].fillna(0)

    # Add safe defaults for dashboard fields
    defaults = {
        "client_type": "Unknown",
        "country": "Unknown",
        "region": "Unknown",
        "acquisition_purpose": "Unknown",
        "loan_applied": "Unknown",
        "satisfaction_score": np.nan,
    }

    for col, default in defaults.items():
        if col not in data.columns:
            data[col] = default

    data["satisfaction_score"] = pd.to_numeric(
        data["satisfaction_score"], errors="coerce"
    )

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

    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_features:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

    n = len(df)

    if n < 2:
        df["cluster"] = 0
        return df, {
            "kmeans_silhouette_sample": 0.0,
            "hierarchical_silhouette_sample": 0.0,
            "n_clusters": 1,
            "samples": n,
        }

    X = preprocessor.fit_transform(df)

    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    n_clusters = min(4, n)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    df["cluster"] = kmeans.fit_predict(X_dense)

    # Silhouette diagnostics on a manageable sample
    rng = np.random.RandomState(42)
    sample_size = min(500, n)
    sample_indices = rng.choice(n, size=sample_size, replace=False)
    X_sample = X_dense[sample_indices]
    km_sample_labels = df["cluster"].to_numpy()[sample_indices]

    def safe_silhouette(features, labels):
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
            return 0.0
        try:
            return float(silhouette_score(features, labels))
        except Exception:
            return 0.0

    kmeans_score = safe_silhouette(X_sample, km_sample_labels)

    try:
        hierarchical_labels = AgglomerativeClustering(
            n_clusters=n_clusters
        ).fit_predict(X_sample)

        hierarchical_score = safe_silhouette(
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
