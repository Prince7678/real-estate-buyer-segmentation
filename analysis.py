
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


# Project folder: CSV files are expected beside analysis.py
BASE = Path(__file__).resolve().parent
DATA = BASE


def find_csv(filename):
    """Find a dataset in the project root or data folder."""
    candidates = [
        BASE / filename,
        BASE / "data" / filename,
    ]

    for path in candidates:
        if path.is_file():
            return path

    # Handle accidentally duplicated download names, e.g. properties (1).csv
    stem = Path(filename).stem
    matches = sorted(BASE.glob(f"{stem}*.csv"))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"{filename} was not found. "
        f"Place it in the repository root or data/ folder."
    )


def clean_numeric(series):
    """Convert currency and comma-formatted values to numbers."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$₹,\s]", "", regex=True)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def load_data():
    """Load and prepare client and property transaction datasets."""
    clients = pd.read_csv(find_csv("clients.csv"))
    properties = pd.read_csv(find_csv("properties.csv"))

    # Normalize column names.
    clients.columns = clients.columns.str.strip().str.lower()
    properties.columns = properties.columns.str.strip().str.lower()

    # Prepare client information.
    if "date_of_birth" in clients.columns:
        clients["date_of_birth"] = pd.to_datetime(
            clients["date_of_birth"],
            errors="coerce",
            dayfirst=True,
        )
        reference_date = pd.Timestamp.today().normalize()
        clients["age"] = (
            (reference_date - clients["date_of_birth"]).dt.days / 365.25
        ).round()

        clients.loc[
            ~clients["age"].between(18, 100), "age"
        ] = np.nan

    # Prepare property information.
    if "transaction_date" in properties.columns:
        properties["transaction_date"] = pd.to_datetime(
            properties["transaction_date"],
            errors="coerce",
            dayfirst=True,
        )

    if "sale_price" in properties.columns:
        properties["sale_price"] = clean_numeric(
            properties["sale_price"]
        )

    if "floor_area_sqft" in properties.columns:
        properties["floor_area_sqft"] = clean_numeric(
            properties["floor_area_sqft"]
        )

    # Normalize transaction status.
    if "listing_status" in properties.columns:
        properties["listing_status"] = (
            properties["listing_status"].astype(str).str.strip().str.title()
        )

    # Join property transactions to the corresponding clients.
    if "client_ref" in properties.columns and "client_ref" in clients.columns:
        data = properties.merge(
            clients,
            on="client_ref",
            how="left",
            suffixes=("_property", "_client"),
        )
    else:
        # Keep the property data usable even when no common key exists.
        data = properties.copy()

    if data.empty:
        raise ValueError("The property dataset contains no records.")

    return clients, properties, data


def cluster_data(data):
    """Segment real-estate transactions using K-Means clustering."""
    df = data.copy()

    numeric_candidates = [
        "age",
        "income",
        "annual_income",
        "monthly_income",
        "budget",
        "investment_budget",
        "sale_price",
        "floor_area_sqft",
        "tower_number",
    ]

    categorical_candidates = [
        "unit_category",
        "listing_status",
        "city",
        "location",
        "property_type",
        "investment_type",
    ]

    numeric_features = [
        col for col in numeric_candidates
        if col in df.columns
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]

    categorical_features = [
        col for col in categorical_candidates
        if col in df.columns and df[col].nunique(dropna=True) > 1
    ]

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No usable numeric or categorical features were found."
        )

    # Use only selected features, not IDs or personal identifiers.
    X = df[numeric_features + categorical_features].copy()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers)
    X_transformed = preprocessor.fit_transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    n_samples = len(df)

    if n_samples < 2:
        df["cluster"] = 0
        metrics = {
            "n_clusters": 1,
            "silhouette_score": None,
            "samples": n_samples,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }
        return df, metrics

    # Limit cluster count to what the available data supports.
    n_clusters = min(4, n_samples - 1)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_transformed)
    df["cluster"] = labels

    score = None
    if 1 < len(np.unique(labels)) < n_samples:
        try:
            score = float(
                silhouette_score(X_transformed, labels)
            )
        except (ValueError, TypeError):
            score = None

    # Optional 2D PCA coordinates for dashboard visualizations.
    try:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_transformed)
        df["pca_x"] = coords[:, 0]
        df["pca_y"] = coords[:, 1]
    except (ValueError, TypeError):
        df["pca_x"] = np.nan
        df["pca_y"] = np.nan

    metrics = {
        "n_clusters": int(len(np.unique(labels))),
        "silhouette_score": score,
        "samples": int(n_samples),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return df, metrics
