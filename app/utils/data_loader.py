
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_housing_data.csv"
)

MODEL_READY_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_ready_data.csv"
)

METRICS_PATH = (
    PROJECT_ROOT / "reports" / "metrics" / "model_metrics.json"
)


@st.cache_data(show_spinner="Loading property dataset...")
def load_cleaned_data():
    """Load the cleaned housing dataset."""

    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {CLEANED_DATA_PATH}"
        )

    df = pd.read_csv(CLEANED_DATA_PATH)

    return df


@st.cache_data(show_spinner="Loading model-ready dataset...")
def load_model_ready_data():
    """Load the dataset used during model training."""

    if not MODEL_READY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found: {MODEL_READY_DATA_PATH}"
        )

    return pd.read_csv(MODEL_READY_DATA_PATH)


@st.cache_data
def load_model_metrics():
    """Load saved classification and regression metrics."""

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    return metrics


def get_dataset_summary(df):
    """Return basic dataset statistics."""

    summary = {
        "total_properties": int(len(df)),
        "total_cities": int(df["City"].nunique()),
        "total_states": int(df["State"].nunique()),
        "average_price": float(df["Price_in_Lakhs"].mean()),
        "median_price": float(df["Price_in_Lakhs"].median()),
        "average_size": float(df["Size_in_SqFt"].mean()),
        "property_types": int(df["Property_Type"].nunique()),
    }

    return summary
