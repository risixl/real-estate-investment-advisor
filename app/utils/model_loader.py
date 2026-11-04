
from pathlib import Path

import joblib
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASSIFIER_PATH = PROJECT_ROOT / "models" / "classifier.pkl"
REGRESSOR_PATH = PROJECT_ROOT / "models" / "regressor.pkl"
BASELINE_PATH = PROJECT_ROOT / "models" / "baseline_regressor.pkl"


@st.cache_resource(show_spinner="Loading investment classification model...")
def load_classifier():
    """Load the trained XGBoost classification pipeline."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_PATH}"
        )

    return joblib.load(CLASSIFIER_PATH)


@st.cache_resource(show_spinner="Loading experimental regression model...")
def load_regressor():
    """Load the trained regression pipeline."""

    if not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Regressor model not found: {REGRESSOR_PATH}"
        )

    return joblib.load(REGRESSOR_PATH)


@st.cache_resource(show_spinner="Loading regression baseline...")
def load_baseline_regressor():
    """Load the baseline regression model."""

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_PATH}"
        )

    return joblib.load(BASELINE_PATH)
