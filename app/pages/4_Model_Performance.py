
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

METRICS_PATH = METRICS_DIR / "model_metrics.json"
CLASSIFICATION_IMPORTANCE_PATH = (
    METRICS_DIR / "classification_feature_importance.csv"
)
REGRESSION_IMPORTANCE_PATH = (
    METRICS_DIR / "regression_feature_importance.csv"
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}"
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    return pd.read_csv(path)


def find_column(dataframe, possible_names):
    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("📈 Machine Learning Model Performance")

st.markdown(
    """
    Evaluate the classification and regression experiments,
    compare model performance, and inspect feature importance.
    """
)

try:
    metrics = load_metrics()

except Exception as error:
    st.error(f"Unable to load model metrics: {error}")
    st.stop()

classification = metrics.get("classification", {})
regression = metrics.get("regression", {})
baseline = metrics.get("regression_baseline", {})


# --------------------------------------------------
# CLASSIFICATION METRICS
# --------------------------------------------------

st.header("1. Investment Classification Model")
st.caption("Model: XGBoost Classifier")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{classification.get('accuracy', 0) * 100:.2f}%",
)

col2.metric(
    "Precision",
    f"{classification.get('precision', 0) * 100:.2f}%",
)

col3.metric(
    "Recall",
    f"{classification.get('recall', 0) * 100:.2f}%",
)

col4.metric(
    "F1 Score",
    f"{classification.get('f1_score', 0) * 100:.2f}%",
)

st.markdown("### Classification Metric Interpretation")

st.markdown(
    """
    - **Accuracy:** Proportion of all predictions that were correct.
    - **Precision:** Proportion of positive predictions that were correct.
    - **Recall:** Proportion of actual positive cases identified.
    - **F1 Score:** Harmonic mean of precision and recall.
    """
)

st.divider()


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("Classification Confusion Matrix")

confusion = classification.get("confusion_matrix")

if confusion is not None:

    confusion_array = pd.DataFrame(confusion).to_numpy()

    if confusion_array.shape == (2, 2):

        confusion_df = pd.DataFrame(
            confusion_array,
            index=[
                "Actual: At or Above City Median",
                "Actual: Below City Median",
            ],
            columns=[
                "Predicted: At or Above City Median",
                "Predicted: Below City Median",
            ],
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=confusion_array,
                x=[
                    "Predicted: 0",
                    "Predicted: 1",
                ],
                y=[
                    "Actual: 0",
                    "Actual: 1",
                ],
                colorscale="Blues",
                showscale=True,
                hovertemplate=(
                    "Actual: %{y}<br>"
                    "Predicted: %{x}<br>"
                    "Count: %{z}<extra></extra>"
                ),
            )
        )

        # Add values directly to the heatmap cells.
        for row in range(2):
            for column in range(2):
                fig.add_annotation(
                    x=column,
                    y=row,
                    text=f"{int(confusion_array[row, column]):,}",
                    showarrow=False,
                    font=dict(
                        size=18,
                        color="black",
                    ),
                )

        fig.update_layout(
            title="Classification Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            height=450,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.dataframe(
            confusion_df,
            width="stretch",
        )

    else:
        st.warning(
            "The confusion matrix must contain exactly four values."
        )

else:
    st.warning(
        "Confusion matrix is not available in the metrics file."
    )

st.divider()


# --------------------------------------------------
# REGRESSION PERFORMANCE
# --------------------------------------------------

st.header("2. Property Price Regression")
st.caption("Model: XGBoost Regressor")

reg_col1, reg_col2, reg_col3 = st.columns(3)

reg_col1.metric(
    "XGBoost MAE",
    f"₹{regression.get('mae', 0):,.2f} Lakhs",
)

reg_col2.metric(
    "XGBoost RMSE",
    f"₹{regression.get('rmse', 0):,.2f} Lakhs",
)

reg_col3.metric(
    "XGBoost R²",
    f"{regression.get('r2', 0):.4f}",
)

st.markdown("### Baseline Comparison")

comparison_data = pd.DataFrame(
    [
        {
            "Model": "Mean Baseline",
            "MAE": baseline.get("mae", 0),
            "RMSE": baseline.get("rmse", 0),
            "R²": baseline.get("r2", 0),
        },
        {
            "Model": "XGBoost Regressor",
            "MAE": regression.get("mae", 0),
            "RMSE": regression.get("rmse", 0),
            "R²": regression.get("r2", 0),
        },
    ]
)

st.dataframe(
    comparison_data,
    width="stretch",
    hide_index=True,
)

regression_chart = px.bar(
    comparison_data,
    x="Model",
    y=["MAE", "RMSE"],
    barmode="group",
    title="Regression Error Comparison",
    labels={
        "value": "Error (Lakhs)",
        "variable": "Metric",
    },
)

st.plotly_chart(
    regression_chart,
    width="stretch",
)

if regression.get("r2", 0) < 0:
    st.error(
        """
        The regression model has a negative R² score and performs
        slightly worse than the mean baseline. It does not provide
        reliable property-price predictions.
        """
    )

st.info(
    """
    **Regression interpretation**

    The dataset does not show meaningful relationships between
    property characteristics and property prices. The regression
    model should therefore not be used as a reliable valuation tool.
    """
)

st.divider()


# --------------------------------------------------
# CLASSIFICATION FEATURE IMPORTANCE
# --------------------------------------------------

st.header("3. Feature Importance")

classification_importance = load_csv(
    str(CLASSIFICATION_IMPORTANCE_PATH)
)

if classification_importance is not None:

    feature_col = find_column(
        classification_importance,
        [
            "feature",
            "Feature",
            "feature_name",
            "Feature_Name",
        ],
    )

    importance_col = find_column(
        classification_importance,
        [
            "importance",
            "Importance",
            "importance_score",
            "Importance_Score",
        ],
    )

    if feature_col and importance_col:

        classification_importance[importance_col] = pd.to_numeric(
            classification_importance[importance_col],
            errors="coerce",
        )

        top_features = (
            classification_importance
            .dropna(subset=[importance_col])
            .sort_values(importance_col, ascending=False)
            .head(15)
        )

        feature_chart = px.bar(
            top_features.sort_values(importance_col),
            x=importance_col,
            y=feature_col,
            orientation="h",
            title="Top 15 Classification Features",
            labels={
                feature_col: "Feature",
                importance_col: "Importance",
            },
        )

        st.plotly_chart(
            feature_chart,
            width="stretch",
        )

        with st.expander(
            "View complete classification feature importance"
        ):
            st.dataframe(
                classification_importance,
                width="stretch",
                hide_index=True,
            )

    else:
        st.warning(
            "Could not identify feature and importance columns "
            "in the classification feature-importance file."
        )

else:
    st.warning(
        "Classification feature importance file not found."
    )


# --------------------------------------------------
# REGRESSION FEATURE IMPORTANCE
# --------------------------------------------------

regression_importance = load_csv(
    str(REGRESSION_IMPORTANCE_PATH)
)

if regression_importance is not None:

    feature_col = find_column(
        regression_importance,
        [
            "feature",
            "Feature",
            "feature_name",
            "Feature_Name",
        ],
    )

    importance_col = find_column(
        regression_importance,
        [
            "importance",
            "Importance",
            "importance_score",
            "Importance_Score",
        ],
    )

    if feature_col and importance_col:

        regression_importance[importance_col] = pd.to_numeric(
            regression_importance[importance_col],
            errors="coerce",
        )

        top_features = (
            regression_importance
            .dropna(subset=[importance_col])
            .sort_values(importance_col, ascending=False)
            .head(15)
        )

        st.subheader("Regression Feature Importance")

        regression_feature_chart = px.bar(
            top_features.sort_values(importance_col),
            x=importance_col,
            y=feature_col,
            orientation="h",
            title="Top 15 Regression Features",
            labels={
                feature_col: "Feature",
                importance_col: "Importance",
            },
        )

        st.plotly_chart(
            regression_feature_chart,
            width="stretch",
        )

        with st.expander(
            "View complete regression feature importance"
        ):
            st.dataframe(
                regression_importance,
                width="stretch",
                hide_index=True,
            )

    else:
        st.warning(
            "Could not identify regression feature-importance columns."
        )

else:
    st.warning(
        "Regression feature importance file not found."
    )

st.divider()


# --------------------------------------------------
# MODEL CONCLUSION
# --------------------------------------------------

st.header("4. Overall Model Assessment")

st.info(
    """
    **Classification:** The classifier achieved an F1 score of
    approximately 71.63% for the experimental relative-price proxy.

    **Regression:** The regressor did not outperform the mean
    baseline and is not suitable for reliable property valuation.

    **Dataset:** The price distribution and feature relationships
    show strong evidence of limited predictive signal. Model
    results should therefore be interpreted cautiously.
    """
)

st.caption(
    "Metrics are loaded from the saved model evaluation results."
)
