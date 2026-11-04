
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import (
    load_cleaned_data,
    load_model_metrics,
    get_dataset_summary,
)

from utils.helpers import (
    format_lakhs,
    format_number,
    format_area,
    safe_percentage,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.18);
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

st.title("🏠 Real Estate Investment Advisor")

st.markdown(
    """
    <div class="subtitle">
        Property market exploration, investment indicators
        and scenario-based future valuation.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_cleaned_data()
    metrics = load_model_metrics()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

required_columns = [
    "State",
    "City",
    "Property_Type",
    "Price_in_Lakhs",
    "Size_in_SqFt",
    "BHK",
    "Price_per_SqFt",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The cleaned dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

st.sidebar.caption(
    "Use these filters to explore the housing dataset."
)

states = sorted(df["State"].dropna().unique().tolist())

selected_states = st.sidebar.multiselect(
    "Select State",
    options=states,
    default=[],
)

filtered_df = df.copy()

if selected_states:
    filtered_df = filtered_df[
        filtered_df["State"].isin(selected_states)
    ]

cities = sorted(filtered_df["City"].dropna().unique().tolist())

selected_cities = st.sidebar.multiselect(
    "Select City",
    options=cities,
    default=[],
)

if selected_cities:
    filtered_df = filtered_df[
        filtered_df["City"].isin(selected_cities)
    ]

property_types = sorted(
    filtered_df["Property_Type"].dropna().unique().tolist()
)

selected_property_types = st.sidebar.multiselect(
    "Property Type",
    options=property_types,
    default=[],
)

if selected_property_types:
    filtered_df = filtered_df[
        filtered_df["Property_Type"].isin(selected_property_types)
    ]

st.sidebar.divider()

st.sidebar.caption(
    "All dashboard statistics update according to the selected filters."
)


if filtered_df.empty:
    st.warning(
        "No properties match the selected filters. "
        "Change your selections."
    )
    st.stop()


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

summary = get_dataset_summary(filtered_df)

st.subheader("Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Properties",
    format_number(summary["total_properties"]),
)

col2.metric(
    "Cities Covered",
    format_number(filtered_df["City"].nunique()),
)

col3.metric(
    "Average Property Price",
    format_lakhs(summary["average_price"]),
)

col4.metric(
    "Average Property Size",
    format_area(summary["average_size"]),
)


# --------------------------------------------------
# ADDITIONAL METRICS
# --------------------------------------------------

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "States Covered",
    format_number(filtered_df["State"].nunique()),
)

col6.metric(
    "Property Types",
    format_number(filtered_df["Property_Type"].nunique()),
)

col7.metric(
    "Median Property Price",
    format_lakhs(filtered_df["Price_in_Lakhs"].median()),
)

col8.metric(
    "Average BHK",
    f"{filtered_df['BHK'].mean():.2f}",
)


st.divider()


# --------------------------------------------------
# CITY-WISE PRICE ANALYSIS
# --------------------------------------------------

st.subheader("City-wise Property Price Analysis")

city_analysis = (
    filtered_df.groupby("City", as_index=False)
    .agg(
        Average_Price=("Price_in_Lakhs", "mean"),
        Median_Price=("Price_in_Lakhs", "median"),
        Property_Count=("Price_in_Lakhs", "count"),
    )
    .sort_values("Average_Price", ascending=False)
)

city_col1, city_col2 = st.columns(2)

with city_col1:

    city_chart = px.bar(
        city_analysis.head(15),
        x="City",
        y="Average_Price",
        title="Top 15 Cities by Average Property Price",
        labels={
            "City": "City",
            "Average_Price": "Average Price (Lakhs)",
        },
        color="Average_Price",
        color_continuous_scale="Blues",
    )

    city_chart.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
    )

    st.plotly_chart(city_chart, use_container_width=True)


with city_col2:

    city_count_chart = px.bar(
        city_analysis.sort_values(
            "Property_Count",
            ascending=False,
        ).head(15),
        x="City",
        y="Property_Count",
        title="Top 15 Cities by Property Records",
        labels={
            "City": "City",
            "Property_Count": "Number of Properties",
        },
        color="Property_Count",
        color_continuous_scale="Teal",
    )

    city_count_chart.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
    )

    st.plotly_chart(city_count_chart, use_container_width=True)


# --------------------------------------------------
# PROPERTY TYPE ANALYSIS
# --------------------------------------------------

st.subheader("Property Type Analysis")

type_col1, type_col2 = st.columns(2)

property_analysis = (
    filtered_df.groupby("Property_Type", as_index=False)
    .agg(
        Property_Count=("Property_Type", "count"),
        Average_Price=("Price_in_Lakhs", "mean"),
    )
)

with type_col1:

    type_chart = px.pie(
        property_analysis,
        names="Property_Type",
        values="Property_Count",
        title="Distribution of Property Types",
        hole=0.45,
    )

    st.plotly_chart(type_chart, use_container_width=True)


with type_col2:

    type_price_chart = px.bar(
        property_analysis.sort_values(
            "Average_Price",
            ascending=False,
        ),
        x="Property_Type",
        y="Average_Price",
        title="Average Price by Property Type",
        labels={
            "Property_Type": "Property Type",
            "Average_Price": "Average Price (Lakhs)",
        },
        color="Average_Price",
        color_continuous_scale="Viridis",
    )

    st.plotly_chart(
        type_price_chart,
        use_container_width=True,
    )


# --------------------------------------------------
# PRICE DISTRIBUTION
# --------------------------------------------------

st.subheader("Property Price Distribution")

price_chart = px.histogram(
    filtered_df,
    x="Price_in_Lakhs",
    nbins=40,
    title="Distribution of Property Prices",
    labels={
        "Price_in_Lakhs": "Property Price (Lakhs)",
        "count": "Number of Properties",
    },
    color_discrete_sequence=["#3B82F6"],
)

st.plotly_chart(price_chart, use_container_width=True)


# --------------------------------------------------
# SIZE VS PRICE
# --------------------------------------------------

st.subheader("Property Size Analysis")

size_chart = px.scatter(
    filtered_df.sample(
        n=min(5000, len(filtered_df)),
        random_state=42,
    ),
    x="Size_in_SqFt",
    y="Price_in_Lakhs",
    color="Property_Type",
    opacity=0.55,
    title="Property Size vs Property Price",
    labels={
        "Size_in_SqFt": "Property Size (Sq.Ft.)",
        "Price_in_Lakhs": "Property Price (Lakhs)",
    },
)

st.plotly_chart(size_chart, use_container_width=True)


# --------------------------------------------------
# PROPERTY DATA TABLE
# --------------------------------------------------

st.subheader("Property Dataset Preview")

preview_columns = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
]

available_preview_columns = [
    column
    for column in preview_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_preview_columns].head(100),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# MODEL PERFORMANCE SNAPSHOT
# --------------------------------------------------

st.divider()

st.subheader("Machine Learning Performance Snapshot")

classification = metrics.get("classification", {})

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Classification Accuracy",
    safe_percentage(classification.get("accuracy", 0)),
)

metric_col2.metric(
    "Precision",
    safe_percentage(classification.get("precision", 0)),
)

metric_col3.metric(
    "Recall",
    safe_percentage(classification.get("recall", 0)),
)

metric_col4.metric(
    "F1 Score",
    safe_percentage(classification.get("f1_score", 0)),
)

st.warning(
    """
    **Important model limitation**

    The investment classifier predicts a proxy label based on whether
    a property's price per square foot is below its city's median.
    It does not predict verified investment returns or guarantee profit.

    The regression model did not outperform the mean baseline.
    Its output is not suitable for reliable property valuation.
    """
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Real Estate Investment Advisor | "
    "Built with Python, Pandas, Scikit-learn, XGBoost, "
    "Plotly, Streamlit and MLflow."
)
