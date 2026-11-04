
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_cleaned_data
from utils.helpers import format_lakhs, format_area


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Property Analysis",
    page_icon="🏘️",
    layout="wide",
)

st.title("🏘️ Property Analysis")

st.markdown(
    """
    Explore properties across Indian cities, compare property
    characteristics, and investigate the available dataset.
    """
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:
    df = load_cleaned_data()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Property Search")

states = sorted(df["State"].dropna().unique().tolist())

selected_state = st.sidebar.selectbox(
    "State",
    ["All"] + states,
)

filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]


cities = sorted(filtered_df["City"].dropna().unique().tolist())

selected_city = st.sidebar.selectbox(
    "City",
    ["All"] + cities,
)

if selected_city != "All":
    filtered_df = filtered_df[
        filtered_df["City"] == selected_city
    ]


localities = sorted(
    filtered_df["Locality"].dropna().unique().tolist()
)

selected_locality = st.sidebar.selectbox(
    "Locality",
    ["All"] + localities,
)

if selected_locality != "All":
    filtered_df = filtered_df[
        filtered_df["Locality"] == selected_locality
    ]


property_types = sorted(
    filtered_df["Property_Type"].dropna().unique().tolist()
)

selected_type = st.sidebar.selectbox(
    "Property Type",
    ["All"] + property_types,
)

if selected_type != "All":
    filtered_df = filtered_df[
        filtered_df["Property_Type"] == selected_type
    ]


# --------------------------------------------------
# PRICE FILTER
# --------------------------------------------------

st.sidebar.subheader("Price Range")

minimum_price = float(filtered_df["Price_in_Lakhs"].min())
maximum_price = float(filtered_df["Price_in_Lakhs"].max())

price_range = st.sidebar.slider(
    "Property Price (Lakhs)",
    min_value=minimum_price,
    max_value=maximum_price,
    value=(minimum_price, maximum_price),
)

filtered_df = filtered_df[
    filtered_df["Price_in_Lakhs"].between(
        price_range[0],
        price_range[1],
    )
]


# --------------------------------------------------
# BHK FILTER
# --------------------------------------------------

bhk_values = sorted(
    filtered_df["BHK"].dropna().unique().tolist()
)

selected_bhk = st.sidebar.multiselect(
    "BHK Configuration",
    options=bhk_values,
    default=bhk_values,
)

if selected_bhk:
    filtered_df = filtered_df[
        filtered_df["BHK"].isin(selected_bhk)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if filtered_df.empty:
    st.warning("No properties match your search criteria.")
    st.stop()


st.subheader("Property Search Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Matching Properties", f"{len(filtered_df):,}")

col2.metric(
    "Average Price",
    format_lakhs(filtered_df["Price_in_Lakhs"].mean()),
)

col3.metric(
    "Average Size",
    format_area(filtered_df["Size_in_SqFt"].mean()),
)

col4.metric(
    "Average BHK",
    f"{filtered_df['BHK'].mean():.2f}",
)


st.divider()


# --------------------------------------------------
# LOCALITY COMPARISON
# --------------------------------------------------

st.subheader("Locality Comparison")

locality_analysis = (
    filtered_df.groupby("Locality", as_index=False)
    .agg(
        Average_Price=("Price_in_Lakhs", "mean"),
        Property_Count=("Locality", "count"),
        Average_Size=("Size_in_SqFt", "mean"),
    )
    .sort_values("Average_Price", ascending=False)
)

top_localities = locality_analysis.head(15)

fig_locality = px.bar(
    top_localities,
    x="Locality",
    y="Average_Price",
    color="Property_Count",
    title="Top 15 Localities by Average Property Price",
    labels={
        "Average_Price": "Average Price (Lakhs)",
        "Property_Count": "Property Records",
    },
    color_continuous_scale="Viridis",
)

fig_locality.update_layout(xaxis_tickangle=-45)

st.plotly_chart(
    fig_locality,
    use_container_width=True,
)


# --------------------------------------------------
# BHK ANALYSIS
# --------------------------------------------------

st.subheader("BHK Configuration Analysis")

bhk_analysis = (
    filtered_df.groupby("BHK", as_index=False)
    .agg(
        Average_Price=("Price_in_Lakhs", "mean"),
        Median_Price=("Price_in_Lakhs", "median"),
        Property_Count=("BHK", "count"),
    )
)

bhk_col1, bhk_col2 = st.columns(2)

with bhk_col1:

    fig_bhk_price = px.bar(
        bhk_analysis,
        x="BHK",
        y="Average_Price",
        title="Average Price by BHK",
        labels={
            "BHK": "Bedrooms",
            "Average_Price": "Average Price (Lakhs)",
        },
    )

    st.plotly_chart(
        fig_bhk_price,
        use_container_width=True,
    )


with bhk_col2:

    fig_bhk_count = px.pie(
        bhk_analysis,
        names="BHK",
        values="Property_Count",
        title="Distribution by BHK",
        hole=0.4,
    )

    st.plotly_chart(
        fig_bhk_count,
        use_container_width=True,
    )


# --------------------------------------------------
# AMENITY ANALYSIS
# --------------------------------------------------

st.subheader("Property Amenities")

amenity_columns = [
    "Amenity_Gym",
    "Amenity_Pool",
    "Amenity_Garden",
    "Amenity_Clubhouse",
    "Amenity_Playground",
]

available_amenities = [
    column
    for column in amenity_columns
    if column in filtered_df.columns
]

if available_amenities:

    amenity_data = pd.DataFrame({
        "Amenity": [
            column.replace("Amenity_", "")
            for column in available_amenities
        ],
        "Property_Count": [
            int(filtered_df[column].fillna(0).sum())
            for column in available_amenities
        ],
    })

    fig_amenities = px.bar(
        amenity_data,
        x="Amenity",
        y="Property_Count",
        title="Amenity Availability Across Properties",
        color="Property_Count",
    )

    st.plotly_chart(
        fig_amenities,
        use_container_width=True,
    )


# --------------------------------------------------
# PROPERTY TABLE
# --------------------------------------------------

st.subheader("Detailed Property Records")

display_columns = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Furnished_Status",
    "Parking_Space",
    "Security",
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns].head(500),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Showing up to 500 of {len(filtered_df):,} matching records."
)


# --------------------------------------------------
# DATA QUALITY NOTICE
# --------------------------------------------------

st.warning(
    """
    Dataset limitation: Price distributions and relationships
    between property prices and characteristics appear unusually
    uniform. These records should not be interpreted as verified
    real-world market prices.
    """
)
