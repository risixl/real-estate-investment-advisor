import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_cleaned_data
from utils.model_loader import load_classifier


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Investment Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Property Investment Analysis")

st.markdown(
    """
    Evaluate a property's characteristics using the trained
    XGBoost classification model.

    The model identifies properties that fall into a relative
    price-opportunity category based on the training methodology.
    """
)


# ============================================================
# 2. LOAD DATA AND MODEL
# ============================================================

try:
    df = load_cleaned_data()
    classifier = load_classifier()

except Exception as error:
    st.error(f"Unable to load required resources: {error}")
    st.stop()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_options(dataframe, column):
    """Return sorted, non-empty categorical options."""

    if column not in dataframe.columns:
        return []

    return sorted(
        dataframe[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def get_cities_for_state(state, city_state_map):
    """Return cities associated with the selected state."""

    return sorted([
        city
        for city, mapped_state in city_state_map.items()
        if mapped_state == state
    ])


def parse_amenities(value):
    """
    Parse amenities using exact comma-separated tokens.

    This matches the preprocessing pipeline.
    """

    if pd.isna(value):
        return set()

    return {
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    }


def get_amenity_features(value):
    """Create the five amenity indicators and their total count."""

    amenities = parse_amenities(value)

    feature_names = [
        "Gym",
        "Pool",
        "Garden",
        "Clubhouse",
        "Playground",
    ]

    features = {
        f"Amenity_{name}": int(name.lower() in amenities)
        for name in feature_names
    }

    features["Amenity_Count"] = sum(features.values())

    return features


def safe_numeric_median(column, fallback=0):
    """Return a safe numerical median from the dataset."""

    if column not in df.columns:
        return fallback

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    value = values.median()

    if pd.isna(value):
        return fallback

    return float(value)


# ============================================================
# 4. VALIDATE REQUIRED DATASET COLUMNS
# ============================================================

required_columns = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Dataset is missing required columns: {missing_columns}"
    )
    st.stop()


# ============================================================
# 5. DATA-DRIVEN CITY-STATE MAPPING
# ============================================================

location_data = df[
    ["City", "State"]
].dropna().copy()

location_data["City"] = (
    location_data["City"].astype(str).str.strip()
)

location_data["State"] = (
    location_data["State"].astype(str).str.strip()
)

# Use the most frequent state for each city.
# This avoids maintaining a separate hardcoded mapping.

city_state_map = (
    location_data.groupby("City")["State"]
    .agg(lambda values: values.mode().iloc[0])
    .to_dict()
)

valid_states = sorted(
    location_data["State"].unique().tolist()
)

if not city_state_map or not valid_states:
    st.error("No usable city-state combinations were found.")
    st.stop()


# ============================================================
# 6. SESSION STATE
# ============================================================

if "selected_state" not in st.session_state:
    st.session_state["selected_state"] = valid_states[0]

if st.session_state["selected_state"] not in valid_states:
    st.session_state["selected_state"] = valid_states[0]


def reset_city_selection():
    """Reset city when the selected state changes."""

    selected_state = st.session_state["selected_state"]

    cities = get_cities_for_state(
        selected_state,
        city_state_map,
    )

    st.session_state["selected_city"] = (
        cities[0] if cities else None
    )


initial_cities = get_cities_for_state(
    st.session_state["selected_state"],
    city_state_map,
)

if not initial_cities:
    st.error("No cities are available for the selected state.")
    st.stop()

if (
    "selected_city" not in st.session_state
    or st.session_state["selected_city"] not in initial_cities
):
    st.session_state["selected_city"] = initial_cities[0]


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.header("Investment Analysis")

st.sidebar.info(
    """
    Enter the property's characteristics to obtain
    a classification from the trained model.
    """
)

st.sidebar.caption(
    "Location options are generated from the cleaned dataset."
)


# ============================================================
# 8. PROPERTY INPUT FORM
# ============================================================

st.subheader("Enter Property Details")


# ------------------------------------------------------------
# LOCATION
# ------------------------------------------------------------

st.markdown("### 1. Location Details")

col1, col2, col3 = st.columns(3)

with col1:
    state = st.selectbox(
        "State",
        options=valid_states,
        key="selected_state",
        on_change=reset_city_selection,
    )

cities = get_cities_for_state(
    state,
    city_state_map,
)

if not cities:
    st.error("No cities are available for this state.")
    st.stop()

if st.session_state.get("selected_city") not in cities:
    st.session_state["selected_city"] = cities[0]

with col2:
    city = st.selectbox(
        "City",
        options=cities,
        key="selected_city",
    )

city_df = df[
    df["City"].astype(str) == str(city)
].copy()

localities = get_options(
    city_df,
    "Locality",
)

if not localities:
    st.error(f"No locality records are available for {city}.")
    st.stop()

with col3:
    locality = st.selectbox(
        "Locality",
        options=localities,
    )


st.divider()


# ------------------------------------------------------------
# PROPERTY DETAILS
# ------------------------------------------------------------

st.markdown("### 2. Property Details")

col1, col2, col3 = st.columns(3)

property_types = get_options(df, "Property_Type")
furnished_options = get_options(df, "Furnished_Status")

if not property_types or not furnished_options:
    st.error("Property category options are unavailable.")
    st.stop()

with col1:
    property_type = st.selectbox(
        "Property Type",
        property_types,
    )

with col2:
    bhk_values = sorted(
        pd.to_numeric(
            df["BHK"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    bhk = st.selectbox(
        "BHK Configuration",
        bhk_values,
        index=(
            bhk_values.index(3)
            if 3 in bhk_values
            else 0
        ),
    )

with col3:
    furnished_status = st.selectbox(
        "Furnished Status",
        furnished_options,
    )

col1, col2 = st.columns(2)

with col1:
    size = st.number_input(
        "Property Size (Sq.Ft.)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50,
    )

with col2:
    year_min = int(
        pd.to_numeric(
            df["Year_Built"],
            errors="coerce",
        ).min()
    )

    year_max = int(
        pd.to_numeric(
            df["Year_Built"],
            errors="coerce",
        ).max()
    )

    year_built = st.number_input(
        "Year Built",
        min_value=year_min,
        max_value=year_max,
        value=min(2020, year_max),
        step=1,
    )


st.divider()


# ------------------------------------------------------------
# BUILDING DETAILS
# ------------------------------------------------------------

st.markdown("### 3. Building Details")

col1, col2 = st.columns(2)

with col1:
    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

with col2:
    floor_no = st.number_input(
        "Property Floor Number",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
    )

if floor_no > total_floors:
    st.warning(
        "Property floor exceeds total floors. "
        "Floor information will be flagged as invalid."
    )


st.divider()


# ------------------------------------------------------------
# NEARBY FACILITIES
# ------------------------------------------------------------

st.markdown("### 4. Nearby Facilities")

col1, col2 = st.columns(2)

with col1:
    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        max_value=100,
        value=int(safe_numeric_median("Nearby_Schools", 3)),
        step=1,
    )

with col2:
    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        max_value=100,
        value=int(safe_numeric_median("Nearby_Hospitals", 2)),
        step=1,
    )

transport_options = get_options(
    df,
    "Public_Transport_Accessibility",
)

if not transport_options:
    st.error("Transport accessibility options are unavailable.")
    st.stop()

transport = st.selectbox(
    "Public Transport Accessibility",
    transport_options,
)


st.divider()


# ------------------------------------------------------------
# AMENITIES AND SAFETY
# ------------------------------------------------------------

st.markdown("### 5. Amenities and Safety")

amenity_options = get_options(
    df,
    "Amenities",
)

if not amenity_options:
    st.error("Amenity options are unavailable.")
    st.stop()

amenities = st.selectbox(
    "Amenities Configuration",
    amenity_options,
)

col1, col2 = st.columns(2)

with col1:
    parking = st.selectbox(
        "Parking Available",
        ["Yes", "No"],
    )

with col2:
    security = st.selectbox(
        "Security Available",
        ["Yes", "No"],
    )

col1, col2 = st.columns(2)

facing_options = get_options(df, "Facing")
owner_options = get_options(df, "Owner_Type")

if not facing_options or not owner_options:
    st.error("Facing or owner type options are unavailable.")
    st.stop()

with col1:
    facing = st.selectbox(
        "Property Facing",
        facing_options,
    )

with col2:
    owner_type = st.selectbox(
        "Owner Type",
        owner_options,
    )

availability_options = get_options(
    df,
    "Availability_Status",
)

if not availability_options:
    st.error("Availability options are unavailable.")
    st.stop()

availability = st.selectbox(
    "Availability Status",
    availability_options,
)


st.divider()


# ============================================================
# 9. PREDICTION
# ============================================================

submitted = st.button(
    "Analyze Investment",
    use_container_width=True,
    type="primary",
)


if submitted:

    floor_invalid = int(
        floor_no > total_floors
    )

    floor_ratio = (
        floor_no / total_floors
        if not floor_invalid
        else np.nan
    )

    age_of_property = 2026 - int(year_built)

    amenity_features = get_amenity_features(amenities)

    property_data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": int(bhk),
        "Size_in_SqFt": float(size),
        "Year_Built": int(year_built),
        "Furnished_Status": furnished_status,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors),
        "Age_of_Property": int(age_of_property),
        "Nearby_Schools": int(nearby_schools),
        "Nearby_Hospitals": int(nearby_hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": int(parking == "Yes"),
        "Security": int(security == "Yes"),
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
        "Floor_Data_Invalid": floor_invalid,
        "Floor_Ratio": floor_ratio,
        **amenity_features,
    }

    input_df = pd.DataFrame([property_data])

    try:

        expected_features = list(
            classifier.feature_names_in_
        )

        missing_features = [
            feature
            for feature in expected_features
            if feature not in input_df.columns
        ]

        unexpected_features = [
            feature
            for feature in input_df.columns
            if feature not in expected_features
        ]

        if missing_features or unexpected_features:
            raise ValueError(
                "Model input schema mismatch. "
                f"Missing: {missing_features}; "
                f"Unexpected: {unexpected_features}"
            )

        input_df = input_df[expected_features]

        prediction = int(
            classifier.predict(input_df)[0]
        )

        probabilities = classifier.predict_proba(input_df)[0]

        class_labels = list(
            classifier.named_steps["model"].classes_
        )

        predicted_probability = float(
            probabilities[class_labels.index(prediction)]
        )

        good_probability = float(
            probabilities[class_labels.index(1)]
        )

        st.session_state["investment_result"] = {
            "prediction": prediction,
            "predicted_probability": predicted_probability,
            "good_probability": good_probability,
            "property_data": property_data.copy(),
            "input_df": input_df.copy(),
        }

    except Exception as error:
        st.error(
            f"Prediction could not be generated: {error}"
        )
        st.exception(error)


# ============================================================
# 10. DISPLAY PREDICTION
# ============================================================

result = st.session_state.get("investment_result")

if result is not None:

    st.divider()

    st.subheader("Investment Classification Result")

    st.caption(
        "Showing the result for the most recently submitted property."
    )

    prediction = result["prediction"]
    predicted_probability = result["predicted_probability"]
    good_probability = result["good_probability"]

    submitted_property = result["property_data"]
    submitted_input_df = result["input_df"]

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.success(
                "Relative Price Opportunity: Below City Median"
            )
        else:
            st.warning(
                "Relative Price Opportunity: At or Above City Median"
            )

    with result_col2:

        st.metric(
            "Model Probability for Predicted Class",
            f"{predicted_probability * 100:.2f}%",
        )

    st.progress(
        min(max(good_probability, 0.0), 1.0),
        text=(
            "Model probability of the below-city-median "
            "price-per-square-foot proxy"
        ),
    )

    st.markdown("### Submitted Property Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "City",
        submitted_property["City"],
    )

    summary_col2.metric(
        "Property Size",
        f"{submitted_property['Size_in_SqFt']:,.0f} sq.ft.",
    )

    summary_col3.metric(
        "BHK",
        str(submitted_property["BHK"]),
    )

    st.markdown("### Prediction Interpretation")

    if prediction == 1:

        st.write(
            """
            Based on the trained classifier, this property is
            assigned to the category associated with a price
            per square foot below its city's median in the
            training data.
            """
        )

    else:

        st.write(
            """
            Based on the trained classifier, this property is
            assigned to the category associated with a price
            per square foot at or above its city's median in
            the training data.
            """
        )

    st.info(
        """
        **Important limitation**

        This is an experimental classification proxy. It does
        not establish that a property is profitable, undervalued,
        or likely to appreciate.

        The probability is the model's estimated probability
        of the proxy class, not the probability of making money.

        The underlying dataset has unusually uniform price
        patterns, so predictions should not be used for
        real-world property purchase decisions.
        """
    )

    with st.expander("View submitted property details"):

        summary_df = pd.DataFrame([
            {
                "Field": key,
                "Submitted Value": value,
            }
            for key, value in submitted_property.items()
        ])

        st.dataframe(
            summary_df,
            width="stretch",
            hide_index=True,
        )

    with st.expander("View model input data"):

        st.dataframe(
            submitted_input_df,
            width="stretch",
            hide_index=True,
        )

else:

    st.info(
        "Enter the property details and click Analyze Investment "
        "to generate a classification."
    )