
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/india_housing_prices.csv"
OUTPUT_PATH = ROOT / "data/processed/cleaned_housing_data.csv"


def preprocess_data():

    df = pd.read_csv(RAW_PATH, low_memory=False)

    print("Original shape:", df.shape)

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Validate unique property IDs
    df = df.drop_duplicates(subset=["ID"], keep="first")

    # Convert numerical columns safely
    numeric_columns = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Price_per_SqFt", "Year_Built",
        "Floor_No", "Total_Floors",
        "Nearby_Schools", "Nearby_Hospitals"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid essential records
    df = df.dropna(
        subset=["Price_in_Lakhs", "Size_in_SqFt", "BHK"]
    )

    df = df[
        (df["Price_in_Lakhs"] > 0)
        & (df["Size_in_SqFt"] > 0)
        & (df["BHK"] > 0)
        & (df["Year_Built"] <= 2026)
    ].copy()

    # Recalculate age consistently
    df["Age_of_Property"] = 2026 - df["Year_Built"]

    # Recalculate price per sqft in lakhs
    df["Price_per_SqFt"] = (
        df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    )

    # Engineer floor ratio
    df["Floor_Ratio"] = (
        df["Floor_No"] / df["Total_Floors"].replace(0, np.nan)
    )

    # Encode binary categorical features
    binary_mapping = {
        "Parking_Space": {"Yes": 1, "No": 0},
        "Security": {"Yes": 1, "No": 0}
    }

    for col, mapping in binary_mapping.items():
        df[col] = df[col].map(mapping)

    # Extract amenities into individual indicators
    amenities = df["Amenities"].fillna("").astype(str)

    for amenity in ["Gym", "Pool", "Garden",
                    "Clubhouse", "Playground"]:

        df[f"Amenity_{amenity}"] = amenities.apply(
            lambda value: int(
                amenity.lower() in [
                    item.strip().lower()
                    for item in value.split(",")
                ]
            )
        )

    df["Amenity_Count"] = df[
        [f"Amenity_{a}" for a in [
            "Gym", "Pool", "Garden", "Clubhouse", "Playground"
        ]]
    ].sum(axis=1)

    # Remove invalid floor relationships
    df = df[
        (df["Floor_No"] >= 0)
        & (df["Total_Floors"] > 0)
        & (df["Floor_No"] <= df["Total_Floors"])
    ].copy()

    # Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessing completed.")
    print("Final shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Saved to:", OUTPUT_PATH)

    return df


if __name__ == "__main__":
    preprocess_data()
