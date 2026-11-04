
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data/raw/india_housing_prices.csv"
OUTPUT_PATH = ROOT / "data/processed/cleaned_housing_data.csv"


def preprocess_data():

    df = pd.read_csv(RAW_PATH, low_memory=False)

    original_rows = len(df)

    print("Original records:", original_rows)

    # Remove exact duplicate records
    df = df.drop_duplicates().copy()

    # Validate unique IDs
    df = df.drop_duplicates(subset="ID", keep="first")

    # Remove only records with invalid essential values
    valid = (
        (df["Price_in_Lakhs"] > 0)
        & (df["Size_in_SqFt"] > 0)
        & (df["BHK"] > 0)
        & (df["Year_Built"] >= 1800)
        & (df["Year_Built"] <= 2026)
    )

    df = df.loc[valid].copy()

    # Recalculate property age
    df["Age_of_Property"] = 2026 - df["Year_Built"]

    # Recalculate price per sqft in lakhs
    df["Price_per_SqFt"] = (
        df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    )

    # Flag floor inconsistencies rather than deleting rows
    df["Floor_Data_Invalid"] = (
        (df["Floor_No"] < 0)
        | (df["Total_Floors"] <= 0)
        | (df["Floor_No"] > df["Total_Floors"])
    ).astype(int)

    df["Floor_Ratio"] = np.where(
        df["Floor_Data_Invalid"] == 0,
        df["Floor_No"] / df["Total_Floors"],
        np.nan
    )

    # Binary encoding
    for col in ["Parking_Space", "Security"]:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0
        })

    # Amenities feature engineering
    amenities = df["Amenities"].fillna("").astype(str)

    amenity_names = [
        "Gym", "Pool", "Garden",
        "Clubhouse", "Playground"
    ]

    for amenity in amenity_names:
        df[f"Amenity_{amenity}"] = amenities.apply(
            lambda value: int(
                amenity.lower() in [
                    item.strip().lower()
                    for item in value.split(",")
                ]
            )
        )

    df["Amenity_Count"] = df[
        [f"Amenity_{a}" for a in amenity_names]
    ].sum(axis=1)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPREPROCESSING SUMMARY")
    print("Original records:", original_rows)
    print("Final records:", len(df))
    print("Records removed:", original_rows - len(df))
    print("Invalid floor records flagged:",
          df["Floor_Data_Invalid"].sum())
    print("Missing values:", df.isnull().sum().sum())
    print("Final shape:", df.shape)
    print("Saved:", OUTPUT_PATH)

    return df


if __name__ == "__main__":
    preprocess_data()
