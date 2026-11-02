
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/cleaned_housing_data.csv"
OUTPUT = ROOT / "data/processed/model_ready_data.csv"


def create_targets():

    df = pd.read_csv(INPUT)

    # City-level median price per sqft
    city_median = df.groupby("City")["Price_per_SqFt"].transform("median")

    # 1 = below or equal to city median
    # 0 = above city median
    df["Good_Investment"] = (
        df["Price_per_SqFt"] <= city_median
    ).astype(int)

    # Regression target: current property price
    df["Target_Property_Price"] = df["Price_in_Lakhs"]

    # Illustrative 5-year scenario, not a learned target
    annual_rate = 0.08
    years = 5

    df["Scenario_Future_Price"] = (
        df["Price_in_Lakhs"] * (1 + annual_rate) ** years
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("Target dataset created.")
    print("\nClassification distribution:")
    print(df["Good_Investment"].value_counts(normalize=True))

    print("\nRegression target summary:")
    print(df["Target_Property_Price"].describe())

    print("\nScenario assumptions:")
    print("Annual appreciation:", annual_rate * 100, "%")
    print("Projection period:", years, "years")

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    create_targets()
