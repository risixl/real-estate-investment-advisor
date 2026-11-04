
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data/raw/india_housing_prices.csv"
)

print("ORIGINAL RECORDS:", len(df))

checks = {
    "Invalid price": df["Price_in_Lakhs"] <= 0,
    "Invalid size": df["Size_in_SqFt"] <= 0,
    "Invalid BHK": df["BHK"] <= 0,
    "Invalid construction year": df["Year_Built"] > 2026,
    "Negative floor": df["Floor_No"] < 0,
    "Invalid total floors": df["Total_Floors"] <= 0,
    "Floor exceeds total floors": (
        df["Floor_No"] > df["Total_Floors"]
    )
}

print("\nINDIVIDUAL VALIDATION FAILURES")

for name, condition in checks.items():
    print(f"{name}: {condition.sum():,}")

invalid_floor = (
    (df["Floor_No"] < 0)
    | (df["Total_Floors"] <= 0)
    | (df["Floor_No"] > df["Total_Floors"])
)

print("\nTOTAL INVALID FLOOR RECORDS:", invalid_floor.sum())

print(
    "\nRECORDS AFTER CORE FILTERS:",
    len(df[
        (df["Price_in_Lakhs"] > 0)
        & (df["Size_in_SqFt"] > 0)
        & (df["BHK"] > 0)
        & (df["Year_Built"] <= 2026)
    ])
)
