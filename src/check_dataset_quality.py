
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data/processed/model_ready_data.csv"
)

print("=" * 60)
print("DATASET REALISM ASSESSMENT")
print("=" * 60)

# Check target distribution
print("\n1. PRICE DISTRIBUTION")

print(
    df["Price_in_Lakhs"]
    .value_counts(bins=10, sort=False)
)

# Check relationship between price and size
print("\n2. PRICE VS SIZE")

print(
    df.groupby(
        pd.qcut(df["Size_in_SqFt"], 10, duplicates="drop")
    )["Price_in_Lakhs"].agg(["count", "mean", "median"])
)

# Check price distribution by city
print("\n3. CITY PRICE VARIATION")

print(
    df.groupby("City")["Price_in_Lakhs"]
    .agg(["mean", "std", "min", "max"])
)

# Check relationship between BHK and price
print("\n4. BHK VS PRICE")

print(
    df.groupby("BHK")["Price_in_Lakhs"]
    .agg(["mean", "median", "std"])
)

# Check whether target appears rounded or uniformly generated
print("\n5. PRICE ROUNDING")

print(
    "Unique decimal places:",
    df["Price_in_Lakhs"].astype(str)
    .str.split(".").str[-1].str.len().value_counts()
)

# Check price range
print("\n6. PRICE QUANTILES")

print(
    df["Price_in_Lakhs"].quantile(
        [0, .05, .10, .25, .50, .75, .90, .95, 1]
    )
)

# Check correlations
print("\n7. FEATURE CORRELATIONS")

numeric = df.select_dtypes(include="number")

corr = numeric.corr()["Price_in_Lakhs"].drop(
    ["Price_in_Lakhs", "Target_Property_Price"],
    errors="ignore"
)

print(corr.abs().sort_values(ascending=False).head(15))

print("\nDATASET QUALITY CHECK COMPLETED")
