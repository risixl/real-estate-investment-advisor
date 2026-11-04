
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/model_ready_data.csv"
REPORT_PATH = ROOT / "reports/metrics/regression_diagnostics.txt"

df = pd.read_csv(DATA_PATH)

target = "Target_Property_Price"

print("=" * 60)
print("REGRESSION DIAGNOSTICS")
print("=" * 60)

print("\n1. TARGET DISTRIBUTION")
print(df[target].describe())

print("\n2. NUMERICAL CORRELATIONS WITH PROPERTY PRICE")

numeric = df.select_dtypes(include="number")

correlations = (
    numeric.corr()[target]
    .drop(target)
    .sort_values(key=abs, ascending=False)
)

print(correlations.head(20))

print("\n3. PROPERTY PRICE BY CITY")

city_stats = df.groupby("City")[target].agg(
    ["count", "mean", "median", "std"]
).sort_values("mean", ascending=False)

print(city_stats.to_string())

print("\n4. PROPERTY PRICE BY PROPERTY TYPE")

print(
    df.groupby("Property_Type")[target]
    .agg(["count", "mean", "median", "std"])
)

print("\n5. PROPERTY PRICE BY BHK")

print(
    df.groupby("BHK")[target]
    .agg(["count", "mean", "median", "std"])
)

print("\n6. PROPERTY PRICE BY SIZE GROUP")

df["Size_Group"] = pd.cut(
    df["Size_in_SqFt"],
    bins=[0, 500, 1000, 1500, 2000, 3000, np.inf],
    labels=["Below 500", "500-1000", "1000-1500",
            "1500-2000", "2000-3000", "Above 3000"]
)

print(
    df.groupby("Size_Group", observed=False)[target]
    .agg(["count", "mean", "median", "std"])
)

print("\n7. LOCALITY PRICE VARIATION")

locality_stats = df.groupby("Locality")[target].agg(
    ["count", "mean", "std"]
)

print(locality_stats.sort_values("mean", ascending=False).head(15))

print("\n8. TARGET VARIABILITY")

print("Target standard deviation:", df[target].std())
print("Unique target values:", df[target].nunique())

print("\n9. PRICE DISTRIBUTION BY CITY")

print(
    df.groupby("City")[target].std().describe()
)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("REGRESSION DIAGNOSTICS REPORT\n\n")

    f.write("TARGET DISTRIBUTION\n")
    f.write(df[target].describe().to_string())

    f.write("\n\nNUMERICAL CORRELATIONS\n")
    f.write(correlations.to_string())

    f.write("\n\nCITY PRICE STATISTICS\n")
    f.write(city_stats.to_string())

    f.write("\n\nPROPERTY TYPE STATISTICS\n")
    f.write(
        df.groupby("Property_Type")[target]
        .agg(["count", "mean", "median", "std"])
        .to_string()
    )

print("\nDiagnostic report saved:", REPORT_PATH)
