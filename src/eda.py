
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data/processed/cleaned_housing_data.csv"
OUTPUT = ROOT / "reports/figures"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

sns.set_theme(style="whitegrid")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=150, bbox_inches="tight")
    plt.close()


# 1. Property price distribution
plt.figure()
sns.histplot(df["Price_in_Lakhs"], bins=40, kde=True)
plt.title("Property Price Distribution")
save_plot("01_price_distribution.png")

# 2. Property size distribution
plt.figure()
sns.histplot(df["Size_in_SqFt"], bins=40, kde=True)
plt.title("Property Size Distribution")
save_plot("02_size_distribution.png")

# 3. Price per sqft by property type
plt.figure()
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt")
plt.title("Price per SqFt by Property Type")
save_plot("03_price_by_property_type.png")

# 4. Property size vs price
plt.figure()
sns.scatterplot(data=df, x="Size_in_SqFt", y="Price_in_Lakhs",
                alpha=0.3)
plt.title("Property Size vs Price")
save_plot("04_size_vs_price.png")

# 5. Outlier analysis
plt.figure()
sns.boxplot(data=df[["Size_in_SqFt", "Price_per_SqFt"]])
plt.title("Size and Price per SqFt Outliers")
save_plot("05_outliers.png")

# 6. Average price per sqft by state
state_price = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
plt.figure(figsize=(10, 7))
state_price.plot(kind="barh")
plt.title("Average Price per SqFt by State")
save_plot("06_state_price.png")

# 7. Average property price by city
city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
plt.figure(figsize=(10, 8))
city_price.plot(kind="barh")
plt.title("Average Property Price by City")
save_plot("07_city_price.png")

# 8. Median property age by locality
locality_age = df.groupby("Locality")["Age_of_Property"].median()
locality_age.nlargest(15).plot(kind="bar", figsize=(10, 5))
plt.title("15 Localities with Highest Median Property Age")
save_plot("08_locality_age.png")

# 9. BHK distribution across cities
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="City", hue="BHK")
plt.xticks(rotation=90)
plt.title("BHK Distribution Across Cities")
save_plot("09_bhk_city.png")

# 10. Top 5 expensive localities
top_localities = (
    df.groupby("Locality")["Price_per_SqFt"]
    .mean().nlargest(5).index
)

top_data = df[df["Locality"].isin(top_localities)]

plt.figure(figsize=(9, 5))
sns.boxplot(data=top_data, x="Locality", y="Price_per_SqFt")
plt.title("Top Five Expensive Localities")
save_plot("10_top_localities.png")

# 11. Numeric correlation
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes(include="number").corr(),
    cmap="coolwarm", center=0
)
plt.title("Numerical Feature Correlation")
save_plot("11_correlation.png")

# 12. Schools vs price per sqft
plt.figure()
sns.boxplot(data=df, x="Nearby_Schools", y="Price_per_SqFt")
plt.title("Schools vs Price per SqFt")
save_plot("12_schools_price.png")

# 13. Hospitals vs price per sqft
plt.figure()
sns.boxplot(data=df, x="Nearby_Hospitals", y="Price_per_SqFt")
plt.title("Hospitals vs Price per SqFt")
save_plot("13_hospitals_price.png")

# 14. Furnishing vs price
plt.figure()
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs")
plt.title("Furnishing Status vs Property Price")
save_plot("14_furnishing_price.png")

# 15. Facing vs price per sqft
plt.figure()
sns.boxplot(data=df, x="Facing", y="Price_per_SqFt")
plt.title("Facing Direction vs Price per SqFt")
save_plot("15_facing_price.png")

# 16. Owner type distribution
plt.figure()
df["Owner_Type"].value_counts().plot(kind="bar")
plt.title("Properties by Owner Type")
save_plot("16_owner_type.png")

# 17. Availability status
plt.figure()
df["Availability_Status"].value_counts().plot(kind="bar")
plt.title("Property Availability Status")
save_plot("17_availability.png")

# 18. Parking vs price
plt.figure()
sns.boxplot(data=df, x="Parking_Space", y="Price_in_Lakhs")
plt.title("Parking vs Property Price")
save_plot("18_parking_price.png")

# 19. Amenities vs price per sqft
plt.figure()
sns.boxplot(data=df, x="Amenity_Count", y="Price_per_SqFt")
plt.title("Amenity Count vs Price per SqFt")
save_plot("19_amenities_price.png")

# 20. Transport vs price per sqft
plt.figure()
sns.boxplot(
    data=df,
    x="Public_Transport_Accessibility",
    y="Price_per_SqFt"
)
plt.title("Transport Accessibility vs Price per SqFt")
save_plot("20_transport_price.png")

print("All 20 EDA visualizations generated.")
print("Saved in:", OUTPUT)
