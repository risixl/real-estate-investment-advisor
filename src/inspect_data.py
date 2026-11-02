
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "india_housing_prices.csv"

REPORT_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "data_profile.txt"

REFERENCE_YEAR = datetime.now().year


# ============================================================
# 2. DATASET INSPECTION FUNCTION
# ============================================================

def inspect_dataset():

    print("=" * 70)
    print("           REAL ESTATE DATASET QUALITY ASSESSMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. VERIFY DATASET
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # --------------------------------------------------------
    # 2. LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Dataset loaded successfully!")

    # Store profiling results for the report
    report_sections = []

    # --------------------------------------------------------
    # 3. DATASET DIMENSIONS
    # --------------------------------------------------------

    print("\n1. DATASET DIMENSIONS")

    dimensions = (
        f"Rows: {df.shape[0]:,}\n"
        f"Columns: {df.shape[1]}"
    )

    print(dimensions)
    report_sections.append(("DATASET DIMENSIONS", dimensions))

    # --------------------------------------------------------
    # 4. COLUMN NAMES
    # --------------------------------------------------------

    print("\n2. COLUMN NAMES")

    column_names = "\n".join(
        f"- {column}" for column in df.columns
    )

    print(column_names)
    report_sections.append(("COLUMN NAMES", column_names))

    # --------------------------------------------------------
    # 5. DATA TYPES
    # --------------------------------------------------------

    print("\n3. DATA TYPES")

    data_types = df.dtypes.to_string()

    print(data_types)
    report_sections.append(("DATA TYPES", data_types))

    # --------------------------------------------------------
    # 6. DATASET INFORMATION
    # --------------------------------------------------------

    print("\n4. DATASET INFORMATION")

    df.info()

    # --------------------------------------------------------
    # 7. MISSING VALUES
    # --------------------------------------------------------

    print("\n5. MISSING VALUES")

    missing = pd.DataFrame({
        "Missing_Count": df.isnull().sum(),
        "Missing_Percentage": (
            df.isnull().mean() * 100
        ).round(2)
    })

    missing_report = missing[
        missing["Missing_Count"] > 0
    ]

    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.to_string())

    report_sections.append(
        ("MISSING VALUES", missing.to_string())
    )

    # --------------------------------------------------------
    # 8. DUPLICATE RECORDS
    # --------------------------------------------------------

    print("\n6. DUPLICATE RECORDS")

    duplicate_rows = df.duplicated().sum()

    print("Complete duplicate rows:", duplicate_rows)

    duplicate_report = (
        f"Complete duplicate rows: {duplicate_rows}"
    )

    if "ID" in df.columns:

        duplicate_ids = df["ID"].duplicated().sum()

        print("Duplicate property IDs:", duplicate_ids)

        duplicate_report += (
            f"\nDuplicate property IDs: {duplicate_ids}"
        )

    report_sections.append(
        ("DUPLICATE RECORDS", duplicate_report)
    )

    # --------------------------------------------------------
    # 9. NUMERICAL STATISTICS
    # --------------------------------------------------------

    print("\n7. NUMERICAL STATISTICS")

    numeric = df.select_dtypes(include=np.number)

    numerical_summary = numeric.describe().T

    print(numerical_summary.to_string())

    report_sections.append(
        ("NUMERICAL STATISTICS",
         numerical_summary.to_string())
    )

    # --------------------------------------------------------
    # 10. CATEGORICAL STATISTICS
    # --------------------------------------------------------

    print("\n8. CATEGORICAL STATISTICS")

    categorical = df.select_dtypes(
        include=["object", "category", "string"]
    )

    if not categorical.empty:

        categorical_summary = categorical.describe().T

        print(categorical_summary.to_string())

        report_sections.append(
            ("CATEGORICAL STATISTICS",
             categorical_summary.to_string())
        )

    else:
        print("No categorical columns found.")

    # --------------------------------------------------------
    # 11. UNIQUE CATEGORICAL VALUES
    # --------------------------------------------------------

    print("\n9. UNIQUE CATEGORICAL VALUES")

    unique_values_report = []

    for column in categorical.columns:

        print(f"\n{column}:")

        values = df[column].value_counts(
            dropna=False
        ).head(15)

        print(values.to_string())

        unique_values_report.append(
            f"\n{column}:\n{values.to_string()}"
        )

    report_sections.append(
        ("UNIQUE CATEGORICAL VALUES",
         "\n".join(unique_values_report))
    )

    # --------------------------------------------------------
    # 12. NUMERICAL ANOMALIES
    # --------------------------------------------------------

    print("\n10. NUMERICAL ANOMALIES")

    if not numeric.empty:

        infinite_count = np.isinf(numeric).sum().sum()

        print("Infinite values:", infinite_count)

        anomaly_report = [
            f"Infinite values: {infinite_count}"
        ]

        for column in numeric.columns:

            negative_count = (numeric[column] < 0).sum()

            if negative_count > 0:

                message = (
                    f"{column}: {negative_count} negative values"
                )

                print(message)
                anomaly_report.append(message)

        if len(anomaly_report) == 1:
            print("No negative numerical values found.")

        report_sections.append(
            ("NUMERICAL ANOMALIES",
             "\n".join(anomaly_report))
        )

    # --------------------------------------------------------
    # 13. PRICE PER SQFT VALIDATION
    # --------------------------------------------------------

    print("\n11. PRICE PER SQFT VALIDATION")

    price_columns = [
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "Price_per_SqFt"
    ]

    if all(column in df.columns for column in price_columns):

        valid_mask = (
            df[price_columns].notna().all(axis=1)
            & (df["Size_in_SqFt"] > 0)
        )

        calculated_lakhs = (
            df.loc[valid_mask, "Price_in_Lakhs"]
            / df.loc[valid_mask, "Size_in_SqFt"]
        )

        difference = (
            df.loc[valid_mask, "Price_per_SqFt"]
            - calculated_lakhs
        ).abs()

        mismatches = (difference > 0.00001).sum()

        print("Valid records:", valid_mask.sum())
        print(
            "Maximum difference:",
            difference.max()
        )
        print(
            "Mean difference:",
            difference.mean()
        )
        print("Mismatches:", mismatches)

        average_price_inr = (
            df.loc[valid_mask, "Price_per_SqFt"].mean()
            * 100000
        )

        print(
            "Average price per sqft (INR):",
            round(average_price_inr, 2)
        )

        price_report = (
            f"Valid records: {valid_mask.sum()}\n"
            f"Maximum difference: {difference.max()}\n"
            f"Mean difference: {difference.mean()}\n"
            f"Mismatches: {mismatches}\n"
            f"Average price per sqft (INR): "
            f"{average_price_inr:.2f}"
        )

    else:

        price_report = (
            "Price validation skipped. Required columns missing."
        )

        print(price_report)

    report_sections.append(
        ("PRICE PER SQFT VALIDATION", price_report)
    )

    # --------------------------------------------------------
    # 14. YEAR BUILT VALIDATION
    # --------------------------------------------------------

    print("\n12. YEAR BUILT VALIDATION")

    if "Year_Built" in df.columns:

        current_year = REFERENCE_YEAR

        print("Current year:", current_year)
        print("Minimum year:", df["Year_Built"].min())
        print("Maximum year:", df["Year_Built"].max())

        invalid_years = (
            (df["Year_Built"] < 1800)
            | (df["Year_Built"] > current_year)
        )

        invalid_count = invalid_years.sum()

        print(
            "Suspicious construction years:",
            invalid_count
        )

        year_report = (
            f"Reference year: {current_year}\n"
            f"Minimum year: {df['Year_Built'].min()}\n"
            f"Maximum year: {df['Year_Built'].max()}\n"
            f"Suspicious construction years: {invalid_count}"
        )

    else:

        year_report = "Year_Built column not available."

        print(year_report)

    report_sections.append(
        ("YEAR BUILT VALIDATION", year_report)
    )

    # --------------------------------------------------------
    # 15. TARGET VARIABLE AVAILABILITY
    # --------------------------------------------------------

    print("\n13. TARGET VARIABLE AVAILABILITY")

    possible_targets = [
        "Future_Price",
        "Future_Price_5_Years",
        "Good_Investment",
        "Investment_Return",
        "Appreciation_Rate"
    ]

    target_report = []

    for target in possible_targets:

        status = (
            "Present"
            if target in df.columns
            else "Absent"
        )

        message = f"{target}: {status}"

        print(message)
        target_report.append(message)

    report_sections.append(
        ("TARGET VARIABLE AVAILABILITY",
         "\n".join(target_report))
    )

    # --------------------------------------------------------
    # 16. PROPERTY AGE VALIDATION
    # --------------------------------------------------------

    print("\n14. PROPERTY AGE VALIDATION")

    if (
        "Year_Built" in df.columns
        and "Age_of_Property" in df.columns
    ):

        expected_age = (
            REFERENCE_YEAR - df["Year_Built"]
        )

        age_mismatches = (
            df["Age_of_Property"] != expected_age
        ).sum()

        print("Reference year:", REFERENCE_YEAR)

        print("Age mismatches:", age_mismatches)

        print(
            "Expected average property age:",
            round(expected_age.mean(), 2)
        )

        age_report = (
            f"Reference year: {REFERENCE_YEAR}\n"
            f"Age mismatches: {age_mismatches}\n"
            f"Expected average property age: "
            f"{expected_age.mean():.2f}"
        )

    else:

        age_report = (
            "Property age validation skipped. "
            "Required columns missing."
        )

        print(age_report)

    report_sections.append(
        ("PROPERTY AGE VALIDATION", age_report)
    )

    # --------------------------------------------------------
    # 17. SAVE DATA PROFILING REPORT
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 70 + "\n")
        file.write("REAL ESTATE DATASET PROFILE REPORT\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            f"Generated: {datetime.now()}\n"
        )

        file.write(
            f"Dataset: {DATA_PATH.name}\n"
        )

        file.write(
            f"Dataset Shape: {df.shape}\n\n"
        )

        for section_title, section_content in report_sections:

            file.write("\n")
            file.write("=" * 70 + "\n")
            file.write(f"{section_title}\n")
            file.write("=" * 70 + "\n")

            file.write(section_content)
            file.write("\n")

        file.write("\n" + "=" * 70 + "\n")
        file.write("END OF DATASET PROFILE REPORT\n")
        file.write("=" * 70)

    # --------------------------------------------------------
    # 18. FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET PROFILING COMPLETED SUCCESSFULLY")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 70)

    return df


# ============================================================
# 19. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    df = inspect_dataset()
