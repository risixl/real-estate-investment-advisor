# Real Estate Investment Advisor
## Technical and Business Project Report

**Project Category:** Data Analytics and Machine Learning  
**Domain:** Indian Residential Real Estate  
**Primary Language:** Python  
**Application Framework:** Streamlit  
**Experiment Tracking:** MLflow

---

# 1. Executive Summary

The Real Estate Investment Advisor is an end-to-end data analytics and machine learning project developed to explore residential property characteristics, analyse pricing patterns, build predictive models, and provide interactive property valuation scenarios.

The project uses a dataset containing 250,000 residential property records and 23 original attributes. The attributes cover location, property type, size, pricing, construction year, furnishing status, amenities, and availability.

The workflow includes:

1. Dataset inspection and quality assessment.
2. Data preprocessing and feature engineering.
3. Exploratory data analysis.
4. Target construction.
5. Classification and regression modelling.
6. Model evaluation and comparison.
7. MLflow experiment tracking.
8. Interactive Streamlit application development.

The project produced a working interactive application with five analytical pages.

The classification model achieved approximately 71.47% accuracy for a relative price-position indicator. However, this target is derived from city-level median prices and does not represent verified investment profitability.

The regression experiment found that XGBoost slightly underperformed a simple mean baseline. This indicates that the available features did not provide useful predictive information for the current-price target in the experiment.

Since the dataset contains no actual future-price observations, five-year property valuation is implemented as an assumption-based appreciation scenario rather than a machine learning forecast.

These distinctions are central to interpreting the project responsibly.

# 2. Business Problem

Property buyers and investors often need to compare residential properties using several factors:

- Property location.
- Property type.
- Number of bedrooms.
- Built-up area.
- Current asking price.
- Price per square foot.
- Property age.
- Furnishing status.
- Amenities.
- Access to schools, hospitals, and public transport.

Analysing these factors manually across a large dataset can be difficult.

The project addresses this problem by providing a centralized analytical application that allows users to explore property information, compare relative pricing, evaluate model results, and calculate future-value scenarios.

## 2.1 Business Objective

Develop an interactive analytical system that:

- Processes a large residential property dataset.
- Identifies patterns in property pricing and characteristics.
- Creates a relative property-price classification indicator.
- Evaluates the feasibility of current-price prediction.
- Calculates future valuation scenarios.
- Presents findings through an accessible dashboard.

## 2.2 Intended Users

- Residential property buyers.
- Real estate analysts.
- Property researchers.
- Data analysts.
- Students and professionals exploring applied machine learning.

The application is an analytical prototype, not a licensed real estate advisory or investment recommendation system.

# 3. Project Objectives

The technical objectives were:

1. Inspect and validate the source dataset.
2. Clean and transform property attributes.
3. Create useful derived features.
4. Conduct exploratory data analysis.
5. Define transparent target variables.
6. Build a classification model.
7. Build and evaluate a regression model.
8. Compare machine learning performance against a baseline.
9. Track experiments using MLflow.
10. Develop a multi-page Streamlit application.
11. Document results, assumptions, and limitations.

# 4. Dataset Description

## 4.1 Dataset Overview

The project dataset is named:

`india_housing_prices.csv`

| Attribute | Description |
|---|---|
| Number of records | 250,000 |
| Original columns | 23 |
| Domain | Residential real estate |
| Geographic coverage | Indian states, cities, and localities |
| Data type | Structured tabular data |

## 4.2 Original Dataset Attributes

The dataset contains the following fields:

| Field | Description |
|---|---|
| ID | Property record identifier |
| State | State of the property |
| City | City of the property |
| Locality | Property locality |
| Property_Type | Residential property category |
| BHK | Number of bedrooms, hall, and kitchen |
| Size_in_SqFt | Property size in square feet |
| Price_in_Lakhs | Property price in lakhs |
| Price_per_SqFt | Price per square foot |
| Year_Built | Construction year |
| Furnished_Status | Furnishing category |
| Floor_No | Property floor number |
| Total_Floors | Total building floors |
| Age_of_Property | Property age |
| Nearby_Schools | Nearby school information |
| Nearby_Hospitals | Nearby hospital information |
| Public_Transport_Accessibility | Transport accessibility |
| Parking_Space | Parking availability |
| Security | Security availability |
| Amenities | Available property amenities |
| Facing | Property facing direction |
| Owner_Type | Property ownership category |
| Availability_Status | Property availability |

## 4.3 Dataset Limitations

The source dataset does not provide:

- Verified historical transaction prices.
- Actual future property prices.
- Verified rental income.
- Realized capital gains.
- Verified investment returns.
- Transaction and maintenance expenses.

Consequently, the project cannot directly train a model to predict actual investment profitability or observed five-year property appreciation.

# 5. Development Environment and Technology Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data loading, cleaning, and transformation |
| NumPy | Numerical calculations |
| Matplotlib | Static charts |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, evaluation, and baseline models |
| XGBoost | Classification and regression |
| Streamlit | Interactive dashboard |
| Plotly | Interactive charts |
| MLflow | Experiment tracking |
| Joblib | Saving and loading trained models |
| SQLite | MLflow tracking backend |

The application was developed and tested in a Python virtual environment on Windows.

# 6. Dataset Assessment

The project dataset contains 250,000 residential property records and 23 original attributes covering property characteristics, location, pricing, amenities, and availability.

## 6.1 Initial Data Quality Findings

| Quality Check | Result |
|---|---:|
| Missing values | 0 |
| Duplicate records | 0 |
| Duplicate property IDs | 0 |
| Infinite numerical values | 0 |
| Construction years | 1990–2023 |

These results describe the initial dataset inspection. They do not mean that every record is logically valid.

## 6.2 Data Validation Observations

Two important data validation issues were identified.

**Price-per-square-foot calculation**

Property prices are recorded in lakhs. The price-per-square-foot calculation therefore required unit correction.

The project uses:

`Price per SqFt = Price in Lakhs / Size in SqFt`

This produces a value in lakhs per square foot. Multiplication by 100,000 converts it into INR per square foot.

**Property age calculation**

The original property age was calculated using an earlier reference year. The preprocessing pipeline recalculates it consistently using 2026:

`Age of Property = 2026 - Year Built`

**Floor validation**

The original dataset contained 116,304 records where `Floor_No` exceeded `Total_Floors`.

These records were flagged instead of automatically deleted. A validity indicator and a derived floor-ratio feature were used to preserve the records while avoiding invalid floor-ratio calculations.

# 7. Data Preprocessing

The preprocessing pipeline is implemented in:

`src/preprocessing.py`

## 7.1 Main Processing Operations

1. Load the original CSV dataset.
2. Remove duplicate rows.
3. Remove duplicate property identifiers.
4. Validate essential numerical fields.
5. Recalculate property age.
6. Correct price-per-square-foot calculations.
7. Validate floor information.
8. Convert parking and security indicators.
9. Extract structured amenity indicators.
10. Generate the cleaned dataset.

## 7.2 Essential Field Validation

The pipeline checks important fields such as:

- Price.
- Property size.
- BHK.
- Construction year.

Records with invalid essential values are excluded where required.

## 7.3 Floor Data Handling

The feature `Floor_Data_Invalid` identifies records where the floor information is inconsistent.

`Floor_Ratio` is calculated only when the floor information is valid.

Invalid floor-ratio values are represented as missing rather than being assigned misleading numerical values.

## 7.4 Parking and Security Transformation

The categorical Yes/No indicators are converted into numerical representations:

- Yes = 1
- No = 0

## 7.5 Amenity Feature Engineering

The original amenities field is parsed into individual indicators:

- Gym
- Pool
- Garden
- Clubhouse
- Playground

An additional `Amenity_Count` feature represents the number of identified amenities.

## 7.6 Preprocessing Output

The cleaned dataset contains:

| Measure | Result |
|---|---:|
| Rows | 250,000 |
| Columns | 31 |
| Records removed | 0 |
| Invalid floor records flagged | 116,304 |
| Missing values after processing | 116,304 |

The missing values are associated with invalid floor-ratio records. They are intentional and represent unavailable valid ratios, not an unnoticed cleaning failure.

Output:

`data/processed/cleaned_housing_data.csv`

# 8. Exploratory Data Analysis

The EDA pipeline is implemented in:

`src/eda.py`

The analysis generated 20 visualizations saved under:

`reports/figures/`

## 8.1 EDA Areas

The visualizations cover:

1. Property price distribution.
2. Property size distribution.
3. Price by property type.
4. Property size versus price.
5. Price outliers.
6. State-level price analysis.
7. City-level price analysis.
8. Locality and property age analysis.
9. BHK distribution by city.
10. Highest-priced localities.
11. Numerical correlation analysis.
12. Nearby schools and property prices.
13. Nearby hospitals and property prices.
14. Furnishing status and property prices.
15. Property facing and prices.
16. Owner type analysis.
17. Property availability analysis.
18. Parking availability and prices.
19. Amenities and property prices.
20. Public transport accessibility and prices.

## 8.2 Purpose of EDA

EDA was used to understand:

- The distribution of property prices.
- Variations in property sizes.
- Differences between property categories.
- Geographic pricing patterns.
- Relationships between numerical variables.
- The distribution of amenities and property characteristics.
- Potential data anomalies.

EDA findings are descriptive. They should not automatically be interpreted as evidence that a feature causes a property to become more valuable.

# 9. Target Variable Strategy and Feature Engineering

The dataset does not contain actual future property prices or verified investment outcomes.

The project therefore uses explicitly defined proxy and scenario variables.

## 9.1 Classification Target: Good_Investment

The classification label is derived by comparing each property's price per square foot with its city's median price per square foot.

Conceptually:

`Good_Investment = 1` when the property's price per square foot is at or below the city median.

`Good_Investment = 0` otherwise.

The generated target distribution is:

| Class | Records |
|---|---:|
| 1 | 125,013 |
| 0 | 124,987 |
| Total | 250,000 |

The name `Good_Investment` is retained in the implementation for compatibility, but its interpretation must be stated precisely.

It represents a **below-city-median price indicator**, not verified investment quality.

A lower relative purchase price does not guarantee future appreciation, rental yield, resale demand, or profitability.

## 9.2 Regression Target: Target_Property_Price

The regression target is:

`Target_Property_Price = Price_in_Lakhs`

This means the model estimates current property prices, not future property prices.

## 9.3 Scenario Future Price

The project creates an illustrative future-value field using an assumed annual appreciation rate of 8% over five years.

Formula:

`Scenario Future Price = Current Price × (1 + 0.08)^5`

The annual appreciation rate is an assumption, not a value learned from the dataset.

This field is used for scenario analysis and is not the output of the trained regression model.

# 10. Machine Learning Methodology

The model training workflow is implemented in:

`src/train_models.py`

## 10.1 Feature Preparation

The training pipeline prepares numerical and categorical features.

Numerical missing values are imputed using the median.

Categorical missing values are imputed using the most frequent category.

Categorical variables are transformed using one-hot encoding with unknown categories handled safely.

The classification pipeline excludes direct pricing fields used to construct the classification label.

The regression pipeline excludes the target and derived future-value fields to avoid direct target leakage.

## 10.2 Classification Model

Algorithm: XGBoost Classifier

Configuration:

| Parameter | Value |
|---|---:|
| Estimators | 120 |
| Maximum depth | 6 |
| Learning rate | 0.08 |
| Subsample | 0.8 |
| Column sample by tree | 0.8 |
| Evaluation metric | Log loss |
| Random state | 42 |
| Training split | 80% |
| Testing split | 20% |

The classification split is stratified to preserve class proportions.

## 10.3 Regression Model

Algorithm: XGBoost Regressor

The regression experiment uses a comparable tree-based configuration and is evaluated against a simple mean baseline.

The baseline predicts the mean training-set property price for every test observation.

This comparison helps determine whether the more complex model provides actual predictive value.

# 11. Model Evaluation

## 11.1 Classification Results

The recorded classification results are:

| Metric | Result |
|---|---:|
| Accuracy | 0.71466 |
| Precision | 0.71232 |
| Recall | 0.72027 |
| F1-score | 0.71628 |

The confusion matrix is:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 17,724 | 7,273 |
| Actual 1 | 6,994 | 18,009 |

The model demonstrates its ability to distinguish the constructed below-city-median price indicator on the held-out test split.

However, these metrics do not demonstrate that the model can identify profitable properties.

## 11.2 Regression Results

The regression target is current property price in lakhs.

| Metric | Mean Baseline | XGBoost |
|---|---:|---:|
| MAE | 122.317776 | 122.364949 |
| RMSE | 141.193240 | 141.286471 |
| R² | Approximately 0.000 | -0.001322 |

### Interpretation

The XGBoost model slightly underperforms the mean baseline.

Its R² is negative, indicating that its predictions perform worse than the baseline mean prediction on the evaluation data.

The result suggests that the available features do not provide useful predictive information for this current-price target in the present experiment.

The findings are consistent with weak or randomized pricing relationships in the dataset. However, the dataset's provenance has not been independently verified, so its generation method cannot be established from these results alone.

## 11.3 Regression Target Distribution

| Statistic | Current Price (₹ lakh) |
|---|---:|
| Count | 250,000 |
| Mean | 254.586854 |
| Standard deviation | 141.349921 |
| Minimum | 10 |
| 25th percentile | 132.55 |
| Median | 253.87 |
| 75th percentile | 376.88 |
| Maximum | 500 |

These statistics describe the target distribution used in the regression experiment.

# 12. MLflow Experiment Tracking

MLflow is used to record and organize the machine learning experiments.

The experiment name is:

`Real_Estate_Investment_Advisor`

The tracking backend uses a local SQLite database:

`mlflow.db`

The model training workflow records experiment information such as:

- Model parameters.
- Evaluation metrics.
- Model artifacts.
- Run information.

The project includes classification and regression experiment runs.

To launch the MLflow interface:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open the local interface at:

`http://127.0.0.1:5000`

The MLflow interface allows experiment runs to be inspected and compared.

# 13. Streamlit Application

The interactive application is implemented in:

`app/app.py`

It uses a multi-page structure to organize the analytical workflow.

## 13.1 Main Dashboard

Provides the main entry point to the application and its analytical sections.

## 13.2 Property Analysis

Allows users to explore property records, property characteristics, and pricing information.

## 13.3 Investment Analysis

Uses the trained classification model to estimate the below-city-median price indicator.

The result should be interpreted as relative pricing, not as a guarantee of investment performance.

## 13.4 Future Valuation

Calculates illustrative future property values using a configurable annual appreciation rate and investment horizon.

Formula:

`Future Value = Present Value × (1 + Rate)^Years`

For example, a current value of ₹50 lakh at an assumed annual appreciation rate of 8% for five years gives approximately ₹73.47 lakh.

This is a scenario calculation, not a machine learning prediction.

## 13.5 Model Performance

Displays the available classification and regression evaluation results.

It allows users to understand the strengths and limitations of the trained models.

## 13.6 MLflow Experiments

Provides access to experiment tracking information and recorded model runs.

# 14. Application Testing

The Streamlit application was manually tested using the following six checks:

| Test | Result |
|---|---|
| Main dashboard loads | PASS |
| Property Analysis works | PASS |
| Investment prediction works | PASS |
| Future Valuation calculates correctly | PASS |
| Model Performance loads | PASS |
| MLflow Experiments loads | PASS |

All six reported application checks passed without errors.

These tests confirm the tested application workflows. They do not replace automated unit, integration, security, or production testing.

# 15. Project Outputs and Artifacts

## 15.1 Processed Data

- `data/processed/cleaned_housing_data.csv`
- `data/processed/model_ready_data.csv`

## 15.2 Trained Models

- `models/classifier.pkl`
- `models/regressor.pkl`
- `models/baseline_regressor.pkl`
- `models/model_metadata.json`

The project also contains a classification label encoder artifact.

## 15.3 Evaluation Reports

- `reports/metrics/model_metrics.json`
- `reports/metrics/classification_confusion_matrix.csv`
- `reports/metrics/classification_feature_importance.csv`
- `reports/metrics/regression_comparison.csv`
- `reports/metrics/regression_diagnostics.txt`
- `reports/metrics/regression_feature_importance.csv`

## 15.4 Visualizations

Twenty EDA charts are generated under:

`reports/figures/`

## 15.5 Documentation

- `README.md`
- `docs/PROJECT_REPORT.md`

# 16. Project Architecture

The project follows a modular workflow.

```text
Raw Housing Dataset
        |
        v
Data Inspection and Quality Checks
        |
        v
Preprocessing Pipeline
        |
        v
Cleaned Housing Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Target Creation and Feature Engineering
        |
        v
Model-Ready Dataset
        |
        v
Classification and Regression Training
        |
        v
Evaluation and MLflow Tracking
        |
        v
Saved Model Artifacts and Reports
        |
        v
Streamlit Application
        |
        v
Interactive Property Analysis
```

The separation of data processing, modelling, application logic, and documentation makes the project easier to maintain and extend.

# 17. Reproducibility and Execution

The project uses a Python virtual environment and a requirements file.

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the main workflow from the project root:

```powershell
python src/preprocessing.py
python src/eda.py
python src/create_targets.py
python src/train_models.py
```

Launch the application:

```powershell
streamlit run app/app.py
```

Launch MLflow separately:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

The raw dataset must be placed in `data/raw/india_housing_prices.csv` before running the workflow.

# 18. Limitations and Responsible Interpretation

This project is an analytical prototype. Its outputs should not be treated as financial advice.

## 18.1 Classification Limitations

The classification label is derived from relative price per square foot within a city.

It does not measure:

- Rental yield.
- Actual resale returns.
- Capital appreciation.
- Property liquidity.
- Legal title quality.
- Construction quality.
- Neighborhood development potential.
- Total ownership costs.

A below-median property price may still represent a poor investment.

## 18.2 Regression Limitations

The regression target is current property price.

The evaluation results show that XGBoost does not outperform the simple mean baseline.

The model should therefore not be presented as a reliable property valuation engine.

## 18.3 Future Valuation Limitations

The future-value calculation assumes a constant appreciation rate over the selected period.

It does not account for:

- Market downturns.
- Inflation.
- Interest rates.
- Property maintenance.
- Taxes.
- Brokerage and transaction expenses.
- Rental income.
- Financing costs.
- Location-specific market changes.

Actual property values may be higher or lower than the scenario result.

## 18.4 Dataset Limitations

The source dataset's provenance and collection methodology have not been independently verified.

The project findings should therefore be interpreted within the boundaries of the available dataset.

# 19. Future Enhancements

The following improvements would make the system more useful for real-world investment analysis.

1. Collect verified historical property transaction data.
2. Obtain actual time-series property price observations.
3. Build a genuine future-price forecasting model.
4. Incorporate rental yield and rental demand.
5. Include taxes, maintenance, and transaction costs.
6. Add location-based infrastructure and development indicators.
7. Evaluate models using time-based and location-grouped splits.
8. Add explainability using SHAP or comparable methods.
9. Introduce model monitoring and drift detection.
10. Improve automated testing and deployment.

# 20. Conclusion

The Real Estate Investment Advisor demonstrates an end-to-end data science workflow, from dataset inspection and preprocessing to exploratory analysis, model development, experiment tracking, and interactive application deployment.

The project successfully processes 250,000 residential property records, generates 20 analytical visualizations, trains and evaluates classification and regression models, records experiments through MLflow, and presents the results through a tested multi-page Streamlit application.

The classification experiment achieves 71.47% accuracy for a relative city-level price indicator. The regression experiment, however, does not outperform the mean baseline. The future valuation feature is therefore implemented transparently as a configurable appreciation scenario.

An important outcome of the project is not simply the construction of machine learning models, but the evaluation of whether the available data supports the intended predictions.

The results demonstrate the importance of target definition, data quality, baseline comparison, leakage prevention, and honest communication of model limitations.

The project provides a foundation for future development using verified real estate transaction and historical pricing data.

---

**End of Project Report**
