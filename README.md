# Real Estate Investment Advisor
## Predicting Property Investment Indicators and Exploring Future Property Value

An end-to-end machine learning and analytics project built using Python, Pandas, Scikit-learn, XGBoost, Streamlit, and MLflow.

The application explores residential property data across India, provides property-level analysis, estimates a below-city-median price indicator, and calculates illustrative five-year property valuations using user-defined appreciation assumptions.

> **Important methodology note:** The dataset does not contain verified investment returns or actual future property prices. The classification output is a price-position proxy, the regression model estimates current property prices, and future valuation is an assumption-based financial scenario. None of these should be interpreted as a guaranteed investment recommendation.

## Project Objectives

- Analyse residential property characteristics and pricing patterns.
- Assess and clean a dataset containing 250,000 property records.
- Explore relationships between property attributes and prices.
- Build a classification model using a city-level median price-per-square-foot indicator.
- Evaluate whether available features can predict current property prices.
- Provide an interactive Streamlit application.
- Track model experiments using MLflow.
- Calculate illustrative future property values under configurable appreciation assumptions.

## Key Features

### 1. Property Analysis
Explore property characteristics, location information, pricing distributions, and other available attributes.

### 2. Investment Indicator
Classify whether a property's price per square foot is at or below its city's median price per square foot.

This is a relative pricing indicator, not a measure of verified profitability or investment quality.

### 3. Future Valuation
Calculate a scenario-based future property value using compound appreciation:

`Future Value = Current Value × (1 + Annual Appreciation Rate) ^ Number of Years`

The default illustrative scenario uses 8% annual appreciation over five years. Users can change the assumptions in the application.

### 4. Model Performance
Review classification and regression evaluation results, including confusion matrix and regression comparison metrics.

### 5. MLflow Experiments
Inspect the recorded model experiments, parameters, metrics, and logged artifacts.

## Dataset

The project uses `india_housing_prices.csv`, containing:

- 250,000 residential property records
- 23 original attributes
- Location, property type, size, pricing, construction year, amenities, and availability information

The raw dataset is not included in this repository by default. Place the source CSV at:

`data/raw/india_housing_prices.csv`

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, evaluation, and baseline models |
| XGBoost | Classification and regression |
| Streamlit | Interactive web application |
| Plotly | Interactive visualizations |
| MLflow | Experiment tracking |
| Joblib | Model serialization |
| SQLite | MLflow tracking backend |

## Project Structure

```text
real_estate_investment_advisor/
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Property_Analysis.py
│   │   ├── 2_Investment_Analysis.py
│   │   ├── 3_Future_Valuation.py
│   │   ├── 4_Model_Performance.py
│   │   └── 5_MLflow_Experiments.py
│   └── utils/
│       ├── data_loader.py
│       ├── helpers.py
│       └── model_loader.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── PROJECT_REPORT.md
│
├── models/
│   └── model_metadata.json
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── create_targets.py
│   ├── train_models.py
│   ├── check_dataset_quality.py
│   ├── audit_filtering.py
│   ├── diagnose_regression.py
│   └── inspect_data.py
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets, model binaries, virtual environments, and MLflow database files may be excluded from Git. Recreate them using the project scripts.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd real_estate_investment_advisor
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add the dataset

Place `india_housing_prices.csv` inside:

```text
data/raw/
```

## Run the Project

Run the scripts from the project root directory.

### Step 1: Preprocess the dataset

```powershell
python src/preprocessing.py
```

Expected output:

```text
data/processed/cleaned_housing_data.csv
```

### Step 2: Generate exploratory analysis

```powershell
python src/eda.py
```

Generated charts are saved under:

```text
reports/figures/
```

### Step 3: Create model targets

```powershell
python src/create_targets.py
```

Expected output:

```text
data/processed/model_ready_data.csv
```

### Step 4: Train and evaluate models

```powershell
python src/train_models.py
```

This creates trained model artifacts, evaluation metrics, feature importance reports, and MLflow experiment records.

### Step 5: Launch the Streamlit application

```powershell
streamlit run app/app.py
```

The application will open in your browser. Streamlit normally provides a local address such as:

```text
http://localhost:8501
```

### Step 6: View MLflow experiments

In a separate terminal, activate the environment and run:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## Model Results

The recorded evaluation results are:

### Classification

| Metric | Result |
|---|---:|
| Accuracy | 71.47% |
| Precision | 71.23% |
| Recall | 72.03% |
| F1-score | 71.63% |

The classifier predicts the below-city-median price indicator. It does not establish whether a property will generate a profit.

### Regression

| Model | MAE (₹ lakh) | RMSE (₹ lakh) | R² |
|---|---:|---:|---:|
| Mean baseline | 122.318 | 141.193 | Approximately 0.000 |
| XGBoost | 122.365 | 141.286 | -0.0013 |

The XGBoost regressor slightly underperforms the mean baseline. The available dataset features therefore do not provide useful predictive power for the current-price target in this experiment.

A negative R² means the model performs worse than predicting the test-set target mean.

## Important Limitations

- The classification label is derived from the city's median price per square foot. It is not based on rental yield, resale returns, or verified investment outcomes.
- The regression target is current property price, not future property price.
- The five-year valuation is a mathematical scenario based on an assumed appreciation rate, not a trained forecasting model.
- The scenario excludes transaction costs, taxes, maintenance, financing costs, inflation, and market volatility.
- Dataset provenance and collection methodology have not been independently verified.
- Model performance on this dataset should not be generalized to the entire Indian real estate market.

## Future Improvements

- Obtain verified historical property transactions and time-series price data.
- Develop a future-price forecasting model using actual historical observations.
- Include rental yield, maintenance, taxes, transaction costs, and financing in investment analysis.
- Validate performance using time-based and location-grouped evaluation.
- Improve geographic coverage and data provenance.
- Add model monitoring and prediction explainability.

## Author

Developed as an end-to-end data analytics and machine learning project focused on Indian residential real estate.

