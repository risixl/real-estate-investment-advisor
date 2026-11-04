# ============================================================
# REAL ESTATE INVESTMENT ADVISOR
# Classification + Regression + MLflow Experiment Tracking
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_ready_data.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

EXPERIMENT_NAME = "Real_Estate_Investment_Advisor"

# Use one consistent local MLflow database.
TRACKING_DB_PATH = ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB_PATH.as_posix()}"

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 65)
    print("LOADING REAL ESTATE DATASET")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = [
        "Good_Investment",
        "Target_Property_Price",
        "ID",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Scenario_Future_Price"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Good_Investment"].isna().any():
        raise ValueError(
            "Classification target contains missing values."
        )

    if df["Target_Property_Price"].isna().any():
        raise ValueError(
            "Regression target contains missing values."
        )

    print("Dataset shape:", df.shape)
    print("Total records:", len(df))

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nClassification distribution:")
    print(df["Good_Investment"].value_counts())

    return df


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

def prepare_features(df):

    # Exclude identifiers, targets and derived price variables.
    # This prevents direct target leakage.

    excluded_columns = [
        "ID",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Good_Investment",
        "Target_Property_Price",
        "Scenario_Future_Price"
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    ).copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_pipeline(X, model):

    categorical_columns = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# ============================================================
# 5. EXTRACT FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(pipeline, filename):

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:

        feature_names = preprocessor.get_feature_names_out()
        importance_values = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        output_path = REPORT_DIR / filename

        importance_df.to_csv(
            output_path,
            index=False
        )

        print(f"\nFeature importance saved: {output_path}")

        print("\nTop 10 important features:")
        print(
            importance_df.head(10).to_string(index=False)
        )

    except Exception as error:

        print(
            "Feature importance extraction warning:",
            error
        )


# ============================================================
# 6. CLASSIFICATION MODEL
# ============================================================

def train_classification(df):

    print("\n" + "=" * 65)
    print("TRAINING INVESTMENT CLASSIFICATION MODEL")
    print("=" * 65)

    X = prepare_features(df)
    y = df["Good_Investment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Input features:", X.shape[1])

    classifier = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(X_train, classifier)

    print("\nFitting classification model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        ).tolist()
    }

    print("\nCLASSIFICATION RESULTS")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(
        "\nAverage predicted investment probability:",
        float(probabilities.mean())
    )

    model_path = MODEL_DIR / "classifier.pkl"

    joblib.dump(pipeline, model_path)

    print("\nClassification model saved:", model_path)

    cm = confusion_matrix(y_test, predictions)

    pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    ).to_csv(
        REPORT_DIR / "classification_confusion_matrix.csv"
    )

    save_feature_importance(
        pipeline,
        "classification_feature_importance.csv"
    )

    return pipeline, metrics


# ============================================================
# 7. REGRESSION MODEL AND BASELINE
# ============================================================

def train_regression(df):

    print("\n" + "=" * 65)
    print("TRAINING PROPERTY VALUATION REGRESSION MODEL")
    print("=" * 65)

    X = prepare_features(df)
    y = df["Target_Property_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Input features:", X.shape[1])

    # --------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------

    print("\nTraining baseline regression model...")

    baseline = DummyRegressor(strategy="mean")

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    baseline_metrics = {
        "mae": float(
            mean_absolute_error(
                y_test,
                baseline_predictions
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    baseline_predictions
                )
            )
        ),
        "r2": float(
            r2_score(
                y_test,
                baseline_predictions
            )
        )
    }

    print("\nBASELINE REGRESSION RESULTS")

    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value}")

    # Save baseline model for reproducibility.
    baseline_path = MODEL_DIR / "baseline_regressor.pkl"

    joblib.dump(baseline, baseline_path)

    print("\nBaseline model saved:", baseline_path)

    # --------------------------------------------------------
    # XGBoost regression model
    # --------------------------------------------------------

    regressor = XGBRegressor(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(X_train, regressor)

    print("\nFitting XGBoost regression model...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_test, predictions)
            )
        ),
        "r2": float(
            r2_score(y_test, predictions)
        )
    }

    print("\nXGBOOST REGRESSION RESULTS")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    model_path = MODEL_DIR / "regressor.pkl"

    joblib.dump(pipeline, model_path)

    print("\nRegression model saved:", model_path)

    comparison = pd.DataFrame([
        {
            "Model": "Dummy Regressor",
            **baseline_metrics
        },
        {
            "Model": "XGBoost Regressor",
            **metrics
        }
    ])

    comparison.to_csv(
        REPORT_DIR / "regression_comparison.csv",
        index=False
    )

    print("\nREGRESSION MODEL COMPARISON")
    print(comparison.to_string(index=False))

    save_feature_importance(
        pipeline,
        "regression_feature_importance.csv"
    )

    return pipeline, metrics, baseline_metrics


# ============================================================
# 8. SAVE EVALUATION RESULTS
# ============================================================

def save_metrics(
    classification_metrics,
    regression_metrics,
    baseline_metrics
):

    results = {
        "classification": classification_metrics,
        "regression": regression_metrics,
        "regression_baseline": baseline_metrics
    }

    output_path = REPORT_DIR / "model_metrics.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nEvaluation metrics saved:", output_path)

    return output_path


# ============================================================
# 9. MLFLOW EXPERIMENT TRACKING
# ============================================================

def log_mlflow_experiments(
    classifier,
    regressor,
    classification_metrics,
    regression_metrics,
    baseline_metrics
):

    print("\n" + "=" * 65)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 65)

    # Explicitly configure the tracking database.
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # --------------------------------------------------------
    # Classification experiment
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="XGBoost_Classification"
    ):

        mlflow.log_params({
            "algorithm": "XGBoostClassifier",
            "n_estimators": 120,
            "max_depth": 6,
            "learning_rate": 0.08,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "target": "Good_Investment"
        })

        mlflow.log_metrics({
            "accuracy": classification_metrics["accuracy"],
            "precision": classification_metrics["precision"],
            "recall": classification_metrics["recall"],
            "f1_score": classification_metrics["f1_score"]
        })

        mlflow.set_tag(
            "target_definition",
            "Below-city-median price proxy"
        )

        try:

            mlflow.sklearn.log_model(
                sk_model=classifier,
                name="classification_model",
                serialization_format="cloudpickle"
            )

            mlflow.set_tag(
                "model_logging_status",
                "success"
            )

            print("Classification model logged successfully.")

        except Exception as error:

            mlflow.set_tag(
                "model_logging_status",
                "failed"
            )

            print(
                "Classification model logging failed:",
                error
            )

        mlflow.log_artifact(
            str(
                REPORT_DIR /
                "classification_feature_importance.csv"
            )
        )

        mlflow.log_artifact(
            str(
                REPORT_DIR /
                "classification_confusion_matrix.csv"
            )
        )

    # --------------------------------------------------------
    # Regression experiment
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="XGBoost_Regression"
    ):

        mlflow.log_params({
            "algorithm": "XGBoostRegressor",
            "n_estimators": 120,
            "max_depth": 6,
            "learning_rate": 0.08,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "target": "Target_Property_Price"
        })

        mlflow.log_metrics({
            "mae": regression_metrics["mae"],
            "rmse": regression_metrics["rmse"],
            "r2": regression_metrics["r2"],
            "baseline_mae": baseline_metrics["mae"],
            "baseline_rmse": baseline_metrics["rmse"],
            "baseline_r2": baseline_metrics["r2"]
        })

        mlflow.set_tag(
            "target_definition",
            "Current property price in lakhs"
        )

        try:

            mlflow.sklearn.log_model(
                sk_model=regressor,
                name="regression_model",
                serialization_format="cloudpickle"
            )

            mlflow.set_tag(
                "model_logging_status",
                "success"
            )

            print("Regression model logged successfully.")

        except Exception as error:

            mlflow.set_tag(
                "model_logging_status",
                "failed"
            )

            print(
                "Regression model logging failed:",
                error
            )

        mlflow.log_artifact(
            str(
                REPORT_DIR /
                "regression_feature_importance.csv"
            )
        )

        mlflow.log_artifact(
            str(
                REPORT_DIR /
                "regression_comparison.csv"
            )
        )

    print("\nMLflow experiment tracking completed.")


# ============================================================
# 10. SAVE PORTABLE MODEL METADATA
# ============================================================

def save_model_metadata(df):

    metadata = {
        "project": "Real Estate Investment Advisor",
        "classification_model": "XGBoost Classifier",
        "regression_model": "XGBoost Regressor",
        "classification_target": "Good_Investment",
        "regression_target": "Target_Property_Price",
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "models_directory": "models",
        "mlflow_experiment": EXPERIMENT_NAME,
        "mlflow_tracking_uri": "sqlite:///mlflow.db",
        "classification_target_note": (
            "Below-city-median price proxy, "
            "not verified investment profitability"
        ),
        "regression_target_note": (
            "Current property price, not future property value"
        ),
        "future_valuation_note": (
            "Illustrative appreciation scenario, "
            "not an ML prediction"
        ),
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nModel metadata saved:", metadata_path)


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("REAL ESTATE INVESTMENT ADVISOR")
    print("MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 65)

    # Configure MLflow before any experiment operations.
    mlflow.set_tracking_uri(TRACKING_URI)

    print("\nMLflow tracking URI:", mlflow.get_tracking_uri())

    # Load data.
    df = load_dataset()

    # Train classification model.
    classifier, classification_metrics = train_classification(df)

    # Train regression and baseline models.
    regressor, regression_metrics, baseline_metrics = (
        train_regression(df)
    )

    # Save evaluation metrics.
    save_metrics(
        classification_metrics,
        regression_metrics,
        baseline_metrics
    )

    # Save portable metadata.
    save_model_metadata(df)

    # Log MLflow experiments.
    log_mlflow_experiments(
        classifier,
        regressor,
        classification_metrics,
        regression_metrics,
        baseline_metrics
    )

    print("\n")
    print("=" * 65)
    print("MODEL TRAINING PIPELINE COMPLETED")
    print("=" * 65)

    print("\nSaved model files:")
    print("- models/classifier.pkl")
    print("- models/regressor.pkl")
    print("- models/baseline_regressor.pkl")
    print("- models/model_metadata.json")

    print("\nSaved evaluation files:")
    print("- reports/metrics/model_metrics.json")
    print("- reports/metrics/regression_comparison.csv")
    print("- reports/metrics/classification_confusion_matrix.csv")
    print("- reports/metrics/classification_feature_importance.csv")
    print("- reports/metrics/regression_feature_importance.csv")


if __name__ == "__main__":
    main()