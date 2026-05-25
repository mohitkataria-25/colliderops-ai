import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from training import feature_engineering
from training import logs

# from xgboost import XGBClassifier

PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_DATASET_MODE = "sample_collider"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-mode",
        default=DEFAULT_DATASET_MODE,
        choices=["sample_collider", "real_cern", "curated_higgs"],
        help="Choose which curated dataset to train on.",
    )

    return parser.parse_args()


def resolve_training_data_path(dataset_mode: str) -> Path:
    """Return the curated training data path for the selected dataset mode."""
    if dataset_mode == "sample_collider":
        return feature_engineering.SAMPLE_CURATED_DATA_DIR

    if dataset_mode == "real_cern":
        return feature_engineering.REAL_CERN_CURATED_DATA_PATH

    if dataset_mode == "curated_higgs":
        return feature_engineering.CURATED_HIGGS_CURATED_DATA_PATH

    raise ValueError(
        f"Unsupported dataset_mode={dataset_mode}. "
        "Supported values: sample_collider, real_cern, curated_higgs"
    )


def load_training_data(
    dataset_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load, validate, and split curated training data."""
    try:
        df = feature_engineering.read_curated_training_data(
            dataset_mode=dataset_mode,
        )

        validation_summary = feature_engineering.validate_model_ready_dataset(
            df=df,
            dataset_mode=dataset_mode,
            require_two_classes=True,
        )

        x_train, x_test, y_train, y_test = feature_engineering.split_features_and_label(
            df=df,
            dataset_mode=dataset_mode,
        )

        return x_train, x_test, y_train, y_test, validation_summary

    except Exception as e:
        print(f"Training data load failed with the following error: {e}")
        raise


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train a Logistic Regression baseline model."""
    print("Starting training for Logistic Regression")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(X=x_train, y=y_train)
    return model


def train_randomforest_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest baseline model."""
    print("Starting training for Random Forest")

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
    )

    model.fit(X=x_train, y=y_train)
    return model

def train_hist_gradient_boosting_classifier(
  x_train: pd.DataFrame,
  y_train: pd.Series,      
)-> HistGradientBoostingClassifier:
    
    """ Train a Hist gradient boosting baseline model. """
    print("Starting training for Hist gradient boosting model.")
    model=HistGradientBoostingClassifier (
           random_state=RANDOM_STATE,
           learning_rate=0.1,
           max_iter=200
    )

    model.fit(X=x_train, y=y_train)
    return model

    

"""
def train_xgboost_classifier(x_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        random_state=1,
        verbosity=0,
    )
    model.fit(x_train, y_train)
    return model
"""


def save_model(model, model_name: str) -> Path:
    """Save a trained model locally and log it as an MLflow artifact."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(model, model_path)
    logs.log_model_artifact(model_path=model_path)

    print(f"Model saved at: {model_path}")
    return model_path


def log_training_metadata(
    dataset_mode: str,
    training_data_path: Path,
    feature_columns: list[str],
    validation_summary: dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Log training parameters and dataset metadata to MLflow."""
    logs.log_training_params(
        {
            "dataset_mode": dataset_mode,
            "training_data_path": str(training_data_path),
            "feature_columns": ",".join(feature_columns),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "models_trained": "logistic_regression, random_forest",
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "label_counts": str(validation_summary.get("label_counts")),
            "two_class_training_ready": validation_summary.get(
                "two_class_training_ready"
            ),
            "schema_valid": validation_summary.get("schema_valid"),
            "features_numeric": validation_summary.get("features_numeric"),
            "null_feature_count": validation_summary.get("null_feature_count"),
            "leakage_prone_columns_excluded": ",".join(
                feature_engineering.LEAKAGE_PRONE_COLUMNS
            ),
            "leakage_policy": "Excluded process identifiers from model feature set.",
        }
    )


def main() -> None:
    """Train baseline models for the selected dataset mode."""
    args = parse_args()
    dataset_mode = args.dataset_mode

    print(f"Loading training data for dataset_mode={dataset_mode}...")

    x_train, x_test, y_train, y_test, validation_summary = load_training_data(
        dataset_mode=dataset_mode,
    )

    feature_columns = feature_engineering.get_feature_columns(dataset_mode=dataset_mode)
    training_data_path = resolve_training_data_path(dataset_mode=dataset_mode)

    with logs.start_training_run(
        run_name=f"training_baseline_models_{dataset_mode}",
    ):
        log_training_metadata(
            dataset_mode=dataset_mode,
            training_data_path=training_data_path,
            feature_columns=feature_columns,
            validation_summary=validation_summary,
            x_train=x_train,
            x_test=x_test,
        )

        logistic_model = train_logistic_regression(
            x_train=x_train,
            y_train=y_train,
        )
        random_forest_model = train_randomforest_classifier(
            x_train=x_train,
            y_train=y_train,
        )

        # xgboost_model = train_xgboost_classifier(x_train=x_train, y_train=y_train)
        hist_Gradient_boosting_model = train_hist_gradient_boosting_classifier(
            x_train=x_train,
            y_train=y_train,
        )
        print(f"Base models generated, saving models at {MODEL_DIR}")

        save_model(logistic_model, f"logistic_regression_{dataset_mode}")
        save_model(random_forest_model, f"random_forest_{dataset_mode}")
        save_model(hist_Gradient_boosting_model, f"hist_gradient_boosting_{dataset_mode}")
        


if __name__ == "__main__":
    main()