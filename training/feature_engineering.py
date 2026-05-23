import pandas as pd


from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PARENT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_MODE = "sample_collider"

SAMPLE_CURATED_DATA_DIR = PARENT_DIR / "data" / "curated" / "training_dataset"
SAMPLE_CURATED_DATA_PATH = SAMPLE_CURATED_DATA_DIR / "*.parquet"

REAL_CERN_CURATED_DATA_DIR = PARENT_DIR / "data" / "curated" / "real_cern_training_dataset"
REAL_CERN_CURATED_DATA_PATH = REAL_CERN_CURATED_DATA_DIR / "training_dataset.csv"

SAMPLE_FEATURE_COLUMNS = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "PRI_tau_pt",
    "PRI_lep_pt",
]

REAL_CERN_FEATURE_COLUMNS = [
    "gen_event_present",
    "gen_event_weight_count",
    "gen_event_signal_process_id",
    "gen_event_qscale",
    "gen_particles_present",
    "gen_particle_count",
    "ak5_genjets_present",
]

BOOLEAN_FEATURE_COLUMNS = [
    "gen_event_present",
    "gen_particles_present",
    "ak5_genjets_present",
]


def read_curated_training_data(dataset_mode: str = DEFAULT_DATASET_MODE) -> pd.DataFrame:
    """Read curated training data for the selected dataset mode."""
    try:
        if dataset_mode == "sample_collider":
            if SAMPLE_CURATED_DATA_DIR.exists():
                df = pd.read_parquet(SAMPLE_CURATED_DATA_DIR)
                print("Sample collider curated data extracted successfully.")
                return df

            raise ValueError(
                f"Sample collider curated data not available at {SAMPLE_CURATED_DATA_PATH}"
            )

        if dataset_mode == "real_cern":
            if REAL_CERN_CURATED_DATA_PATH.exists():
                df = pd.read_csv(REAL_CERN_CURATED_DATA_PATH)
                print("Real CERN curated data extracted successfully.")
                return df

            raise ValueError(
                f"Real CERN curated data not available at {REAL_CERN_CURATED_DATA_PATH}. "
                "Run python -m etl.real_cern_etl_job first."
            )

        raise ValueError(
            f"Unsupported dataset_mode={dataset_mode}. "
            "Supported values: sample_collider, real_cern"
        )

    except Exception as e:
        raise ValueError(f"Curated data extract failed with following error {e}")


def get_feature_columns(dataset_mode: str = DEFAULT_DATASET_MODE) -> list[str]:
    """Return feature columns for the selected dataset mode."""
    if dataset_mode == "sample_collider":
        return SAMPLE_FEATURE_COLUMNS

    if dataset_mode == "real_cern":
        return REAL_CERN_FEATURE_COLUMNS

    raise ValueError(
        f"Unsupported dataset_mode={dataset_mode}. "
        "Supported values: sample_collider, real_cern"
    )


def validate_required_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate that all required feature and label columns exist."""
    required_columns = set(feature_columns + ["label"])
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def normalize_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Normalize feature columns into model-ready numeric values."""
    x = df[feature_columns].copy()

    for column in BOOLEAN_FEATURE_COLUMNS:
        if column in x.columns:
            x[column] = x[column].astype(bool).astype(int)

    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    if x.isnull().any().any():
        null_columns = x.columns[x.isnull().any()].tolist()
        raise ValueError(f"Null or non-numeric values found in feature columns: {null_columns}")

    return x


# --- Inserted function: validate_model_ready_dataset ---
def validate_model_ready_dataset(
    df: pd.DataFrame,
    dataset_mode: str = DEFAULT_DATASET_MODE,
    require_two_classes: bool = False,
) -> dict:
    """Validate that a curated dataset is shaped correctly for ML workflows.

    This is useful for validating real CERN extracted data before a background
    class has been added. Training still requires two classes, but schema and
    feature validation can pass with a signal-only dataset.
    """
    feature_columns = get_feature_columns(dataset_mode=dataset_mode)
    validate_required_columns(df=df, feature_columns=feature_columns)
    x = normalize_features(df=df, feature_columns=feature_columns)

    label_counts = df["label"].value_counts().to_dict()
    unique_labels = sorted(df["label"].dropna().unique().tolist())

    if require_two_classes and len(unique_labels) < 2:
        raise ValueError(
            "Dataset validation requires at least two label classes, but only "
            f"found: {unique_labels}"
        )

    return {
        "dataset_mode": dataset_mode,
        "row_count": len(df),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "label_counts": label_counts,
        "unique_labels": unique_labels,
        "two_class_training_ready": len(unique_labels) >= 2,
        "schema_valid": True,
        "features_numeric": True,
        "null_feature_count": int(x.isnull().sum().sum()),
    }

def split_features_and_label(
    df: pd.DataFrame,
    dataset_mode: str = DEFAULT_DATASET_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split curated data into train/test feature and label sets."""
    feature_columns = get_feature_columns(dataset_mode=dataset_mode)
    validate_required_columns(df=df, feature_columns=feature_columns)

    x = normalize_features(df=df, feature_columns=feature_columns)

    label_mapping = {
        "background": 0,
        "signal": 1,
    }

    y = df["label"].map(label_mapping)
    if y.isnull().any():
        invalid_labels = df.loc[y.isnull(), "label"].unique()
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    if y.nunique() < 2:
        raise ValueError(
            "Training requires at least two label classes. "
            "The current dataset appears to contain only one class. "
            "For real_cern mode, add a background record before training a binary classifier."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=0.8,
        random_state=42,
        stratify=y,
    )

    print(f"Splitting {dataset_mode} dataset into training and testing samples...")
    return x_train, x_test, y_train, y_test

"""
def get_feature_columns(df:pd.DataFrame)->pd.DataFrame:

    print("Extracting predictors...")
    work_df = df.copy()
    x = work_df.drop(['label', 'event_id'], axis=1)
   
    return x
"""

def main():
    dataset_mode = DEFAULT_DATASET_MODE

    print(f"Extracting curated data for dataset_mode={dataset_mode}")
    cern_df = read_curated_training_data(dataset_mode=dataset_mode)

    validation_summary = validate_model_ready_dataset(
        df=cern_df,
        dataset_mode=dataset_mode,
        require_two_classes=False,
    )
    print("Dataset validation summary:")
    print(validation_summary)

    split_features_and_label(df=cern_df, dataset_mode=dataset_mode)

if __name__ == "__main__":
    main()

    