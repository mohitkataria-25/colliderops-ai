import pandas as pd


from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PARENT_DIR = Path(__file__).resolve().parents[1]
CURATED_DATA_DIR = PARENT_DIR / "data" / "curated" / "training_dataset"
CURATED_DATA_PATH = CURATED_DATA_DIR / "*.parquet"
FEATURE_COLUMNS = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "PRI_tau_pt",
    "PRI_lep_pt",
]


def read_curated_training_data()->pd.DataFrame:
    
    try:
        if CURATED_DATA_DIR.exists():
            df = pd.read_parquet(CURATED_DATA_DIR)
            print("CERN data extracted successfully.")
            return df
        else:
            raise ValueError (f"CERN data not available, validate if data is available at {CURATED_DATA_PATH}")
    except Exception as e:
        raise ValueError (f"CERN data extract failed with following error {e}")

def split_features_and_label(df: pd.DataFrame)->tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    #x = get_feature_columns(df)
    x = df[FEATURE_COLUMNS]

    label_mapping = {
        "background": 0,
        "signal": 1
    }

    y = df['label'].map(label_mapping)
    if y.isnull().any():
        invalid_labels = df.loc[y.isnull(), "label"].unique()
        raise ValueError(f"Invalid labels found: {invalid_labels}")

    x_train,x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

    print("Splitting cern dataset into training and testing samples...")
    return x_train, x_test, y_train, y_test

"""
def get_feature_columns(df:pd.DataFrame)->pd.DataFrame:

    print("Extracting predictors...")
    work_df = df.copy()
    x = work_df.drop(['label', 'event_id'], axis=1)
   
    return x
"""

def main ():

    print("Extracting CERN data from source csv")
    cern_df = read_curated_training_data()

    split_features_and_label(df=cern_df)

if __name__ == "__main__":
    main()

    