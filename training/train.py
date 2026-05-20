from training import feature_engineering
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from training import logs

#from xgboost import XGBClassifier

PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_training_data()->tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    try:
        df = feature_engineering.read_curated_training_data()
        x_train, x_test, y_train, y_test = feature_engineering.split_features_and_label(df=df)
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        print(f"Training data load failed with the following error: {e}")
        raise

def train_logistic_regression(x_train, y_train)->LogisticRegression:
    
    print("Starting training for Logistic Regression ")
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    model.fit(X=x_train, y=y_train)
    return model


def train_randomforest_classifier(x_train, y_train)->RandomForestClassifier:
    
    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X=x_train, y=y_train)
    return model

"""
def train_xgboost_classifier (x_train, y_train)-> XGBClassifier:
    model = XGBClassifier(
        random_state=1,
        verbosity=0
    )
    model.fit(x_train, y_train)
    return model
"""

def save_model(model, model_name: str)->Path:
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logs.log_model_artifact(model_path=model_path)   
    print(f"Model saved at: {model_path}")
    return model_path

def main():

    
    print("Loading training data...")

    x_train, x_test, y_train, y_test = load_training_data()
    
    

    with logs.start_training_run(run_name="training_baseline_models"):
        logs.log_training_params(
            {
                "training_data_path": str(feature_engineering.CURATED_DATA_DIR),
                "feature_columns": ",".join(feature_engineering.FEATURE_COLUMNS),
                "test_size":TEST_SIZE,
                "random_state":RANDOM_STATE,
                "models_trained":"logistic_regression_forest, random_forest",
                "train_rows":len(x_train),
                "test_rows":len(x_test),
            }
        )

        logistic_model = train_logistic_regression(
            x_train=x_train, 
            y_train=y_train,
        )
        random_forest_model = train_randomforest_classifier(
            x_train=x_train, 
            y_train=y_train,
        )
        #xgboost_model = train_xgboost_classifier(x_train=x_train, y_train=y_train)
        
        print(f"Base Models generated, saving models at {MODEL_DIR}")

        save_model(logistic_model, "logistic_regression_baseline")
        save_model(random_forest_model, "random_forest_baseline")
        #save_model(xgboost_model, "xgboost_baseline")

if __name__ == "__main__":
    main()