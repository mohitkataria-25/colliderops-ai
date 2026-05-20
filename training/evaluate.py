from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from training import feature_engineering
from training import logs
from pathlib import Path
import joblib
from datetime import datetime

PARENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PARENT_DIR / "models"
METRICS_DIR = PARENT_DIR / "evaluation_metrics"

def load_saved_models():
    
    logistic_model_path = MODEL_DIR / "logistic_regression_baseline.joblib"
    random_forest_model_path = MODEL_DIR / "random_forest_baseline.joblib"

    logistic_model = joblib.load(logistic_model_path)
    random_forest_model = joblib.load(random_forest_model_path)

    return logistic_model, random_forest_model

def load_test_data():
    
    df = feature_engineering.read_curated_training_data()
    x_train, x_test, y_train, y_test = feature_engineering.split_features_and_label(df=df)

    return x_test, y_test


def compare_performance(model, predictors,target, model_name):
    
    print(f"Generating predictions for model: {model_name}")
    prediction = model.predict(predictors)

    recall = recall_score(target, prediction, zero_division=0)
    precision = precision_score(target, prediction, zero_division=0)
    accuracy = accuracy_score(target, prediction)
    f1 = f1_score(target, prediction, zero_division=0)

    performance_df = pd.DataFrame(
        {   
            "model_name": model_name,
            "recall_score": recall,
            "precision_score": precision,
            "accuracy_score": accuracy,
            "f1_score": f1,
        },
        index=[0],
    )

    return performance_df

def main():


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved models...")
            
    logistic_model, random_forest_model = load_saved_models()
    print("Loading test data...")

    x_test, y_test = load_test_data()
    print("Generating performance metrics for logistic_model...")

    with logs.start_training_run(run_name="evaluate_baseline_models"):

            logs.log_training_params(
                {
                "evaluation_data_path": str(feature_engineering.CURATED_DATA_DIR),
                "feature_columns": ",".join(feature_engineering.FEATURE_COLUMNS),
                "models_evaluated": "logistic_regression_baseline,random_forest_baseline",
                "logistic_model_path": str(MODEL_DIR / "logistic_regression_baseline.joblib"),
                "random_forest_model_path": str(MODEL_DIR / "random_forest_baseline.joblib"),
                "test_rows": len(x_test),
                "metrics_output_path": str(METRICS_DIR),
            }        
        )
            logistic_model_metrics = compare_performance(model=logistic_model, 
                                                                    predictors=x_test, 
                                                                    target=y_test, 
                                                                    model_name="logistic_regression_baseline")
        

            random_forest_model_metrics = compare_performance(model=random_forest_model, 
                                                                    predictors=x_test, 
                                                                    target=y_test, 
                                                                    model_name="random_forest_baseline")        
            logs.log_training_metrics(
                {
                    "logistic_recall": logistic_model_metrics.loc[0, "recall_score"],
                    "logistic_precision": logistic_model_metrics.loc[0, "precision_score"],
                    "logistic_accuracy": logistic_model_metrics.loc[0, "accuracy_score"],
                    "logistic_f1": logistic_model_metrics.loc[0, "f1_score"],
                    "random_forest_recall": random_forest_model_metrics.loc[0, "recall_score"],
                    "random_forest_precision": random_forest_model_metrics.loc[0, "precision_score"],
                    "random_forest_accuracy": random_forest_model_metrics.loc[0, "accuracy_score"],
                    "random_forest_f1": random_forest_model_metrics.loc[0, "f1_score"],
            }        
        )

            comparison_df = pd.concat(
                [logistic_model_metrics, random_forest_model_metrics],
                ignore_index=True,
            )

            metrics_output_path = METRICS_DIR / f"model_comparison_{timestamp}.csv"
            metrics_output_path = str(metrics_output_path)
            comparison_df.to_csv(metrics_output_path, index=False)

            logs.log_file_artifact(
                file_path=metrics_output_path,
                artifact_path="evaluation_metrics"
                )

if __name__ == "__main__":
    main()
