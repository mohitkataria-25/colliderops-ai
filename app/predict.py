import pandas as pd
from app import model_loader
from app import schemas
from io import StringIO


DEFAULT_MODEL = model_loader.get_active_model_name()
PREDICTION_FEATURE_COLUMNS = [

    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "PRI_tau_pt",
    "PRI_lep_pt",
]

EXPORT_COLUMNS = [
    "event_id",
    "prediction",
    "probability",
    "risk_level",
    "needs_review",
    "model_name",
    "model_version",
]

LABEL_MAPPING = {
    0: "background",
    1: "signal",
}
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

def read_prediction_csv(file:str)->pd.DataFrame:

    if file.endswith(".csv"):
        csv_df = pd.read_csv(file)
        return csv_df

    raise FileNotFoundError (f"The file {file} is not a valid format.")

def validate_prediction_dataframe(df:pd.DataFrame):

    missing_columns =  set(PREDICTION_FEATURE_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required prediction columns {missing_columns}")
        
def dataframe_to_prediction_payloads(df:pd.DataFrame)->list[dict]:    
    
    validate_prediction_dataframe(df=df)
    prediction_df = df[PREDICTION_FEATURE_COLUMNS].copy()


    return prediction_df.to_dict(orient="records")

def predict_events_from_dataframe(df:pd.DataFrame, model_to_load=DEFAULT_MODEL,) -> dict:
    
    payloads = dataframe_to_prediction_payloads(df=df)

    result = predict_batch_events(payloads=payloads,
                                  model_to_load=model_to_load,
                                  )
    
    if "event_id" in df.columns:
        for index, prediction in enumerate(result["predictions"]):
            prediction['event_id'] = str(df.iloc[index]["event_id"])
    
    return result

def predictions_to_dataframe(prediction_result:dict)-> pd.DataFrame:

    predictions = prediction_result.get("predictions", [])

    if not predictions:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    
    export_df = pd.DataFrame(predictions)

    for column in EXPORT_COLUMNS:
        if column not in export_df.columns:
            export_df[column] = None
    
    return export_df[EXPORT_COLUMNS]

def export_predictions_to_csv(prediction_result: dict) -> str:

    export_df = predictions_to_dataframe(prediction_result=prediction_result)

    csv_buffer = StringIO()
    export_df.to_csv(csv_buffer, index=False)

    return csv_buffer.getvalue()

def get_prediction_input(payload) -> dict:
    """Convert an incoming Pydantic payload or dictionary into a plain dict."""
    if hasattr(payload, "model_dump"):
        return payload.model_dump()

    return dict(payload)


def build_prediction_dataframe(input_data: dict) -> pd.DataFrame:
    """Convert input dictionary into a one-row DataFrame for model inference."""
    return pd.DataFrame([input_data])


def map_prediction_label(prediction_id: int) -> str:
    """Map numeric model output to a readable prediction label."""
    return LABEL_MAPPING.get(int(prediction_id), "unknown")


def get_prediction_probability(model, input_df: pd.DataFrame) -> float | None:
    """Return the highest class probability when the model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_df)[0]
    return round(float(max(probabilities)), 4)


def predict_single_event(
    payload: schemas.PredictionRequest | dict,
    model_to_load: str = DEFAULT_MODEL,
    loaded_model=None,
) -> dict:
    """Run model inference for a single collider event."""
    input_data = get_prediction_input(payload)

    model = loaded_model or model_loader.load_model(model_to_load=model_to_load)
    metadata = model_loader.get_model_metadata(model_to_load=model_to_load)

    input_df = build_prediction_dataframe(input_data)

    prediction_id = model.predict(input_df)[0]
    prediction_label = map_prediction_label(prediction_id)
    probability = get_prediction_probability(model, input_df)
    risk_level = get_risk_level(probability=probability)
    needs_review = should_review_prediction(probability=probability)

    return {
        "prediction": prediction_label,
        "probability": probability,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "risk_level": risk_level,
        "needs_review": needs_review,
    }   

def predict_batch_events(
        payloads: list[schemas.PredictionRequest | dict],
        model_to_load: str = DEFAULT_MODEL
)->dict:
    predictions = []

    model = model_loader.load_model(model_to_load=model_to_load)
    for payload in payloads:
        prediction = predict_single_event(
            payload=payload,
            model_to_load=model_to_load,
            loaded_model=model,
        )
        predictions.append(prediction)
    
    summary = summarize_batch_predictions(predictions=predictions)
    
    return {
        "predictions": predictions,
        "summary": summary,
    }

def get_risk_level(probability: float = None, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD)->str:

    if probability is None:
        return"unknown"
    
    elif probability >= threshold:
        return"high_confidence"
    
    else:
        return "low_confidence"
    
def should_review_prediction(probability: float = None, threshold : float = DEFAULT_CONFIDENCE_THRESHOLD)->bool:

    if probability is None:
        return True
    elif probability >= threshold:
        return False
    
    else:
        return True

def summarize_batch_predictions(predictions:list)-> dict:
    
    
    if not predictions:
        return {
            "total_events": 0,
            "signal_count": 0,
            "background_count": 0,
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "review_required_count": 0,
            "average_probability": None,
        }
    
    total_events = len(predictions)
    signal_count = sum(
        1 for prediction in predictions
        if prediction.get("prediction") == "signal"
    )
    background_count = sum(
        1 for prediction in predictions
        if prediction.get("prediction") == "background"
    )

    high_confidence_count = sum(
        1 for prediction in predictions
        if prediction.get("risk_level") == "high_confidence"
    )

    low_confidence_count = sum(
        1 for prediction in predictions
        if prediction.get("risk_level") == "low_confidence"
    )

    review_required_count = sum(
        1 for prediction in predictions
        if prediction.get("needs_review") is True
    )

    probabilities = [
        prediction.get("probability")
        for prediction in predictions
        if prediction.get("probability") is not None
    ]

    average_probability = None
    if probabilities:
        average_probability = round(sum(probabilities) / len(probabilities), 4)
    
    return {
        "total_events": total_events,
        "signal_count": signal_count,
        "background_count": background_count,
        "high_confidence_count": high_confidence_count,
        "low_confidence_count": low_confidence_count,
        "review_required_count": review_required_count,
        "average_probability": average_probability,
    }
