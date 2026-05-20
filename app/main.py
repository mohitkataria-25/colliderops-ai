import pandas as pd
from io import StringIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import (
                        HealthResponse, 
                        PredictionRequest, 
                        PredictionResponse, 
                        BatchPredictionResponse, 
                        AskRequest, 
                        AskResponse
                        )


from app.predict import (
    export_predictions_to_csv,
    predict_batch_events,
    predict_events_from_dataframe,
    predict_single_event,
)

from rag.pipeline import run_pipeline, initialize_rag_pipeline

rag_state = initialize_rag_pipeline()

def create_app()->FastAPI:

    """ Create and confirm FastApi application"""
    app = FastAPI(
        title = "ColliderOpsAI",
        description = "ML inference API for CERN-style collider event classification",
        version = "0.1.0",
    )

    return app

app = create_app()

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request:PredictionRequest):
    return predict_single_event(payload=request)

@app.post("/batch-predict", response_model=BatchPredictionResponse)
def predict_batch(requests:list[PredictionRequest]):
    return predict_batch_events(payloads=requests)

@app.post("/ask", response_model=AskResponse)
def ask_question(request:AskRequest):
    result = run_pipeline(question=request.question, pipeline_state=rag_state)
    return {
        "question":result['question'],
        "answer": result['answer'],
        "sources": result['sources'],
    }

@app.post("/batch-predict-file", response_model=BatchPredictionResponse)
def predict_batch_file(file:UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only csv files are supported for batch prediction."
        )
    try:
        df = pd.read_csv(file.file)
        return predict_events_from_dataframe(df=df)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(error)}") from error
    
@app.post("/batch-predict-file-export")
def predict_batch_file_export(file: UploadFile = File(...)):
    """Run batch prediction from uploaded CSV and return prediction results as downloadable CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only csv files are supported for batch prediction export.",
        )

    try:
        df = pd.read_csv(file.file)
        prediction_result = predict_events_from_dataframe(df=df)
        csv_output = export_predictions_to_csv(prediction_result=prediction_result)

        return StreamingResponse(
            StringIO(csv_output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=colliderops_predictions.csv"
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export prediction results: {str(error)}",
        ) from error
