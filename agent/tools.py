from rag.pipeline import initialize_rag_pipeline, run_pipeline
from agent.state import AgentState

from app.model_loader import get_model_metadata
from app.predict import predict_single_event as app_predict_single_event
from app.predict import predict_batch_events as app_predict_batch_events
from app.predict import predict_events_from_dataframe, export_predictions_to_csv

from pathlib import Path
import pandas as pd

_RAG_STATE = None

def get_rag_state():
     
    global _RAG_STATE

    if _RAG_STATE is None:
         _RAG_STATE = initialize_rag_pipeline()
        
    return _RAG_STATE

def rag_question_tool(state:AgentState)->dict:

    
    question = state.get("user_query")

    if not question:
        return {
            "tool_name": "rag_question_tool",
            "question": None,
            "answer": None,
            "sources": [],
            "status": "error",
            "error": "Missing user_query in agent state.",
        }
    try:

        rag_state = get_rag_state()
        result = run_pipeline(question=question, pipeline_state=rag_state)
        
        return {
            "tool_name": "rag_question_tool",
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "status": "success",
            "error":None
        }
    except Exception as e:
        return{
            "tool_name": "rag_question_tool",
            "question": question,
            "answer": None,
            "sources": [],
            "status": "error",
            "error": str(e),
        }
    
def model_metadata_tool(state:AgentState)->dict:
        
    try:
        metadata = get_model_metadata()
        return {
            "tool_name": "model_metadata_tool",
            "metadata": metadata,
            "status": "success",
            "error": None,
        }

    except Exception as e:
        return {
            "tool_name": "model_metadata_tool",
            "metadata": None,
            "status": "error",
            "error": str(e),
        }

def single_prediction_tool(state:AgentState)->dict:

    prediction_payload = state.get("prediction_payload")

    if not prediction_payload:
        return {
            "tool_name": "single_prediction_tool",
            "prediction_payload": None,
            "prediction_result":None,
            "status": "error",
            "error":"Prediction payload is missing."
        }
    
    try:
        
        prediction_details = app_predict_single_event(payload=prediction_payload)

        return {
            "tool_name": "single_prediction_tool",
            "prediction_payload": prediction_payload,
            "prediction_result": prediction_details,
            "status": "success",
            "error": None,
        }

    except Exception as e:
       
        return {
            "tool_name": "single_prediction_tool",
            "prediction_payload": prediction_payload,
            "prediction_result":None,
            "status":"error",
            "error":str(e)
    }

def batch_prediction_tool(state:AgentState)->dict:

        batch_prediction_payloads = state.get("batch_prediction_payloads")

        if not batch_prediction_payloads:
            return{
                "tool_name": "batch_prediction_tool",
                "batch_prediction_payloads": None,
                "prediction_result":None,
                "batch_summary": None,
                "status": "error",
                "error":"Batch Prediction payload is missing."
            }

        try:
            prediction_results = app_predict_batch_events(payloads=batch_prediction_payloads)

            return{
                "tool_name":"batch_prediction_tool",
                "batch_prediction_payloads": batch_prediction_payloads,
                "prediction_result":prediction_results,
                "batch_summary": prediction_results.get("summary"),
                "status":"success",
                "error":None,
            }
        
        except Exception as e:
            return{
                "tool_name":"batch_prediction_tool",
                "batch_prediction_payloads": batch_prediction_payloads,
                "prediction_result":None,
                "batch_summary": None,
                "status":"error",
                "error":str(e),
            }

def batch_file_prediction_tool(state:AgentState)->dict:
    
    csv_file_path = state.get("uploaded_file_path")

    if not csv_file_path:
        return{
            "tool_name": "batch_file_prediction_tool",
            "uploaded_file_path": None,
            "prediction_result": None,
            "batch_summary": None,
            "status": "error",
            "error": "Uploaded file path is missing.",
        }
    
    try:
        df = pd.read_csv(csv_file_path)
        prediction_result = predict_events_from_dataframe(df=df)

        return {
            "tool_name": "batch_file_prediction_tool",
            "uploaded_file_path": csv_file_path,
            "prediction_result": prediction_result,
            "batch_summary": prediction_result.get("summary"),
            "status": "success",
            "error": None,
        }
    
    except Exception as e:

        return{
            "tool_name": "batch_file_prediction_tool",
            "uploaded_file_path": csv_file_path,
            "prediction_result": None,
            "batch_summary": None,
            "status": "error",
            "error": str(e),
        }


def batch_export_tool(state: AgentState) -> dict:
    csv_file_path = state.get("uploaded_file_path")

    if not csv_file_path:
        return {
            "tool_name": "batch_export_tool",
            "uploaded_file_path": None,
            "csv_output": None,
            "prediction_result": None,
            "batch_summary": None,
            "status": "error",
            "error": "Uploaded file path is missing.",
        }

    try:
        df = pd.read_csv(csv_file_path)
        prediction_result = predict_events_from_dataframe(df=df)
        csv_output = export_predictions_to_csv(prediction_result=prediction_result)

        return {
            "tool_name": "batch_export_tool",
            "uploaded_file_path": csv_file_path,
            "csv_output": csv_output,
            "prediction_result": prediction_result,
            "batch_summary": prediction_result.get("summary"),
            "status": "success",
            "error": None,
        }

    except Exception as e:
        return {
            "tool_name": "batch_export_tool",
            "uploaded_file_path": csv_file_path,
            "csv_output": None,
            "prediction_result": None,
            "batch_summary": None,
            "status": "error",
            "error": str(e),
        }

def etl_status_tool(state: AgentState) -> dict:
    try:
        project_root = Path(__file__).resolve().parents[1]

        raw_data_path = project_root / "data" / "raw" / "collider_events.csv"
        processed_output_path = project_root / "data" / "processed" / "collider_events"
        curated_output_path = project_root / "data" / "curated" / "training_dataset"

        raw_data_exists = raw_data_path.exists()
        processed_data_exists = processed_output_path.exists()
        curated_data_exists = curated_output_path.exists()
        training_ready = curated_data_exists

        if not raw_data_exists:
            next_action = (
                "Raw data is missing. Add a raw dataset or run "
                "python -m etl.download_cern_data."
            )
        elif not processed_data_exists or not curated_data_exists:
            next_action = (
                "Raw data exists, but processed/curated outputs are missing. "
                "Run python -m etl.glue_etl_job."
            )
        else:
            next_action = (
                "Curated data is available. You can run training, evaluation, "
                "or batch prediction workflows."
            )

        return {
            "tool_name": "etl_status_tool",
            "etl_status": {
                "raw_data_exists": raw_data_exists,
                "processed_data_exists": processed_data_exists,
                "curated_data_exists": curated_data_exists,
                "training_ready": training_ready,
                "raw_data_path": str(raw_data_path),
                "processed_output_path": str(processed_output_path),
                "curated_output_path": str(curated_output_path),
                "next_action": next_action,
            },
            "status": "success",
            "error": None,
        }

    except Exception as e:
        return {
            "tool_name": "etl_status_tool",
            "etl_status": None,
            "status": "error",
            "error": str(e),
        }
    