"""Intent routing logic for ColliderOpsAI LangGraph workflows."""

from __future__ import annotations

from agent.state import AgentState


RAG_KEYWORDS = [
    "what is",
    "explain",
    "documentation",
    "docs",
    "model card",
    "evaluation card",
    "data dictionary",
    "how does",
    "why",
    "rag",
    "research",
]

MODEL_METADATA_KEYWORDS = [
    "active model",
    "current model",
    "currently active",
    "model currently active",
    "which model",
    "served model",
    "serving model",
    "model metadata",
    "model version",
    "model type",
    "model name",
    "confidence threshold",
    "features",
    "feature columns",
    "mlflow",
    "artifact",
]

SINGLE_PREDICTION_KEYWORDS = [
    "classify this event",
    "predict this event",
    "single prediction",
    "one event",
    "signal or background",
]

BATCH_PREDICTION_KEYWORDS = [
    "batch prediction",
    "batch predict",
    "multiple events",
    "predict these events",
]

BATCH_FILE_KEYWORDS = [
    "uploaded csv",
    "csv file",
    "file prediction",
    "predict file",
    "run prediction on file",
    "batch file",
]

BATCH_EXPORT_KEYWORDS = [
    "export",
    "download",
    "csv output",
    "prediction export",
    "downloadable csv",
]

ETL_STATUS_KEYWORDS = [
    "etl",
    "data pipeline",
    "raw data",
    "processed data",
    "curated data",
    "training ready",
    "pipeline status",
    "can i train",
]


def _contains_any(query: str, keywords: list[str]) -> bool:
    """Return True when query contains any known keyword."""
    return any(keyword in query for keyword in keywords)


def route_intent(state: AgentState) -> str:
    """Route the user request to the correct LangGraph node."""
    user_query = state.get("user_query", "")

    if not user_query:
        return "unknown"

    query = user_query.lower().strip()

    if state.get("prediction_payload"):
        return "single_prediction"

    if state.get("batch_prediction_payloads"):
        return "batch_prediction"

    if state.get("uploaded_file_path") and _contains_any(query, BATCH_EXPORT_KEYWORDS):
        return "batch_export"

    if state.get("uploaded_file_path"):
        return "batch_file_prediction"

    if _contains_any(query, ETL_STATUS_KEYWORDS):
        return "etl_status"

    if _contains_any(query, MODEL_METADATA_KEYWORDS):
        return "model_metadata"

    if _contains_any(query, SINGLE_PREDICTION_KEYWORDS):
        return "single_prediction"

    if _contains_any(query, BATCH_EXPORT_KEYWORDS):
        return "batch_export"

    if _contains_any(query, BATCH_FILE_KEYWORDS):
        return "batch_file_prediction"

    if _contains_any(query, BATCH_PREDICTION_KEYWORDS):
        return "batch_prediction"

    if _contains_any(query, RAG_KEYWORDS):
        return "rag_question"

    return "rag_question"