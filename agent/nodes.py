"""LangGraph node functions for ColliderOpsAI agent workflows."""

from __future__ import annotations

from agent.state import AgentState
from agent.tools import (
    batch_export_tool,
    batch_file_prediction_tool,
    batch_prediction_tool,
    etl_status_tool,
    model_metadata_tool,
    rag_question_tool,
    single_prediction_tool,
)
from agent.router import route_intent


def route_intent_node(state: AgentState) -> AgentState:
    """Classify the user request and store the selected intent in state."""
    intent = route_intent(state)
    state["intent"] = intent
    return state


def rag_question_node(state: AgentState) -> AgentState:
    """Run the RAG question-answering tool and update state."""
    result = rag_question_tool(state)

    if result["status"] == "success":
        state["rag_answer"] = result.get("answer")
        state["retrieved_documents"] = result.get("sources", [])
        state["final_response"] = result.get("answer")
    else:
        _append_error(state, result.get("error"))
        state["final_response"] = result.get("error")

    return state


def model_metadata_node(state: AgentState) -> AgentState:
    """Load model metadata and update state."""
    result = model_metadata_tool(state)

    if result["status"] == "success":
        metadata = result.get("metadata")
        state["model_metadata"] = metadata
        state["final_response"] = _format_model_metadata_response(metadata)
    else:
        _append_error(state, result.get("error"))
        state["final_response"] = result.get("error")

    return state


def single_prediction_node(state: AgentState) -> AgentState:
    """Run single-event prediction and update state."""
    result = single_prediction_tool(state)

    if result["status"] == "success":
        prediction_result = result.get("prediction_result")
        state["prediction_result"] = prediction_result
        state["final_response"] = _format_prediction_response(prediction_result)
    else:
        _append_error(state, result.get("error"))
        state["prediction_result"] = None
        state["final_response"] = result.get("error")

    return state


def batch_prediction_node(state: AgentState) -> AgentState:
    """Run batch prediction from payloads and update state."""
    result = batch_prediction_tool(state)

    if result["status"] == "success":
        prediction_result = result.get("prediction_result")
        batch_summary = result.get("batch_summary")
        state["prediction_result"] = prediction_result
        state["batch_summary"] = batch_summary
        state["final_response"] = _format_batch_summary_response(batch_summary)
    else:
        _append_error(state, result.get("error"))
        state["prediction_result"] = None
        state["batch_summary"] = None
        state["final_response"] = result.get("error")

    return state


def batch_file_prediction_node(state: AgentState) -> AgentState:
    """Run batch prediction from an uploaded CSV file and update state."""
    result = batch_file_prediction_tool(state)

    if result["status"] == "success":
        prediction_result = result.get("prediction_result")
        batch_summary = result.get("batch_summary")
        state["prediction_result"] = prediction_result
        state["batch_summary"] = batch_summary
        state["final_response"] = _format_batch_summary_response(batch_summary)
    else:
        _append_error(state, result.get("error"))
        state["prediction_result"] = None
        state["batch_summary"] = None
        state["final_response"] = result.get("error")

    return state


def batch_export_node(state: AgentState) -> AgentState:
    """Run batch prediction export and update state."""
    result = batch_export_tool(state)

    if result["status"] == "success":
        prediction_result = result.get("prediction_result")
        state["prediction_result"] = prediction_result
        state["batch_summary"] = result.get("batch_summary")
        state["csv_output"] = result.get("csv_output")
        state["final_response"] = (
            "Prediction export generated successfully. "
            "CSV output is available in state['csv_output']."
        )
    else:
        _append_error(state, result.get("error"))
        state["final_response"] = result.get("error")

    return state


def etl_status_node(state: AgentState) -> AgentState:
    """Check ETL/data readiness and update state."""
    result = etl_status_tool(state)

    if result["status"] == "success":
        etl_status = result.get("etl_status")
        state["etl_status"] = etl_status
        state["final_response"] = _format_etl_status_response(etl_status)
    else:
        _append_error(state, result.get("error"))
        state["etl_status"] = None
        state["final_response"] = result.get("error")

    return state


def unknown_intent_node(state: AgentState) -> AgentState:
    """Handle requests that cannot be routed confidently."""
    state["final_response"] = (
        "I could not confidently route this request. Try asking about RAG documentation, "
        "model metadata, prediction, batch prediction, CSV export, or ETL status."
    )
    return state


def _append_error(state: AgentState, error: str | None) -> None:
    """Append an error message to state."""
    if not error:
        return

    errors = state.get("errors", [])
    errors.append(error)
    state["errors"] = errors


def _format_prediction_response(prediction_result: dict | None) -> str:
    """Create a concise text response for a single prediction."""
    if not prediction_result:
        return "Prediction could not be generated."

    return (
        f"Prediction: {prediction_result.get('prediction')} | "
        f"Probability: {prediction_result.get('probability')} | "
        f"Risk level: {prediction_result.get('risk_level')} | "
        f"Needs review: {prediction_result.get('needs_review')}"
    )


def _format_batch_summary_response(batch_summary: dict | None) -> str:
    """Create a concise text response for a batch prediction summary."""
    if not batch_summary:
        return "Batch prediction completed, but no summary was available."

    return (
        f"Batch prediction completed for {batch_summary.get('total_events')} events. "
        f"Signal: {batch_summary.get('signal_count')}, "
        f"Background: {batch_summary.get('background_count')}, "
        f"High confidence: {batch_summary.get('high_confidence_count')}, "
        f"Low confidence: {batch_summary.get('low_confidence_count')}, "
        f"Review required: {batch_summary.get('review_required_count')}, "
        f"Average probability: {batch_summary.get('average_probability')}."
    )


def _format_model_metadata_response(metadata: dict | None) -> str:
    """Create a concise text response for model metadata."""
    if not metadata:
        return "Model metadata is not available."

    return (
        f"Active model: {metadata.get('model_name')} "
        f"({metadata.get('model_file')}) | "
        f"Version: {metadata.get('model_version')} | "
        f"Type: {metadata.get('model_type')} | "
        f"Stage: {metadata.get('model_stage')} | "
        f"Confidence threshold: {metadata.get('confidence_threshold')}"
    )


def _format_etl_status_response(etl_status: dict | None) -> str:
    """Create a concise text response for ETL status."""
    if not etl_status:
        return "ETL status is not available."

    return (
        f"ETL status — raw data: {etl_status.get('raw_data_exists')}, "
        f"processed data: {etl_status.get('processed_data_exists')}, "
        f"curated data: {etl_status.get('curated_data_exists')}, "
        f"training ready: {etl_status.get('training_ready')}. "
        f"Next action: {etl_status.get('next_action')}"
    )