"""Smoke tests for ColliderOpsAI LangGraph agent workflows."""

from __future__ import annotations

from agent.graph import run_agent


SAMPLE_PAYLOAD = {
    "DER_mass_MMC": 138.4,
    "DER_mass_transverse_met_lep": 51.6,
    "DER_mass_vis": 97.8,
    "PRI_tau_pt": 32.6,
    "PRI_lep_pt": 44.1,
}


BATCH_PAYLOADS = [
    SAMPLE_PAYLOAD,
    {
        "DER_mass_MMC": 80.1,
        "DER_mass_transverse_met_lep": 88.4,
        "DER_mass_vis": 58.3,
        "PRI_tau_pt": 18.4,
        "PRI_lep_pt": 24.1,
    },
]


def test_agent_routes_to_model_metadata():
    result = run_agent(
        {
            "user_query": "What model is currently active?",
            "errors": [],
        }
    )

    assert result.get("intent") == "model_metadata"
    assert "Active model" in result.get("final_response", "")
    assert result.get("model_metadata") is not None


def test_agent_routes_to_etl_status():
    result = run_agent(
        {
            "user_query": "Is my curated data ready for training?",
            "errors": [],
        }
    )

    assert result.get("intent") == "etl_status"
    assert "ETL status" in result.get("final_response", "")
    assert result.get("etl_status") is not None


def test_agent_routes_to_single_prediction_when_payload_exists():
    result = run_agent(
        {
            "user_query": "Classify this event.",
            "prediction_payload": SAMPLE_PAYLOAD,
            "errors": [],
        }
    )

    prediction_result = result.get("prediction_result")

    assert result.get("intent") == "single_prediction"
    assert prediction_result is not None
    assert prediction_result.get("prediction") in ["signal", "background"]
    assert "Prediction:" in result.get("final_response", "")


def test_agent_routes_to_batch_prediction_when_batch_payloads_exist():
    result = run_agent(
        {
            "user_query": "Run batch prediction on these events.",
            "batch_prediction_payloads": BATCH_PAYLOADS,
            "errors": [],
        }
    )

    prediction_result = result.get("prediction_result")
    batch_summary = result.get("batch_summary")

    assert result.get("intent") == "batch_prediction"
    assert prediction_result is not None
    assert batch_summary is not None
    assert batch_summary.get("total_events") == 2
    assert "Batch prediction completed" in result.get("final_response", "")


def test_agent_rag_route_can_be_mocked(monkeypatch):
    from agent import nodes

    def mock_rag_question_tool(state):
        return {
            "tool_name": "rag_question_tool",
            "question": state.get("user_query"),
            "answer": "ColliderOpsAI is an AI-assisted research workbench.",
            "sources": [
                {
                    "file_name": "research_workbench_overview.md",
                    "source": "docs/research_workbench_overview.md",
                    "document_type": "md",
                }
            ],
            "status": "success",
            "error": None,
        }

    monkeypatch.setattr(nodes, "rag_question_tool", mock_rag_question_tool)

    result = run_agent(
        {
            "user_query": "What is ColliderOpsAI?",
            "errors": [],
        }
    )

    assert result.get("intent") == "rag_question"
    assert result.get("rag_answer") == "ColliderOpsAI is an AI-assisted research workbench."
    assert result.get("retrieved_documents")
    assert result.get("final_response") == "ColliderOpsAI is an AI-assisted research workbench."
