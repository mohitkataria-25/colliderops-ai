from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    errors: list[str]
    intent: str

    prediction_payload: dict[str, Any] | None
    batch_prediction_payloads: list[dict[str, Any]] | None
    uploaded_file_path: str | None

    prediction_result: dict[str, Any] | None
    batch_summary: dict[str, Any] | None
    csv_output: str | None

    retrieved_documents: list[Any]
    rag_answer: str | None

    model_metadata: dict[str, Any] | None
    etl_status: dict[str, Any] | None

    final_response: str | None