

from unittest.mock import Mock, patch

from langchain_classic.schema import Document

from rag.pipeline import extract_sources, generate_answer, run_retrieval_only
from rag.prompts import build_rag_prompt, format_documents_for_prompt


SAMPLE_DOCUMENTS = [
    Document(
        page_content="ColliderOpsAI is an AI-assisted research workbench for collider-style datasets.",
        metadata={
            "file_name": "research_workbench_overview.md",
            "source": "docs/research_workbench_overview.md",
            "document_type": "md",
        },
    ),
    Document(
        page_content="The model classifies events as signal or background and adds confidence triage.",
        metadata={
            "file_name": "model_card.md",
            "source": "docs/model_card.md",
            "document_type": "md",
        },
    ),
]


def test_format_documents_for_prompt_includes_content_and_sources():
    context = format_documents_for_prompt(SAMPLE_DOCUMENTS)

    assert "ColliderOpsAI is an AI-assisted research workbench" in context
    assert "signal or background" in context
    assert "research_workbench_overview.md" in context
    assert "model_card.md" in context


def test_build_rag_prompt_includes_question_and_context():
    question = "What is ColliderOpsAI?"

    prompt = build_rag_prompt(question=question, documents=SAMPLE_DOCUMENTS)

    assert question in prompt
    assert "Retrieved context" in prompt
    assert "ColliderOpsAI is an AI-assisted research workbench" in prompt


def test_extract_sources_returns_source_metadata():
    sources = extract_sources(SAMPLE_DOCUMENTS)

    assert len(sources) == 2
    assert sources[0]["file_name"] == "research_workbench_overview.md"
    assert sources[0]["source"] == "docs/research_workbench_overview.md"
    assert sources[0]["document_type"] == "md"


def test_run_retrieval_only_uses_pipeline_retriever():
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = SAMPLE_DOCUMENTS

    pipeline_state = {"retriever": mock_retriever}

    result = run_retrieval_only(
        question="What does the model do?",
        pipeline_state=pipeline_state,
    )

    mock_retriever.invoke.assert_called_once_with("What does the model do?")
    assert result == SAMPLE_DOCUMENTS


def test_generate_answer_returns_llm_content():
    mock_response = Mock()
    mock_response.content = "ColliderOpsAI helps researchers classify and triage collider-style events."

    mock_llm = Mock()
    mock_llm.invoke.return_value = mock_response

    answer = generate_answer(
        question="What is ColliderOpsAI?",
        documents=SAMPLE_DOCUMENTS,
        llm=mock_llm,
    )

    assert "classify and triage" in answer
    mock_llm.invoke.assert_called_once()


@patch("rag.pipeline.generate_answer")
def test_run_pipeline_returns_question_answer_sources_and_documents(mock_generate_answer):
    from rag.pipeline import run_pipeline

    mock_generate_answer.return_value = "ColliderOpsAI is a research workbench."

    mock_retriever = Mock()
    mock_retriever.invoke.return_value = SAMPLE_DOCUMENTS

    pipeline_state = {
        "retriever": mock_retriever,
        "answer_llm": Mock(),
    }

    result = run_pipeline(
        question="What is ColliderOpsAI?",
        pipeline_state=pipeline_state,
    )

    assert result["question"] == "What is ColliderOpsAI?"
    assert result["answer"] == "ColliderOpsAI is a research workbench."
    assert "sources" in result
    assert "retrieved_documents" in result
    assert len(result["sources"]) == 2
    assert result["retrieved_documents"] == SAMPLE_DOCUMENTS