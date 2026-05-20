from langchain_classic.schema import Document

SYSTEM_PROMPT = """
You are ColliderOpsAI Research Assistant.

You help researchers, ML engineers, and analysts understand the ColliderOpsAI

project, dataset, model behavior, evaluation results, and prediction workflow.

Use only the provided context to answer the question.

If the context does not contain enough information, say that the available

documentation does not provide enough detail.

Keep answers clear, practical, and grounded in the retrieved documents.

When possible, mention the source document names used for the answer.
"""

ANSWER_PROMPT_TEMPLATE = """
System instructions:

{system_prompt}

Retrieved context:

{context}

User question:

{question}

Answer:
"""

def format_documents_for_prompt(documents:list[Document])->str:

    """ Format retrieved documents into a readable context block."""

    formatted_chunks = []

    for index, document in enumerate(documents, start=1):
        file_name = document.metadata.get("file_name", "unknown_space")
        source = document.metadata.get("source", "unknown_path")
    
        formatted_chunk = f"""
[Document {index}]
Source file: {file_name}
Source path: {source}

{document.page_content}
"""
        formatted_chunks.append(formatted_chunk.strip())
    
    return "\n\n---\n\n".join(formatted_chunks)

def build_rag_prompt(question:str, documents:list[Document]) -> str:

    context = format_documents_for_prompt(documents=documents)

    return ANSWER_PROMPT_TEMPLATE.format(
        system_prompt = SYSTEM_PROMPT.strip(),
        context = context,
        question = question,
    )

