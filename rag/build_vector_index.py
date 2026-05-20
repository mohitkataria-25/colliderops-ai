from pathlib import Path
from typing import List

from langchain_classic.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_classic.schema import Document
from langchain_classic.vectorstores import Chroma

from rag.config import EmbeddingConfig
from rag.ingest_docs import ingest_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSIST_DIRECTORY = PROJECT_ROOT / "rag" / "vector_store" / "chroma"


def create_embeddings():
    """Create the embedding model used to convert text chunks into vectors."""
    return SentenceTransformerEmbeddings(
        model_name=EmbeddingConfig.MODEL_NAME
    )


def create_vector_store(
    chunks: List[Document],
    embedding_model,
    persist_directory: str | Path = PERSIST_DIRECTORY,
) -> Chroma:
    """Create and persist a Chroma vector store from document chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_directory),
    )


def build_vector_index() -> Chroma:
    """Build the local vector index from project documentation."""
    chunks = ingest_documents()

    if not chunks:
        raise ValueError("No document chunks found. Check ingest_docs.py and docs/ directory.")

    embedding_model = create_embeddings()

    vector_db = create_vector_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )

    print(f"Indexed {len(chunks)} chunks")
    print(f"Vector store saved at: {PERSIST_DIRECTORY}")

    vector_db.persist()
    return vector_db


if __name__ == "__main__":
    build_vector_index()