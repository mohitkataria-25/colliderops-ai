"""
Central configuration module.

Each class owns configuration for a single responsibility.
No logic. No imports from project code. Constants only.
"""


# =========================
# Embeddings / Ingest
# =========================
class EmbeddingConfig:

    MODEL_NAME = "BAAI/bge-base-en-v1.5"
    DEVICE = "cuda"

# =========================
# Embeddings / Ingest
# =========================
class RetrievalConfig:

    STRATEGY = "similarity"
    TOP_K = 3

# =========================
# Embeddings / Ingest
# =========================
class AnswerModelConfig:
    """
    Configuration for the main answer-generation LLM.
    """
    REPO_ID = "TheBloke/Llama-2-13B-chat-GGUF"
    MODEL_FILE = "llama-2-13b-chat.Q5_K_M.gguf"

    CONTEXT_SIZE = 4096
    GPU_LAYERS = 38

    TEMPERATURE = 0.2
    TOP_P = 0.95
    MAX_TOKENS = 512
    N_BATCHES = 38 