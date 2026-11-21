"""Shared configuration: paths, model names, and the embedding/vector-store factories."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"
LOG_DIR = BASE_DIR / "logs"

FAQ_PATH = DATA_DIR / "faq.csv"
TICKETS_DB_PATH = DATA_DIR / "tickets.db"
GUIDE_PDF_PATH = DATA_DIR / "telecom_guide.pdf"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # FR-09: local embeddings
# FR-13 specifies qwen/qwen3.6-27b, but Groq no longer serves it; qwen3.8-27b is the
# closest available model. Override with LLM_MODEL in .env.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = 0                                           # FR-12
LLM_REASONING_EFFORT = "none"                                 # FR-13

HISTORY_MAX_EXCHANGES = 5     # agent PRD FR-50: recent user/assistant exchanges passed as context

# ---- Refund agent (agent PRD) ----
DEMO_USER_ID = "user_123"     # FR-15/16: stands in for the authenticated session's user
REFUND_APPROVAL_LIMIT = 499   # FR-36/37: refunds above this go to the Support Team
REFUND_WINDOW_DAYS = 7        # FR-25
AGENT_MAX_STEPS = 4           # cap on LLM tool-calling rounds per refund turn

TOP_K = 3                     # FR-07: top-3 per collection
GUIDE_CHUNK_SIZE = 600        # FR-16
GUIDE_CHUNK_OVERLAP = 100     # FR-16


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the local sentence-transformers model once per process."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(collection_name: str):
    """Open (or create) a persisted Chroma collection."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def rebuild_collection(collection_name: str, documents, ids):
    """Idempotent (FR-17): drop the collection's contents and re-add every document."""
    store = get_vectorstore(collection_name)
    store.reset_collection()
    store.add_documents(documents=documents, ids=ids)
    return len(documents)
