"""Merged retriever: queries every registered collection in parallel (FR-06, FR-07, FR-08).

To add a knowledge source (NFR-06): write an ingest_<name>.py that fills a Chroma
collection, then add an entry to COLLECTIONS below.
"""
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from config import CHROMA_DIR, TOP_K, get_vectorstore

# collection name -> label used in the prompt context
COLLECTIONS = {
    "faq": "FAQ",
    "tickets": "TICKETS",
    "guides": "GUIDES",
}


def build_retriever() -> RunnableParallel:
    """RunnableParallel that returns {collection_name: [Document, ...]} for a question."""
    return RunnableParallel(
        {
            name: get_vectorstore(name).as_retriever(search_kwargs={"k": TOP_K})
            for name in COLLECTIONS
        }
    )


def flatten(results: dict[str, list[Document]]) -> list[Document]:
    """Merge per-collection results into one list, in registry order."""
    return [doc for name in COLLECTIONS for doc in results.get(name, [])]


def format_context(results: dict[str, list[Document]]) -> str:
    """Source-labelled context block for the prompt (FR-08)."""
    sections = []
    for name, label in COLLECTIONS.items():
        docs = results.get(name, [])
        body = "\n\n".join(
            f"[{doc.metadata.get('ref', label)}]\n{doc.page_content}" for doc in docs
        ) or "(no documents retrieved)"
        sections.append(f"=== {label} ===\n{body}")
    return "\n\n".join(sections)


def collection_counts() -> dict[str, int]:
    """Number of documents per collection — used to warn if ingestion hasn't run.

    Reads Chroma directly so it doesn't need to load the embedding model.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {c.name for c in client.list_collections()}
    return {
        name: client.get_collection(name).count() if name in existing else 0
        for name in COLLECTIONS
    }
