"""Chunk data/telecom_guide.pdf (600 chars / 100 overlap) into the `guides` collection (FR-16)."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import GUIDE_CHUNK_OVERLAP, GUIDE_CHUNK_SIZE, GUIDE_PDF_PATH, rebuild_collection

COLLECTION = "guides"


def load_guide_documents():
    pages = PyPDFLoader(str(GUIDE_PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=GUIDE_CHUNK_SIZE, chunk_overlap=GUIDE_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    docs, ids = [], []
    for i, chunk in enumerate(chunks, start=1):
        page = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata = {
            "source": "GUIDE",
            "ref": f"{GUIDE_PDF_PATH.name} p.{page} #{i}",
            "page": page,
        }
        docs.append(chunk)
        ids.append(f"guide-{i}")
    return docs, ids


def main():
    docs, ids = load_guide_documents()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[guides] ingested {n} guide chunks")


if __name__ == "__main__":
    main()
