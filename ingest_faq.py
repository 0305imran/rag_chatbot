"""Ingest data/faq.csv into the `faq` collection — 1 row = 1 document (FR-14)."""
import csv

from langchain_core.documents import Document

from config import FAQ_PATH, rebuild_collection

COLLECTION = "faq"


def load_faq_documents():
    docs, ids = [], []
    with open(FAQ_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            faq_id = (row.get("id") or str(len(ids) + 1)).strip()
            docs.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "source": "FAQ",
                        "ref": f"FAQ-{faq_id}",
                        "question": question,
                        "category": (row.get("category") or "").strip(),
                    },
                )
            )
            ids.append(f"faq-{faq_id}")
    return docs, ids


def main():
    docs, ids = load_faq_documents()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[faq] ingested {n} FAQ entries")


if __name__ == "__main__":
    main()
