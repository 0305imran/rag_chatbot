"""Ingest resolved tickets from data/tickets.db into the `tickets` collection (FR-15)."""
import sqlite3

from langchain_core.documents import Document

from config import TICKETS_DB_PATH, rebuild_collection

COLLECTION = "tickets"


def load_ticket_documents():
    conn = sqlite3.connect(TICKETS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE status = 'resolved' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    docs, ids = [], []
    for r in rows:
        docs.append(
            Document(
                page_content=(
                    f"Issue: {r['issue_type']} ({r['category']})\n"
                    f"Customer problem: {r['description']}\n"
                    f"Resolution: {r['resolution']}"
                ),
                metadata={
                    "source": "TICKET",
                    "ref": r["ticket_id"],
                    "category": r["category"],
                    "issue_type": r["issue_type"],
                },
            )
        )
        ids.append(f"ticket-{r['ticket_id']}")
    return docs, ids


def main():
    docs, ids = load_ticket_documents()
    n = rebuild_collection(COLLECTION, docs, ids)
    print(f"[tickets] ingested {n} resolved tickets")


if __name__ == "__main__":
    main()
