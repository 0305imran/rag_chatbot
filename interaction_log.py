"""Append-only JSONL interaction log (FR-05a). One line per answer, one line per feedback."""
import json
import uuid
from datetime import datetime, timezone

from config import LOG_DIR

LOG_FILE = LOG_DIR / "interactions.jsonl"


def _write(record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_answer(question: str, answer: str, sources: list[str], latency_s: float,
               first_token_s: float | None, channel: str) -> str:
    interaction_id = uuid.uuid4().hex
    _write({
        "event": "answer",
        "interaction_id": interaction_id,
        "channel": channel,
        "question": question,
        "answer": answer,
        "sources": sources,
        "first_token_s": round(first_token_s, 3) if first_token_s is not None else None,
        "latency_s": round(latency_s, 3),
    })
    return interaction_id


def log_feedback(interaction_id: str, rating: str) -> None:
    """rating: 'up' or 'down'."""
    _write({"event": "feedback", "interaction_id": interaction_id, "rating": rating})
