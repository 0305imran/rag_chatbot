"""Append-only JSONL interaction log (FR-05a). One line per answer, one line per feedback."""
import json
import logging
import uuid
from datetime import datetime, timezone

from config import LOG_DIR

LOG_FILE = LOG_DIR / "interactions.jsonl"


_agent_logger = logging.getLogger("telecom.agent")


def setup_agent_logging() -> None:
    """Send workflow steps (agent PRD §12) to stderr and logs/agent.log. Safe to call repeatedly."""
    if _agent_logger.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        _agent_logger.addHandler(handler)
    _agent_logger.setLevel(logging.INFO)
    _agent_logger.propagate = False


class Trace:
    """Collects the major workflow steps of one turn, e.g. "[Intent] refund_request".

    Steps are logged and also kept so the UI can show them. Only actions and results are
    recorded, never the LLM's reasoning.
    """

    def __init__(self):
        self.steps: list[str] = []

    def __call__(self, step: str) -> None:
        self.steps.append(step)
        _agent_logger.info(step)


def _write(record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_answer(question: str, answer: str, sources: list[str], latency_s: float,
               first_token_s: float | None, channel: str, intent: str | None = None) -> str:
    interaction_id = uuid.uuid4().hex
    _write({
        "event": "answer",
        "interaction_id": interaction_id,
        "channel": channel,
        "intent": intent,
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
