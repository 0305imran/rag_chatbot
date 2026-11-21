"""Router: intent detection in front of the existing RAG pipeline (agent PRD §7, §15).

    requirement_inquiry -> TelecomRAG (unchanged retrieval, plus conversation history)
    refund_request      -> RefundAgent
"""
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import HISTORY_MAX_EXCHANGES
from intent import REFUND_REQUEST, REQUIREMENT_INQUIRY, IntentClassifier
from interaction_log import Trace
from refund_agent import (AMBIGUOUS, AWAITING_AMOUNT, AWAITING_CONFIRMATION, RefundAgent,
                          RefundContext, amounts_in, parse_confirmation)


@dataclass
class AssistantResponse:
    intent: str
    stream: Iterator[str]
    sources: list[Document] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    confirmation: dict | None = None   # refund awaiting Confirm / Cancel


def to_history(messages: list[dict], max_exchanges: int = HISTORY_MAX_EXCHANGES) -> list[BaseMessage]:
    """Last N user/assistant exchanges as chat messages (FR-45, FR-46, FR-50)."""
    recent = messages[-2 * max_exchanges:] if max_exchanges > 0 else []
    out = []
    for m in recent:
        if m.get("role") == "user":
            out.append(HumanMessage(m["content"]))
        elif m.get("role") == "assistant":
            out.append(AIMessage(m["content"]))
    return out


class SupportAssistant:
    def __init__(self, rag=None, classifier=None, refund_agent=None):
        if rag is None or classifier is None or refund_agent is None:
            from chain import TelecomRAG, build_llm
            llm = build_llm(streaming=False)   # FR-04: same Groq model/config as the RAG chain
            rag = rag or TelecomRAG()
            classifier = classifier or IntentClassifier(llm)
            refund_agent = refund_agent or RefundAgent(llm)
        self.rag = rag
        self.classifier = classifier
        self.refund_agent = refund_agent

    def route(self, message: str, history: list[BaseMessage], ctx: RefundContext, trace) -> str:
        status = ctx.session.status
        if status == AWAITING_CONFIRMATION:
            # A pending refund answer is never re-interpreted by the LLM (FR-36c..e).
            trace(f"[Intent] {REFUND_REQUEST} (reply to pending confirmation)")
            return REFUND_REQUEST
        if status == AWAITING_AMOUNT and (amounts_in(message) or parse_confirmation(message) != AMBIGUOUS):
            trace(f"[Intent] {REFUND_REQUEST} (reply to pending amount question)")
            return REFUND_REQUEST
        intent = self.classifier.classify(message, history)
        trace(f"[Intent] {intent}")
        return intent

    def handle(self, message: str, messages_before: list[dict], ctx: RefundContext) -> AssistantResponse:
        """Answer one user message. ``messages_before`` is the session history, excluding it."""
        trace = Trace()
        history = to_history(messages_before)
        intent = self.route(message, history, ctx, trace)

        if intent == REFUND_REQUEST:
            result = self.refund_agent.handle(message, history, ctx, trace)
            return AssistantResponse(intent, iter([result.text]), [], trace.steps, result.confirmation)

        if ctx.session.pending:
            trace(f"[Agent] pending refund ({ctx.session.status}) dropped: customer moved on")
            ctx.session.reset()
        response = self.rag.ask(message, history)
        trace(f"[RAG] retrieved {len(response.sources)} documents")
        return AssistantResponse(REQUIREMENT_INQUIRY, response.stream, response.sources, trace.steps)
