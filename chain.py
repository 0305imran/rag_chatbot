"""RAG chain: merged retrieval -> grounded prompt -> Qwen3.6-27B on Groq -> streamed text."""
import os
from dataclasses import dataclass
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, LLM_REASONING_EFFORT, LLM_TEMPERATURE
from retriever import build_retriever, flatten, format_context

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information to answer that. "
    "Please call 611 or use the MyTelecom app to reach our support team."
)

SYSTEM_PROMPT = """You are a friendly, concise telecom customer-care assistant.

Answer the customer's question using ONLY the context below, which comes from three \
verified sources: FAQ entries, resolved support tickets, and the official telecom guide.

Rules:
- Use only facts stated in the context. Never use outside knowledge, and never invent \
prices, fees, timelines, policies, phone numbers, or URLs.
- Resolved tickets describe what fixed the issue for another customer; present them as \
things the customer can try or ask support about, not as facts about this customer's account.
- You have no access to the customer's account, balance, bill, or usage. If they ask for \
personal account data, explain you can't see it and tell them where to find it if the \
context says so.
- Prefer short numbered steps for troubleshooting.
- If the context does not contain enough information to answer, reply exactly:
"{fallback}"

Context:
{context}"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    ).partial(fallback=FALLBACK_MESSAGE)


def build_llm():
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://console.groq.com)."
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        reasoning_effort=LLM_REASONING_EFFORT,
        streaming=True,
    )


@dataclass
class RAGResponse:
    """Retrieved sources plus a token stream for the answer."""
    sources: list[Document]
    stream: Iterator[str]


class TelecomRAG:
    def __init__(self):
        self.retriever = build_retriever()
        self.answer_chain = build_prompt() | build_llm() | StrOutputParser()

    def retrieve(self, question: str) -> dict[str, list[Document]]:
        return self.retriever.invoke(question)

    def ask(self, question: str) -> RAGResponse:
        results = self.retrieve(question)
        stream = self.answer_chain.stream(
            {"question": question, "context": format_context(results)}
        )
        return RAGResponse(sources=flatten(results), stream=stream)
