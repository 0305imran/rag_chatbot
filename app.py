"""Streamlit chat UI for the telecom RAG chatbot. Run: streamlit run app.py"""
import time

import streamlit as st

from chain import TelecomRAG
from interaction_log import log_answer, log_feedback
from retriever import collection_counts

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "I was charged twice on my bill. What should I do?",
    "How do I activate roaming before travelling abroad?",
    "My phone says SIM not recognised. How do I fix it?",
    "My eSIM activation failed. What can I do?",
    "There's an echo on my calls. How can I fix it?",
    "How do I download my itemised bill?",
    "I can't log in to the MyTelecom app.",
]

SOURCE_ICONS = {"FAQ": "❓", "TICKET": "🎫", "GUIDE": "📘"}

st.set_page_config(page_title="Telecom Support Assistant", page_icon="📶", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def get_rag() -> TelecomRAG:
    return TelecomRAG()


def render_sources(sources: list[dict]) -> None:
    """FR-13a: expandable list of every retrieved document."""
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            icon = SOURCE_ICONS.get(s["source"], "•")
            st.markdown(f"**{icon} {s['source']} · {s['ref']}**")
            st.caption(s["content"])


def render_feedback(msg: dict, idx: int) -> None:
    """FR-05a: thumbs up/down written to the interaction log once per change."""
    rating = st.feedback("thumbs", key=f"fb_{idx}")
    if rating is not None:
        label = "up" if rating == 1 else "down"
        if msg.get("rating") != label:
            msg["rating"] = label
            log_feedback(msg["interaction_id"], label)


def render_assistant_extras(msg: dict, idx: int) -> None:
    if msg.get("sources"):
        render_sources(msg["sources"])
    if msg.get("interaction_id"):
        render_feedback(msg, idx)


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # FR-03: per-session history
pending = st.session_state.pop("pending_question", None)

# ---------- sidebar ----------
with st.sidebar:
    st.title("📶 Telecom Support")
    st.caption("Answers come only from our FAQ, resolved tickets, and official guides.")
    st.subheader("Try a sample question")
    for i, q in enumerate(SAMPLE_QUESTIONS):          # FR-02
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            pending = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):   # FR-04
        st.session_state.messages = []
        for k in [k for k in st.session_state if k.startswith("fb_")]:
            del st.session_state[k]
        st.rerun()
    st.caption("Need a human? Call **611** or use the **MyTelecom** app.")

# ---------- main ----------
st.title("Telecom Support Assistant")

counts = collection_counts()
if not all(counts.values()):
    st.error(f"Knowledge base is empty or incomplete ({counts}). Run `python ingest_all.py` first.")
    st.stop()

try:
    rag = get_rag()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if not st.session_state.messages and not pending:
    st.info("Ask about data, billing, SIM/eSIM, roaming, calls, or your account — "
            "or pick a sample question from the sidebar.")

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg, idx)

question = st.chat_input("Type your question…") or pending   # FR-01

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        first_token = {"t": None}
        try:
            with st.spinner("Searching knowledge base…"):
                response = rag.ask(question)

            def timed_stream():
                for token in response.stream:
                    if first_token["t"] is None:
                        first_token["t"] = time.perf_counter() - start
                    yield token

            answer = st.write_stream(timed_stream())   # FR-05: token streaming
        except Exception as e:
            st.error(f"Sorry, something went wrong: {e}")
            st.stop()

        sources = [
            {"source": d.metadata.get("source", "?"), "ref": d.metadata.get("ref", "?"),
             "content": d.page_content}
            for d in response.sources
        ]
        interaction_id = log_answer(
            question, answer, [s["ref"] for s in sources],
            time.perf_counter() - start, first_token["t"], channel="web",
        )
        msg = {"role": "assistant", "content": answer, "sources": sources,
               "interaction_id": interaction_id}
        st.session_state.messages.append(msg)
        render_assistant_extras(msg, len(st.session_state.messages) - 1)
