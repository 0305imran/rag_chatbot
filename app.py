"""Streamlit chat UI for the telecom RAG chatbot. Run: streamlit run app.py"""
import time

import streamlit as st

from assistant import SupportAssistant
from config import DEMO_USER_ID
from interaction_log import log_answer, log_feedback, setup_agent_logging
from refund_agent import RefundContext
from refund_data import USERS, new_store
from refund_tools import RefundTools
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

REFUND_SAMPLES = [
    "I want a refund of ₹499.",
    "I want a refund of ₹999.",
    "I want a refund.",
    "Give me a refund of ₹2,000.",
]

CONFIRM_TEXT, CANCEL_TEXT = "Yes, proceed", "Cancel"

SOURCE_ICONS = {"FAQ": "❓", "TICKET": "🎫", "GUIDE": "📘"}

st.set_page_config(page_title="Telecom Support Assistant", page_icon="📶", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def get_assistant() -> SupportAssistant:
    setup_agent_logging()
    return SupportAssistant()


def new_refund_context(user_id: str) -> RefundContext:
    """Per-session simulated account data; user_id stands in for the authenticated session."""
    return RefundContext(user_id=user_id, tools=RefundTools(new_store()))


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


def render_confirmation(msg: dict, idx: int) -> None:
    """Agent PRD §11: Confirm Refund / Cancel, only while this refund is still pending."""
    ctx = st.session_state.refund_ctx
    is_last = idx == len(st.session_state.messages) - 1
    if st.session_state.get("_busy") or not (
            msg.get("confirmation") and is_last and ctx.session.status == "awaiting_confirmation"):
        return
    col1, col2 = st.columns(2)
    if col1.button("✅ Confirm Refund", key=f"confirm_{idx}", type="primary", use_container_width=True):
        st.session_state.pending_question = CONFIRM_TEXT
        st.rerun()
    if col2.button("Cancel", key=f"cancel_{idx}", use_container_width=True):
        st.session_state.pending_question = CANCEL_TEXT
        st.rerun()


def render_assistant_extras(msg: dict, idx: int) -> None:
    if msg.get("sources"):
        render_sources(msg["sources"])
    if msg.get("intent") == "refund_request" and msg.get("trace"):
        with st.expander("Agent steps"):
            st.code("\n".join(msg["trace"]), language=None)
    render_confirmation(msg, idx)
    if msg.get("interaction_id"):
        render_feedback(msg, idx)


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []   # FR-03: per-session history
if "refund_ctx" not in st.session_state:
    st.session_state.refund_ctx = new_refund_context(DEMO_USER_ID)
pending = st.session_state.pop("pending_question", None)

# ---------- sidebar ----------
with st.sidebar:
    st.title("📶 Telecom Support")
    st.caption("Answers come only from our FAQ, resolved tickets, and official guides.")
    st.subheader("Try a sample question")
    for i, q in enumerate(SAMPLE_QUESTIONS):          # FR-02
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            pending = q
    st.subheader("Try a refund request")
    for i, q in enumerate(REFUND_SAMPLES):
        if st.button(q, key=f"refund_sample_{i}", use_container_width=True):
            pending = q
    st.divider()

    # Simulated authenticated session (agent PRD FR-15..17): the user never types an ID.
    ctx = st.session_state.refund_ctx
    user_ids = list(USERS)
    chosen = st.selectbox("Signed-in demo account", user_ids, index=user_ids.index(ctx.user_id),
                          format_func=lambda u: f"{USERS[u]['name']} ({u})")
    if chosen != ctx.user_id:
        st.session_state.refund_ctx = ctx = new_refund_context(chosen)
    with st.expander("Account data (simulated)"):
        for r in ctx.tools.get_user_account(ctx.user_id)["recharges"]:
            st.caption(f"{r['recharge_id']} · ₹{r['recharge_amount']} · {r['recharge_status']} · "
                       f"{r['recharge_age_days']}d ago · {r['refund_status']}")
    if st.button("↺ Reset demo account data", use_container_width=True):
        st.session_state.refund_ctx = new_refund_context(ctx.user_id)
        st.rerun()
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):   # FR-04, agent FR-48
        st.session_state.messages = []
        st.session_state.refund_ctx.session.reset()   # §11: also drops any pending refund
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
    assistant = get_assistant()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if not st.session_state.messages and not pending:
    st.info("Ask about data, billing, SIM/eSIM, roaming, calls, or your account, ask for a "
            "refund of a recent recharge — or pick a sample from the sidebar.")

question = st.chat_input("Type your question…") or pending   # FR-01
st.session_state._busy = bool(question)   # don't draw stale Confirm buttons mid-turn

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg, idx)

if question:
    history_before = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        first_token = {"t": None}
        try:
            with st.spinner("Thinking…"):
                response = assistant.handle(question, history_before, st.session_state.refund_ctx)

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
            time.perf_counter() - start, first_token["t"], channel="web", intent=response.intent,
        )
        msg = {"role": "assistant", "content": answer, "sources": sources,
               "interaction_id": interaction_id, "intent": response.intent,
               "trace": response.trace, "confirmation": response.confirmation}
        st.session_state.messages.append(msg)
        st.session_state._busy = False
        render_assistant_extras(msg, len(st.session_state.messages) - 1)
