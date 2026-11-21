# RAG Telecom Customer Care Chatbot

Answers Tier-1 telecom support questions using only three local knowledge sources: FAQ (`data/faq.csv`), resolved tickets (`data/tickets.db`) and the PDF guide (`data/telecom_guide.pdf`). See [PRD.md](PRD.md).

It also handles **refund requests** with a tool-using Refund Agent. See [agent_PRD.md](agent_PRD.md) and [Refund agent](#refund-agent) below.

**Stack:** LangChain (LCEL) · ChromaDB · `all-MiniLM-L6-v2` (local embeddings) · `qwen/qwen3.6-27b` on Groq · Streamlit

## Setup (project-local venv only)

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu torch -r requirements.txt
cp .env.example .env        # then put your GROQ_API_KEY in .env
```

Without uv: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`. The extra index above just gets the smaller CPU-only torch build.
On macOS or Linux, the venv's Python is `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

## Ingest the knowledge base

```bash
.venv/Scripts/python.exe ingest_all.py
```

This builds `chroma_store/`, which holds 25 FAQs, 19 resolved tickets and 37 guide chunks. You can safely run it again. You can also run a single source: `ingest_faq.py`, `ingest_tickets.py` or `ingest_guides.py`.

## Run

```bash
.venv/Scripts/python.exe -m streamlit run app.py   # web UI at http://localhost:8501
.venv/Scripts/python.exe main.py                   # CLI REPL, type 'quit' to exit
```

The first start takes about 30 s while torch and the embedding model load. After that, the model stays cached.

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths, model settings, embedding and Chroma factories |
| `ingest_*.py` | One ingest script per knowledge source |
| `retriever.py` | Collection registry and parallel merged retriever (top-3 each) |
| `chain.py` | Grounded prompt (with recent conversation history), Groq LLM, streaming answer and sources |
| `assistant.py` | Router: intent detection, then RAG or the Refund Agent; trims history to the last 5 exchanges |
| `intent.py` | LLM intent classifier (`requirement_inquiry` / `refund_request`) with structured output |
| `refund_agent.py` | Refund Agent: LLM tool loop, workflow state machine, confirmation parsing, guardrails |
| `refund_tools.py` | `get_user_account`, `validate_refund`, `process_refund`, `submit_to_support` (deterministic, simulated) |
| `refund_data.py` | Hard-coded demo accounts |
| `app.py` | Streamlit UI with sample questions, sources, 👍/👎 and clear button |
| `main.py` | CLI REPL |
| `interaction_log.py` | Writes answers and feedback to `logs/interactions.jsonl`, and agent workflow steps to `logs/agent.log` |
| `tests/` | Refund agent and router tests, using a scripted fake LLM (no API key needed) |

## Adding a knowledge source

1. Write `ingest_<name>.py` that calls `config.rebuild_collection("<name>", docs, ids)`. Give each document metadata `source` and `ref`.
2. Add `"<name>": "<LABEL>"` to `COLLECTIONS` in `retriever.py`.

## Refund agent

```
User -> Intent Detection -> requirement_inquiry -> existing RAG pipeline
                         -> refund_request      -> Refund Agent
                                                   get_user_account -> determine amount -> validate_refund
                                                   ineligible -> reject
                                                   eligible, ≤ ₹499 -> Confirm Refund / Cancel -> process_refund
                                                   eligible, > ₹499 -> submit_to_support (no outcome is claimed)
```

- **The LLM** (the same Groq model as RAG) classifies intent. As the agent, it may only call `get_user_account` and `validate_refund`, and its job is to work out which amount the customer means.
- **Application code** decides everything about money:
  - It checks that the amount was really stated by the customer or is on their account.
  - It applies the eligibility rules: completed, within 7 days, not already refunded, and no more than the recharge amount.
  - It enforces the ₹499 approval limit.
  - It requires an explicit "Yes, proceed" (or the Confirm button). Ambiguous replies get the question again.
  - It is the only caller of `process_refund` and `submit_to_support`, which re-check the rules themselves.
- **Refunds are simulated.** Each session gets its own copy of the demo data. The sidebar lets you switch the demo account (Rahul `user_123` has ₹999 and ₹499 recharges; Priya `user_789` has a 15-day-old ₹299 recharge, for Scenario 3), view its data, and reset it.
- **Conversation history** (the last 5 exchanges) is passed to the RAG prompt and to both refund LLM calls. The retrieval query is still the raw question. **Clear conversation** resets the history and any pending refund. In the CLI, type `clear`.

Run the tests (no Groq key or knowledge base needed):

```bash
.venv/Scripts/python.exe -m pip install pytest
.venv/Scripts/python.exe -m pytest tests
```
