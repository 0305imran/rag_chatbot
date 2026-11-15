# RAG Telecom Customer Care Chatbot

Answers Tier-1 telecom support questions using only three local knowledge sources: FAQ (`data/faq.csv`), resolved tickets (`data/tickets.db`) and the PDF guide (`data/telecom_guide.pdf`). See [PRD.md](PRD.md).

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
| `chain.py` | Grounded prompt, Groq LLM, streaming answer and sources |
| `app.py` | Streamlit UI with sample questions, sources, 👍/👎 and clear button |
| `main.py` | CLI REPL |
| `interaction_log.py` | Writes answers and feedback to `logs/interactions.jsonl` |

## Adding a knowledge source

1. Write `ingest_<name>.py` that calls `config.rebuild_collection("<name>", docs, ids)`. Give each document metadata `source` and `ref`.
2. Add `"<name>": "<LABEL>"` to `COLLECTIONS` in `retriever.py`.
