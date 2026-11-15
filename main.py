"""CLI REPL for the telecom RAG chatbot (FR-18, FR-19). Type 'quit' to exit."""
import sys
import time

from chain import TelecomRAG
from interaction_log import log_answer
from retriever import collection_counts


def main():
    counts = collection_counts()
    if not all(counts.values()):
        print(f"Knowledge base is empty or incomplete {counts}. Run: python ingest_all.py")
        sys.exit(1)

    try:
        rag = TelecomRAG()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    print("Telecom Support Assistant — ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break

        start = time.perf_counter()
        first_token = None
        answer_parts = []
        try:
            response = rag.ask(question)
            print("Bot: ", end="", flush=True)
            for token in response.stream:
                if first_token is None:
                    first_token = time.perf_counter() - start
                answer_parts.append(token)
                print(token, end="", flush=True)
        except Exception as e:  # network / API errors shouldn't kill the REPL
            print(f"\n[error] {e}\n")
            continue

        refs = [d.metadata.get("ref", "?") for d in response.sources]
        print(f"\n\nSources: {', '.join(refs)}\n")
        log_answer(question, "".join(answer_parts), refs,
                   time.perf_counter() - start, first_token, channel="cli")

    print("Goodbye!")


if __name__ == "__main__":
    main()
