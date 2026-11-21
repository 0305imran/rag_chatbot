"""CLI REPL for the telecom support chatbot (FR-18, FR-19). Type 'quit' to exit, 'clear' to reset."""
import sys
import time

from assistant import SupportAssistant
from config import DEMO_USER_ID
from interaction_log import log_answer, setup_agent_logging
from refund_agent import RefundContext
from refund_data import new_store
from refund_tools import RefundTools
from retriever import collection_counts


def main():
    counts = collection_counts()
    if not all(counts.values()):
        print(f"Knowledge base is empty or incomplete {counts}. Run: python ingest_all.py")
        sys.exit(1)

    try:
        assistant = SupportAssistant()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
    setup_agent_logging()

    # Simulated authenticated session (agent PRD FR-15..17).
    ctx = RefundContext(user_id=DEMO_USER_ID, tools=RefundTools(new_store()))
    messages: list[dict] = []   # agent PRD FR-45: in-session history

    print("Telecom Support Assistant — ask a question or request a refund.\n"
          "Type 'clear' to reset the conversation, 'quit' to exit.\n")
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
        if question.lower() == "clear":   # agent PRD FR-48
            messages.clear()
            ctx.session.reset()
            print("(conversation cleared)\n")
            continue

        start = time.perf_counter()
        first_token = None
        answer_parts = []
        try:
            response = assistant.handle(question, messages, ctx)
            print("Bot: ", end="", flush=True)
            for token in response.stream:
                if first_token is None:
                    first_token = time.perf_counter() - start
                answer_parts.append(token)
                print(token, end="", flush=True)
        except Exception as e:  # network / API errors shouldn't kill the REPL
            print(f"\n[error] {e}\n")
            continue

        answer = "".join(answer_parts)
        refs = [d.metadata.get("ref", "?") for d in response.sources]
        if refs:
            print(f"\n\nSources: {', '.join(refs)}")
        if response.confirmation:
            print("\n(Reply 'Yes, proceed' to confirm or 'Cancel'.)")
        print()
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        log_answer(question, answer, refs, time.perf_counter() - start, first_token,
                   channel="cli", intent=response.intent)

    print("Goodbye!")


if __name__ == "__main__":
    main()
