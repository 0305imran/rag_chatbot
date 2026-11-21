"""Intent detection (agent PRD 6.1): requirement_inquiry vs refund_request."""
import re
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"
INTENTS = (REQUIREMENT_INQUIRY, REFUND_REQUEST)

INTENT_PROMPT = """You route messages for a telecom customer-care assistant.
Classify the customer's LATEST message into exactly one intent:

- refund_request: the customer wants money back, or wants a recharge/payment cancelled \
or reversed, or is continuing a refund conversation (for example, answering which \
amount to refund).
- requirement_inquiry: everything else, including questions about HOW refunds, \
cancellations, or billing work, troubleshooting, and account or plan questions.

Examples:
"Why is my internet slow?" -> requirement_inquiry
"I want a refund of ₹999." -> refund_request
"Please cancel my last recharge." -> refund_request
"Reverse yesterday's recharge." -> refund_request
"How do I cancel a recharge?" -> requirement_inquiry
"What is the refund policy?" -> requirement_inquiry
"I was charged twice on my bill. What should I do?" -> requirement_inquiry

Use the earlier conversation only to understand short follow-ups."""


class IntentResult(BaseModel):
    """Structured intent value (FR-05)."""
    intent: Literal["requirement_inquiry", "refund_request"] = Field(
        description="The customer's intent for their latest message.")


class IntentClassifier:
    """LLM classifier using the app's existing Groq model (FR-04)."""

    def __init__(self, llm):
        self.llm = llm
        self.structured = llm.with_structured_output(IntentResult)

    def classify(self, message: str, history: list[BaseMessage]) -> str:
        messages = [SystemMessage(INTENT_PROMPT), *history, HumanMessage(message)]
        try:
            result = self.structured.invoke(messages)
            if isinstance(result, IntentResult):
                return result.intent
            if isinstance(result, dict) and result.get("intent") in INTENTS:
                return result["intent"]
        except Exception:
            pass
        # Fallback: plain completion, look for the label in the text.
        try:
            text = str(self.llm.invoke(messages).content)
            if re.search(r"\brefund_request\b", text):
                return REFUND_REQUEST
        except Exception:
            pass
        return REQUIREMENT_INQUIRY   # safe default: answering a question never moves money
