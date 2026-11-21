"""Refund agent + router tests (agent PRD §8 scenarios, §9 guardrails).

The LLM is replaced with a scripted fake so the deterministic workflow is tested
without network access. Run: python -m pytest tests
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from assistant import SupportAssistant, to_history
from chain import RAGResponse
from intent import REFUND_REQUEST, REQUIREMENT_INQUIRY
from refund_agent import (AWAITING_AMOUNT, AWAITING_CONFIRMATION, CONFIRM, LLM_TOOLS, REJECT,
                          AMBIGUOUS, REFUND_CANCELLED, REFUND_PROCESSED, REFUND_REJECTED,
                          REFUND_SUBMITTED, RefundAgent, RefundContext, amounts_in,
                          parse_confirmation)
from refund_data import new_store
from refund_tools import RefundTools


# ---------- fakes ----------
def call(name, **args):
    return {"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}


class AgentLLM:
    """Plays the tool-using agent: get_user_account, then validate_refund(amount) or a reply.

    ``amount_for`` maps the latest user text to the amount the "LLM" would choose.
    """

    def __init__(self, amount_for=None):
        self.amount_for = amount_for or (lambda text: None)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        last_human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        if not any(getattr(m, "tool_call_id", None) == "call_get_user_account" for m in messages):
            return AIMessage(content="", tool_calls=[call("get_user_account")])
        amount = self.amount_for(last_human.content)
        if amount is None:
            return AIMessage(content="AMOUNT_UNCLEAR")
        return AIMessage(content="", tool_calls=[call("validate_refund", amount=amount)])


def first_stated_amount(text):
    found = sorted(amounts_in(text))
    return found[0] if len(found) == 1 else None


class SpyTools(RefundTools):
    def __init__(self, store):
        super().__init__(store)
        self.calls = []

    def process_refund(self, user_id, amount, **kw):
        self.calls.append(("process_refund", amount))
        return super().process_refund(user_id, amount, **kw)

    def submit_to_support(self, user_id, amount):
        self.calls.append(("submit_to_support", amount))
        return super().submit_to_support(user_id, amount)


class KeywordClassifier:
    def classify(self, message, history):
        return REFUND_REQUEST if "refund" in message.lower() else REQUIREMENT_INQUIRY


class FakeRAG:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None):
        self.calls.append((question, list(history or [])))
        return RAGResponse(sources=[], stream=iter(["grounded answer"]))


@pytest.fixture
def ctx():
    return RefundContext(user_id="user_123", tools=SpyTools(new_store()))


@pytest.fixture
def bot():
    rag = FakeRAG()
    return SupportAssistant(rag=rag, classifier=KeywordClassifier(),
                            refund_agent=RefundAgent(AgentLLM(first_stated_amount)))


class Chat:
    """Drives the assistant like the UI does, keeping history."""

    def __init__(self, bot, ctx):
        self.bot, self.ctx, self.messages = bot, ctx, []

    def say(self, text):
        r = self.bot.handle(text, self.messages, self.ctx)
        answer = "".join(r.stream)
        self.messages += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
        r.text = answer
        return r


def recharge(ctx, rid):
    return next(r for r in ctx.tools.store["users"][ctx.user_id]["recharges"] if r["recharge_id"] == rid)


# ---------- Scenario 1: successful refund ----------
def test_scenario1_successful_refund_needs_confirmation(bot, ctx):
    chat = Chat(bot, ctx)
    r = chat.say("I want a refund of ₹499.")
    assert r.intent == REFUND_REQUEST
    assert ctx.session.status == AWAITING_CONFIRMATION
    assert r.confirmation == {"amount": 499, "recharge_id": "rch_457"}
    assert "Refund eligible: ₹499" in r.text
    assert ctx.tools.calls == []                          # FR-36b: request is not confirmation
    assert recharge(ctx, "rch_457")["refund_status"] == "not_refunded"

    r = chat.say("Yes, proceed")
    assert ctx.session.status == REFUND_PROCESSED
    assert ctx.tools.calls == [("process_refund", 499)]
    assert recharge(ctx, "rch_457")["refund_status"] == "refunded"     # FR-33
    assert recharge(ctx, "rch_457")["refund_id"].startswith("ref_")    # FR-34
    assert recharge(ctx, "rch_457")["refund_id"] in r.text
    assert r.trace == ["[Intent] refund_request (reply to pending confirmation)", "[User] confirmed",
                       "[Agent] process_refund(amount=499)",
                       f"[Refund] success refund_id={recharge(ctx, 'rch_457')['refund_id']}"]
    # The other recharge is still refundable.
    assert recharge(ctx, "rch_456")["refund_status"] == "not_refunded"
    assert ctx.tools.validate_refund("user_123", 999)["eligible"]


def test_scenario1_trace_matches_prd_logging(bot, ctx):
    r = Chat(bot, ctx).say("I want a refund of ₹499.")
    assert r.trace == ["[Intent] refund_request", "[Agent] refund_requested",
                       "[Agent] get_user_account", "[Agent] validate_refund(amount=499)",
                       "[Validation] eligible=True", "[Agent] awaiting_user_confirmation"]


# ---------- Scenario 2: amount exceeds recharge ----------
def test_scenario2_amount_exceeds_recharge(bot, ctx):
    r = Chat(bot, ctx).say("Give me a refund of ₹2,000.")
    assert ctx.session.status == REFUND_REJECTED
    assert "exceeds your recharge amount of ₹999" in r.text
    assert ctx.tools.calls == []


# ---------- Scenario 3: recharge too old ----------
def test_scenario3_recharge_outside_window(bot):
    ctx = RefundContext(user_id="user_789", tools=SpyTools(new_store()))
    r = Chat(bot, ctx).say("I want a refund of ₹299.")
    assert ctx.session.status == REFUND_REJECTED
    assert "outside the 7-day refund window" in r.text
    assert ctx.tools.calls == []


def test_scenario3_age_15_on_demo_user(ctx):
    recharge(ctx, "rch_457")["recharge_age_days"] = 15
    v = ctx.tools.validate_refund("user_123", 499)
    assert not v["eligible"] and "7-day" in v["reason"]


# ---------- Scenario 4: already refunded ----------
def test_scenario4_already_refunded(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("I want a refund of ₹499.")
    chat.say("Yes, proceed")
    ctx.tools.calls.clear()
    r = chat.say("I want a refund of ₹499.")
    assert ctx.session.status == REFUND_REJECTED
    assert "already been refunded" in r.text
    assert ctx.tools.calls == []


# ---------- Scenario 5: existing RAG query ----------
def test_scenario5_rag_query(bot, ctx):
    r = Chat(bot, ctx).say("Why is my mobile internet slow?")
    assert r.intent == REQUIREMENT_INQUIRY
    assert r.text == "grounded answer"
    assert bot.rag.calls == [("Why is my mobile internet slow?", [])]


# ---------- Scenario 6: above the approval limit ----------
def test_scenario6_above_limit_goes_to_support(bot, ctx):
    r = Chat(bot, ctx).say("I want a refund of ₹999.")
    assert ctx.session.status == REFUND_SUBMITTED
    assert r.confirmation is None                                 # §11: no confirmation step
    assert ctx.tools.calls == [("submit_to_support", 999)]        # never process_refund
    assert recharge(ctx, "rch_456")["refund_status"] == "not_refunded"
    assert "submitted to our Support Team" in r.text
    assert "process it or take the necessary action" in r.text
    for claim in ("approved", "rejected", "has been processed", "refunded"):   # FR-40a, GR-09
        assert claim not in r.text.lower()
    assert len(ctx.tools.store["support_queue"]) == 1


# ---------- Scenario 7: follow-up uses history, retrieval query unchanged ----------
def test_scenario7_follow_up_gets_history(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("How do I activate international roaming?")
    chat.say("Should I do that before I travel?")
    question, history = bot.rag.calls[-1]
    assert question == "Should I do that before I travel?"        # FR-49
    assert [m.content for m in history] == ["How do I activate international roaming?",
                                            "grounded answer"]


def test_refund_agent_also_receives_history(ctx):
    seen = {}

    class Recording(AgentLLM):
        def invoke(self, messages):
            seen["messages"] = messages
            return super().invoke(messages)

    bot = SupportAssistant(rag=FakeRAG(), classifier=KeywordClassifier(),
                           refund_agent=RefundAgent(Recording(first_stated_amount)))
    chat = Chat(bot, ctx)
    chat.say("How do I check my balance?")
    chat.say("I want a refund of ₹499.")
    contents = [m.content for m in seen["messages"]]
    assert "How do I check my balance?" in contents


def test_history_is_limited():
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(30)]
    assert [m.content for m in to_history(msgs, 5)] == [str(i) for i in range(20, 30)]


# ---------- Missing amount (6.8) ----------
def test_missing_amount_with_two_eligible_recharges_asks(bot, ctx):
    chat = Chat(bot, ctx)
    r = chat.say("I want a refund.")
    assert ctx.session.status == AWAITING_AMOUNT
    assert "more than one recent recharge - ₹999 and ₹499" in r.text
    assert "Which amount" in r.text
    r = chat.say("499")                    # routed to the agent even without the word "refund"
    assert r.intent == REFUND_REQUEST
    assert ctx.session.status == AWAITING_CONFIRMATION
    assert r.confirmation["amount"] == 499


def test_missing_amount_single_eligible_above_limit_is_offered(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("I want a refund of ₹499.")
    chat.say("Yes, proceed")
    ctx.tools.calls.clear()
    r = chat.say("I want a refund.")
    assert ctx.session.status == AWAITING_AMOUNT and ctx.session.offered_amount == 999
    assert ctx.tools.calls == []
    r = chat.say("yes")
    assert ctx.session.status == REFUND_SUBMITTED
    assert ctx.tools.calls == [("submit_to_support", 999)]


def test_missing_amount_single_eligible_within_limit_goes_to_confirmation(bot, ctx):
    recharge(ctx, "rch_456")["refund_status"] = "refunded"
    r = Chat(bot, ctx).say("I want a refund.")
    assert ctx.session.status == AWAITING_CONFIRMATION
    assert r.confirmation["amount"] == 499


def test_no_eligible_recharge_explains(bot, ctx):
    for rid in ("rch_456", "rch_457"):
        recharge(ctx, rid)["refund_status"] = "refunded"
    r = Chat(bot, ctx).say("I want a refund.")
    assert ctx.session.status == REFUND_REJECTED
    assert r.text.count("already been refunded") == 2


def test_invented_amount_is_ignored(ctx):
    """FR-44: an amount the customer never stated and that isn't on the account is dropped."""
    bot = SupportAssistant(rag=FakeRAG(), classifier=KeywordClassifier(),
                           refund_agent=RefundAgent(AgentLLM(lambda text: 350)))
    r = Chat(bot, ctx).say("I want a refund.")
    assert ctx.session.status == AWAITING_AMOUNT
    assert "[Guardrail] ignored amount ₹350: not stated by customer or on account" in r.trace
    assert ctx.tools.calls == []


def test_account_amount_chosen_by_llm_above_limit_is_offered_not_submitted(ctx):
    """'Refund my bigger recharge' -> 999 from account data: ask before submitting."""
    bot = SupportAssistant(rag=FakeRAG(), classifier=KeywordClassifier(),
                           refund_agent=RefundAgent(AgentLLM(lambda text: 999)))
    Chat(bot, ctx).say("Refund my bigger recharge")
    assert ctx.session.status == AWAITING_AMOUNT and ctx.session.offered_amount == 999
    assert ctx.tools.calls == []


def test_llm_failure_falls_back_deterministically(ctx):
    class Broken(AgentLLM):
        def invoke(self, messages):
            raise ConnectionError("groq down")

    bot = SupportAssistant(rag=FakeRAG(), classifier=KeywordClassifier(),
                           refund_agent=RefundAgent(Broken()))
    r = Chat(bot, ctx).say("I want a refund of ₹499.")
    assert ctx.session.status == AWAITING_CONFIRMATION
    assert "[Agent] get_user_account" in r.trace


# ---------- Confirmation handling (6.7) ----------
def test_cancel_does_not_refund(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("I want a refund of ₹499.")
    r = chat.say("Cancel")
    assert ctx.session.status == REFUND_CANCELLED
    assert ctx.tools.calls == []
    assert recharge(ctx, "rch_457")["refund_status"] == "not_refunded"
    assert "Nothing has been changed" in r.text


@pytest.mark.parametrize("reply", ["maybe", "yes but wait", "I guess", "hmm", "what?",
                                   "Why is my internet slow?"])
def test_ambiguous_confirmation_asks_again(bot, ctx, reply):
    chat = Chat(bot, ctx)
    chat.say("I want a refund of ₹499.")
    r = chat.say(reply)
    assert ctx.session.status == AWAITING_CONFIRMATION
    assert r.confirmation is not None and "clear answer" in r.text
    assert ctx.tools.calls == []


@pytest.mark.parametrize("text,expected", [
    ("Yes, proceed", CONFIRM), ("yes", CONFIRM), ("Confirm", CONFIRM), ("go ahead", CONFIRM),
    ("Yes please proceed with the refund", CONFIRM), ("OK", CONFIRM),
    ("No", REJECT), ("Cancel", REJECT), ("No, don't proceed", REJECT), ("not now", REJECT),
    ("maybe", AMBIGUOUS), ("yes no", AMBIGUOUS), ("", AMBIGUOUS), ("sure, but later", AMBIGUOUS),
])
def test_parse_confirmation(text, expected):
    assert parse_confirmation(text) == expected


def test_clear_conversation_drops_pending_refund(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("I want a refund of ₹499.")
    ctx.session.reset()                                  # what the Clear button does
    chat.messages.clear()
    r = chat.say("Yes, proceed")                         # no longer confirms anything
    assert ctx.tools.calls == []
    assert r.intent == REQUIREMENT_INQUIRY


def test_requirement_inquiry_drops_pending_amount_question(bot, ctx):
    chat = Chat(bot, ctx)
    chat.say("I want a refund.")
    assert ctx.session.status == AWAITING_AMOUNT
    chat.say("How do I activate roaming?")
    assert ctx.session.status is None
    r = chat.say("yes")
    assert r.intent == REQUIREMENT_INQUIRY and ctx.tools.calls == []


# ---------- Tool-level guardrails ----------
def test_process_and_submit_not_exposed_to_llm():
    names = {t["function"]["name"] for t in LLM_TOOLS}
    assert names == {"get_user_account", "validate_refund"}               # FR-23a


def test_process_refund_refuses_above_limit_and_ineligible(ctx):
    assert not ctx.tools.process_refund("user_123", 999)["success"]       # GR-08
    assert not ctx.tools.process_refund("user_123", 2000)["success"]      # GR-04
    assert recharge(ctx, "rch_456")["refund_status"] == "not_refunded"


@pytest.mark.parametrize("amount,eligible,rid,reason", [
    (499, True, "rch_457", "eligible"),
    (999, True, "rch_456", "eligible"),
    (300, True, "rch_457", "eligible"),        # partial: tightest covering recharge
    (0, False, None, "greater than zero"),
    (-5, False, None, "greater than zero"),
    (2000, False, "rch_456", "exceeds"),
])
def test_validate_refund_rules(ctx, amount, eligible, rid, reason):
    v = ctx.tools.validate_refund("user_123", amount)
    assert v["eligible"] is eligible and v["recharge_id"] == rid and reason in v["reason"]


def test_exact_match_identifies_refunded_recharge(ctx):
    """FR-27a: ₹499 means rch_457 even when it's refunded (not 'fall back' to rch_456)."""
    recharge(ctx, "rch_457")["refund_status"] = "refunded"
    v = ctx.tools.validate_refund("user_123", 499)
    assert not v["eligible"] and v["recharge_id"] == "rch_457"


def test_pending_recharge_rejected(ctx):
    recharge(ctx, "rch_457")["recharge_status"] = "pending"
    assert "not completed" in ctx.tools.validate_refund("user_123", 499)["reason"]


def test_account_tool_returns_copy(ctx):
    acct = ctx.tools.get_user_account("user_123")
    acct["recharges"][0]["refund_status"] = "refunded"
    assert recharge(ctx, "rch_456")["refund_status"] == "not_refunded"


def test_sessions_do_not_share_data(bot):
    a = RefundContext(user_id="user_123", tools=SpyTools(new_store()))
    b = RefundContext(user_id="user_123", tools=SpyTools(new_store()))
    chat = Chat(bot, a)
    chat.say("I want a refund of ₹499.")
    chat.say("Yes, proceed")
    assert recharge(b, "rch_457")["refund_status"] == "not_refunded"


def test_amounts_in():
    assert amounts_in("refund ₹2,000 for rch_457") == {2000.0}
    assert amounts_in("Rs. 499.50 or 999") == {499.5, 999.0}
