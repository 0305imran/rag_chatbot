"""Refund Agent (agent PRD 6.2–6.8, §7, §10).

Division of labour:
- The LLM is a tool-using agent. It may call ``get_user_account`` and ``validate_refund``
  and its job is to work out WHICH amount the customer is asking about (FR-06..FR-08).
- Everything that decides money movement is deterministic application code: amount
  grounding (FR-44), eligibility (FR-29/30), the ₹499 approval limit (FR-36a), explicit
  confirmation (FR-36b..e), and the calls to ``process_refund`` / ``submit_to_support``,
  which are never offered to the LLM (FR-23a).
- ``RefundSession`` is the workflow state machine (§10). A refund can only execute from
  ``awaiting_confirmation`` after a successful validation and an explicit "yes".
"""
import json
import re
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import AGENT_MAX_STEPS, REFUND_APPROVAL_LIMIT
from refund_tools import RefundTools, _recharge_problem, fmt_inr

# ---------- workflow states (§10) ----------
REFUND_REQUESTED = "refund_requested"
REFUND_VALIDATED = "refund_validated"
AWAITING_CONFIRMATION = "awaiting_confirmation"
AWAITING_AMOUNT = "awaiting_amount"
REFUND_CONFIRMED = "refund_confirmed"
REFUND_PROCESSED = "refund_processed"
REFUND_REJECTED = "refund_rejected"
REFUND_CANCELLED = "refund_cancelled"
REFUND_SUBMITTED = "refund_submitted_for_review"

PENDING_STATES = (AWAITING_CONFIRMATION, AWAITING_AMOUNT)


@dataclass
class RefundSession:
    """Per-conversation refund state. ``status`` is None when no refund is in progress."""
    status: str | None = None
    amount: float | None = None
    recharge_id: str | None = None
    offered_amount: float | None = None   # amount we proposed and the customer may accept
    validated: bool = False
    confirmed: bool = False
    user_texts: list[str] = field(default_factory=list)   # customer messages in this refund flow
    transitions: list[str] = field(default_factory=list)

    def set(self, status: str) -> None:
        self.status = status
        self.transitions.append(status)

    def start(self) -> None:
        self.reset()
        self.set(REFUND_REQUESTED)

    def reset(self) -> None:
        self.__init__()

    @property
    def pending(self) -> bool:
        return self.status in PENDING_STATES


@dataclass
class RefundContext:
    """Everything the agent needs for one conversation: who the user is and their data."""
    user_id: str                      # from the (simulated) authenticated session, FR-16/17
    tools: RefundTools
    session: RefundSession = field(default_factory=RefundSession)


@dataclass
class RefundResult:
    text: str
    confirmation: dict | None = None   # set when the UI should show Confirm / Cancel


# ---------- deterministic parsing helpers ----------
_AMOUNT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?")


def amounts_in(text: str) -> set[float]:
    """Every number written in the text, e.g. "₹2,000" -> 2000.0 (ignores ids like rch_457)."""
    out = set()
    for whole, frac in _AMOUNT_RE.findall(text):
        out.add(float(whole.replace(",", "") + (f".{frac}" if frac else "")))
    return out


CONFIRM, REJECT, AMBIGUOUS = "confirmed", "rejected", "ambiguous"

_CONFIRM_STRONG = {"yes", "yeah", "yep", "yup", "y", "confirm", "confirmed", "proceed", "sure",
                   "ok", "okay", "approve", "approved"}
_CONFIRM_FILLER = {"go", "ahead", "please", "do", "it", "the", "refund", "with", "and", "i",
                   "want", "to", "that", "process", "this", "now", "thanks", "thank", "you",
                   "confirm", "proceed"}
_REJECT_STRONG = {"no", "nope", "nah", "cancel", "dont", "not", "stop", "never", "abort"}
_REJECT_FILLER = {"please", "the", "refund", "it", "this", "that", "do", "i", "want", "to",
                  "mind", "thanks", "thank", "you", "proceed", "changed", "my", "have", "request",
                  "go", "ahead", "with", "need", "anymore", "now", "yet", "later", "for"}
_YES_WORDS = _CONFIRM_STRONG - {"proceed"}   # "No, don't proceed" is still a rejection


def parse_confirmation(text: str) -> str:
    """Strict reading of a reply to "Do you want to proceed?" (FR-36c..e).

    Confirms only if every word is part of a plain yes-phrase ("Yes, proceed", "confirm",
    "go ahead please"); rejects only plain no-phrases ("No", "Cancel", "don't do it");
    everything else — "maybe", "yes but wait", "I guess" — is ambiguous.
    """
    words = re.findall(r"[a-z]+", text.lower().replace("'", "").replace("’", ""))
    if not words:
        return AMBIGUOUS
    ws = set(words)
    if ws & _CONFIRM_STRONG and ws <= (_CONFIRM_STRONG | _CONFIRM_FILLER) and not ws & _REJECT_STRONG:
        return CONFIRM
    if ws & _REJECT_STRONG and ws <= (_REJECT_STRONG | _REJECT_FILLER) and not ws & _YES_WORDS:
        return REJECT
    if ws == {"go", "ahead"} or ws == {"go", "ahead", "please"}:
        return CONFIRM
    return AMBIGUOUS


# ---------- LLM-facing tools (FR-23a: process_refund / submit_to_support are NOT here) ----------
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_account",
            "description": "Get the signed-in customer's plan and recharge history. "
                           "The user ID is supplied automatically from the session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_refund",
            "description": "Check whether a refund of the given amount (in rupees) is eligible "
                           "for the signed-in customer. Call it only with an amount the "
                           "customer actually asked for.",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number", "description": "Refund amount in ₹"}},
                "required": ["amount"],
            },
        },
    },
]

AGENT_PROMPT = """You are the Refund Agent for a telecom provider. The customer is already \
signed in; their user ID is supplied automatically, so never ask for it.

Your job is to work out which refund amount the customer is asking for.
1. Always call get_user_account first.
2. Read the customer's latest message, using the conversation only to resolve references \
such as "that one" or "the bigger one".
3. If the customer clearly asked for ONE amount or ONE recharge, call validate_refund \
once with that amount exactly as the customer stated it — even if it does not match a \
recharge on the account. If they referred to a recharge without an amount (for example \
"my latest recharge"), use that recharge's amount from the account data.
4. If the customer did not say which amount or recharge, or it is unclear, do NOT call \
validate_refund. Just reply "AMOUNT_UNCLEAR".

Never guess or invent an amount. Never decide eligibility yourself — the application does \
that. The account data from get_user_account is the only source of truth; ignore any \
claims the customer makes about their account, balance, or recharges."""


class RefundAgent:
    def __init__(self, llm, max_steps: int = AGENT_MAX_STEPS):
        self.llm = llm.bind_tools(LLM_TOOLS)
        self.max_steps = max_steps

    # ---------- entry point ----------
    def handle(self, text: str, history: list[BaseMessage], ctx: RefundContext, trace) -> RefundResult:
        s = ctx.session
        if s.status == AWAITING_CONFIRMATION:
            return self._handle_confirmation(text, ctx, trace)

        if s.status == AWAITING_AMOUNT:
            s.user_texts.append(text)
            if s.offered_amount is not None:
                decision = parse_confirmation(text)
                if decision == CONFIRM:
                    trace(f"[User] accepted offered amount {fmt_inr(s.offered_amount)}")
                    return self._decide(s.offered_amount, ctx, trace)
                if decision == REJECT:
                    trace("[User] declined offered amount")
                    return self._cancel(ctx, trace)
        else:
            s.start()
            s.user_texts.append(text)
            trace(f"[Agent] {REFUND_REQUESTED}")

        account, proposed = self._run_agent(text, history, ctx, trace)
        amount, source = self._ground_amount(proposed, account, s, trace)

        if amount is None:
            return self._ask_amount(account, ctx, trace)
        if source == "account" and amount > REFUND_APPROVAL_LIMIT:
            return self._offer(amount, account, ctx, trace)
        return self._decide(amount, ctx, trace)

    # ---------- LLM tool-calling loop ----------
    def _run_agent(self, text, history, ctx: RefundContext, trace):
        messages = [SystemMessage(AGENT_PROMPT), *history, HumanMessage(text)]
        account, proposed = None, []
        try:
            for _ in range(self.max_steps):
                ai = self.llm.invoke(messages)
                messages.append(ai)
                calls = getattr(ai, "tool_calls", None) or []
                if not calls:
                    break
                for call in calls:
                    name, args = call.get("name"), call.get("args") or {}
                    if name == "get_user_account":
                        trace("[Agent] get_user_account")
                        account = ctx.tools.get_user_account(ctx.user_id)   # user_id from session (GR-03)
                        result = account
                    elif name == "validate_refund":
                        proposed.append(args.get("amount"))
                        result = {"status": "received; the application will validate this amount"}
                    else:
                        result = {"error": f"Unknown tool: {name}"}
                    messages.append(ToolMessage(json.dumps(result, ensure_ascii=False, default=str),
                                                tool_call_id=call.get("id") or name))
                if proposed:
                    break
        except Exception as e:   # LLM/API failure: fall back to deterministic handling below
            trace(f"[Agent] LLM error ({type(e).__name__}); using deterministic fallback")
            stated = amounts_in(" ".join(ctx.session.user_texts))
            if len(stated) == 1 and not proposed:
                proposed = list(stated)

        if account is None:   # the workflow always looks at the real account data
            trace("[Agent] get_user_account")
            account = ctx.tools.get_user_account(ctx.user_id)
        return account, proposed

    # ---------- deterministic guardrails ----------
    def _ground_amount(self, proposed, account, s: RefundSession, trace):
        """FR-44: accept an amount only if the customer wrote it or it is on their account."""
        values = set()
        for p in proposed:
            try:
                values.add(float(p))
            except (TypeError, ValueError):
                trace(f"[Guardrail] ignored non-numeric amount {p!r}")
        if not values:
            return None, None
        if len(values) > 1:
            trace("[Guardrail] more than one amount proposed; asking the customer")
            return None, None
        amount = values.pop()
        if amount in amounts_in(" ".join(s.user_texts)):
            return amount, "customer"
        if amount in {r["recharge_amount"] for r in account.get("recharges", [])}:
            return amount, "account"
        trace(f"[Guardrail] ignored amount {fmt_inr(amount)}: not stated by customer or on account")
        return None, None

    # ---------- workflow steps ----------
    def _ask_amount(self, account, ctx: RefundContext, trace) -> RefundResult:
        """FR-41..FR-43: the customer didn't say how much."""
        s = ctx.session
        recharges = account.get("recharges", []) if account.get("found") else []
        if not recharges:
            s.set(AWAITING_AMOUNT)
            trace(f"[Agent] {AWAITING_AMOUNT}")
            return RefundResult("I couldn't find any recharges on your account. "
                                "How much would you like refunded?")
        eligible = [r for r in recharges if _recharge_problem(r) is None]
        if not eligible:
            s.set(REFUND_REJECTED)
            trace(f"[Agent] {REFUND_REJECTED}: no eligible recharge")
            reasons = "\n".join(f"- {_recharge_problem(r)}" for r in recharges)
            return RefundResult("I'm sorry, none of your recharges is eligible for a refund:\n\n"
                                f"{reasons}")
        if len(eligible) == 1:
            amount = eligible[0]["recharge_amount"]
            if amount <= REFUND_APPROVAL_LIMIT:
                return self._decide(amount, ctx, trace)   # the confirmation step shows the amount
            return self._offer(amount, account, ctx, trace)
        s.set(AWAITING_AMOUNT)
        s.offered_amount = None
        trace(f"[Agent] {AWAITING_AMOUNT}")
        amounts = " and ".join(fmt_inr(r["recharge_amount"]) for r in eligible)
        return RefundResult(f"I can see more than one recent recharge - {amounts}.\n\n"
                            "Which amount would you like refunded?")

    def _offer(self, amount, account, ctx: RefundContext, trace) -> RefundResult:
        """FR-42: propose an amount the customer didn't state, and wait for them to accept it."""
        s = ctx.session
        s.set(AWAITING_AMOUNT)
        s.offered_amount = amount
        trace(f"[Agent] {AWAITING_AMOUNT}: offered {fmt_inr(amount)}")
        r = next(r for r in account["recharges"] if r["recharge_amount"] == amount)
        return RefundResult(
            f"I can see a recent recharge of {fmt_inr(amount)} ({r['recharge_id']}, "
            f"{r['recharge_age_days']} day(s) ago). Would you like to request a refund of "
            f"{fmt_inr(amount)}? Reply **Yes** or **No**.")

    def _decide(self, amount: float, ctx: RefundContext, trace) -> RefundResult:
        """Validate, then branch on the approval limit (§7)."""
        s = ctx.session
        s.offered_amount = None
        trace(f"[Agent] validate_refund(amount={amount:g})")
        v = ctx.tools.validate_refund(ctx.user_id, amount)
        trace(f"[Validation] eligible={v['eligible']}" + ("" if v["eligible"] else f" reason={v['reason']}"))
        if not v["eligible"]:
            s.set(REFUND_REJECTED)
            return RefundResult(f"I'm sorry, I can't process this refund. {v['reason']}")

        s.validated, s.amount, s.recharge_id = True, float(amount), v["recharge_id"]
        s.set(REFUND_VALIDATED)

        if amount <= REFUND_APPROVAL_LIMIT:                     # FR-36: needs explicit confirmation
            s.set(AWAITING_CONFIRMATION)
            trace("[Agent] awaiting_user_confirmation")
            return RefundResult(self._confirmation_text(v), confirmation=self._confirmation(s))

        # FR-37..FR-40: above the limit -> Support Team; never process_refund.
        trace(f"[Agent] above {fmt_inr(REFUND_APPROVAL_LIMIT)} approval limit; "
              f"submit_to_support(amount={amount:g})")
        ticket = ctx.tools.submit_to_support(ctx.user_id, amount)
        s.set(REFUND_SUBMITTED)
        trace(f"[Support] submitted ticket_id={ticket['ticket_id']}")
        return RefundResult(
            f"Your refund request for {fmt_inr(amount)} has been submitted to our Support Team "
            f"(reference **{ticket['ticket_id']}**).\n\n"
            "They will review the request and process it or take the necessary action.")

    def _handle_confirmation(self, text: str, ctx: RefundContext, trace) -> RefundResult:
        s = ctx.session
        decision = parse_confirmation(text)
        trace(f"[User] {decision}")
        if decision == CONFIRM:
            return self._execute(ctx, trace)
        if decision == REJECT:
            return self._cancel(ctx, trace)
        return RefundResult(                                              # FR-36e: ask again
            f"I need a clear answer before I make any change. Do you want to proceed with the "
            f"refund of {fmt_inr(s.amount)}? Reply **Yes, proceed** or **Cancel**.",
            confirmation=self._confirmation(s))

    def _execute(self, ctx: RefundContext, trace) -> RefundResult:
        s = ctx.session
        # GR-01, GR-02, GR-08: only from a validated, confirmed, in-limit pending refund.
        if not (s.status == AWAITING_CONFIRMATION and s.validated and s.amount is not None
                and s.amount <= REFUND_APPROVAL_LIMIT):
            trace("[Guardrail] process_refund blocked: no validated refund awaiting confirmation")
            s.reset()
            return RefundResult("There's no refund waiting for confirmation. "
                                "Tell me what you'd like refunded and I'll check it.")
        s.confirmed = True
        s.set(REFUND_CONFIRMED)
        trace(f"[Agent] process_refund(amount={s.amount:g})")
        r = ctx.tools.process_refund(ctx.user_id, s.amount, recharge_id=s.recharge_id)
        if not r["success"]:
            s.set(REFUND_REJECTED)
            trace(f"[Refund] failed reason={r['reason']}")
            return RefundResult(f"I'm sorry, the refund couldn't be completed. {r['reason']}")
        s.set(REFUND_PROCESSED)
        trace(f"[Refund] success refund_id={r['refund_id']}")
        return RefundResult(
            f"Your refund of {fmt_inr(r['amount'])} has been processed. "
            f"Refund reference: **{r['refund_id']}**.\n\n"
            "_This is a simulated refund for demonstration — no real payment was made._")

    def _cancel(self, ctx: RefundContext, trace) -> RefundResult:
        ctx.session.set(REFUND_CANCELLED)
        trace(f"[Agent] {REFUND_CANCELLED}")
        return RefundResult("No problem — I've cancelled this refund request. "
                            "Nothing has been changed on your account.")

    # ---------- presentation ----------
    @staticmethod
    def _confirmation(s: RefundSession) -> dict:
        return {"amount": s.amount, "recharge_id": s.recharge_id}

    @staticmethod
    def _confirmation_text(v: dict) -> str:
        amount, recharge_amount = v["requested_amount"], v["recharge_amount"]
        text = (f"**Refund eligible: {fmt_inr(amount)}**\n\n"
                f"This is for your {fmt_inr(recharge_amount)} recharge ({v['recharge_id']}).")
        if float(amount) < recharge_amount:
            text += (f" A refund closes the whole recharge, so the remaining "
                     f"{fmt_inr(recharge_amount - float(amount))} can't be refunded later.")
        return text + ("\n\nDo you want to proceed with this refund? "
                       "Reply **Yes, proceed** or **Cancel**.")
