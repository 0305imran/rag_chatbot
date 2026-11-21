"""Refund tools with deterministic business rules (agent PRD 6.4–6.7).

Only ``get_user_account`` and ``validate_refund`` are ever offered to the LLM.
``process_refund`` and ``submit_to_support`` are called by application code only
(FR-23a), and both re-check the rules themselves so they can't be misused.
Everything here is simulated: no payment, billing, or CRM API is called (FR-35, GR-07).
"""
import copy
import uuid

from config import REFUND_APPROVAL_LIMIT, REFUND_WINDOW_DAYS


def fmt_inr(amount: float) -> str:
    return f"₹{amount:,.0f}" if float(amount).is_integer() else f"₹{amount:,.2f}"


def _recharge_problem(recharge: dict) -> str | None:
    """Why this recharge can't be refunded at all, or None if it can (FR-24, 25, 28)."""
    rid, amt = recharge["recharge_id"], fmt_inr(recharge["recharge_amount"])
    if recharge["recharge_status"] != "completed":
        return f"Your {amt} recharge ({rid}) is not completed (status: {recharge['recharge_status']})."
    if recharge["refund_status"] == "refunded":
        return f"Your {amt} recharge ({rid}) has already been refunded."
    if recharge["recharge_age_days"] > REFUND_WINDOW_DAYS:
        return (f"Your {amt} recharge ({rid}) was made {recharge['recharge_age_days']} days ago, "
                f"outside the {REFUND_WINDOW_DAYS}-day refund window.")
    return None


def select_recharge(recharges: list[dict], amount: float) -> dict | None:
    """Deterministically pick the recharge a requested amount applies to (FR-27a).

    1. A recharge whose amount matches exactly (even if already refunded); among several,
       a refundable one first, then the most recent.
    2. Otherwise the smallest refundable recharge that covers the amount.
    3. Otherwise the smallest recharge that covers the amount (so its problem is reported).
    4. Otherwise the largest recharge (the request exceeds every recharge).
    """
    if not recharges:
        return None

    def rank(r):
        return (_recharge_problem(r) is not None, r["recharge_age_days"])

    exact = [r for r in recharges if r["recharge_amount"] == amount]
    if exact:
        return min(exact, key=rank)
    covering = [r for r in recharges if r["recharge_amount"] >= amount]
    if covering:
        return min(covering, key=lambda r: (_recharge_problem(r) is not None,
                                            r["recharge_amount"], r["recharge_age_days"]))
    return max(recharges, key=lambda r: (r["recharge_amount"], -r["recharge_age_days"]))


class RefundTools:
    """Tool implementations bound to one session's simulated data store."""

    def __init__(self, store: dict):
        self.store = store

    def _user(self, user_id: str) -> dict | None:
        return self.store["users"].get(user_id)

    # ---------- exposed to the LLM ----------
    def get_user_account(self, user_id: str) -> dict:
        """FR-19: the customer's plan and recharge history (a copy; callers can't mutate it)."""
        user = self._user(user_id)
        if user is None:
            return {"found": False, "user_id": user_id}
        return {"found": True, "user_id": user_id, **copy.deepcopy(user)}

    def validate_refund(self, user_id: str, amount: float) -> dict:
        """FR-20, FR-24..FR-30: deterministic eligibility check. Never mutates state."""
        result = {"eligible": False, "requested_amount": amount,
                  "recharge_amount": None, "recharge_id": None}
        user = self._user(user_id)
        if user is None:
            return {**result, "reason": "No account found for this user."}
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {**result, "reason": "The refund amount is not a valid number."}
        if amount <= 0:                                                   # FR-26
            return {**result, "reason": "The refund amount must be greater than zero."}

        recharge = select_recharge(user["recharges"], amount)
        if recharge is None:
            return {**result, "reason": "There are no recharges on your account."}
        result.update(recharge_amount=recharge["recharge_amount"],
                      recharge_id=recharge["recharge_id"])

        problem = _recharge_problem(recharge)
        if problem:
            return {**result, "reason": problem}
        if amount > recharge["recharge_amount"]:                          # FR-27, GR-04
            return {**result, "reason": (
                f"The requested amount of {fmt_inr(amount)} exceeds your recharge amount "
                f"of {fmt_inr(recharge['recharge_amount'])}.")}
        return {**result, "eligible": True, "reason": "Refund is eligible."}

    # ---------- application-only (FR-23a) ----------
    def process_refund(self, user_id: str, amount: float, *, recharge_id: str | None = None) -> dict:
        """FR-21, FR-33, FR-34: simulated refund. Re-validates and enforces the approval limit."""
        check = self.validate_refund(user_id, amount)
        if not check["eligible"]:
            return {"success": False, "amount": amount, "reason": check["reason"]}
        if recharge_id is not None and check["recharge_id"] != recharge_id:
            return {"success": False, "amount": amount,
                    "reason": "The recharge for this refund changed; please start again."}
        if float(amount) > REFUND_APPROVAL_LIMIT:                         # GR-08
            return {"success": False, "amount": amount,
                    "reason": f"Refunds above {fmt_inr(REFUND_APPROVAL_LIMIT)} need Support Team review."}

        recharge = next(r for r in self._user(user_id)["recharges"]
                        if r["recharge_id"] == check["recharge_id"])
        refund_id = f"ref_{uuid.uuid4().hex[:8]}"
        recharge.update(refund_status="refunded", refund_id=refund_id, refunded_amount=amount)  # FR-28a
        return {"success": True, "refund_id": refund_id, "amount": amount,
                "recharge_id": recharge["recharge_id"], "status": "refunded"}

    def submit_to_support(self, user_id: str, amount: float) -> dict:
        """FR-21a, FR-38: queue the request for human review. Refund status is unchanged."""
        check = self.validate_refund(user_id, amount)
        ticket = {
            "ticket_id": f"sup_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "amount": amount,
            "recharge_id": check["recharge_id"],
            "status": "submitted_for_review",
        }
        self.store["support_queue"].append(ticket)
        return {"submitted": True, **ticket}
