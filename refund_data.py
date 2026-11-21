"""Hard-coded, simulated customer accounts for the refund agent (agent PRD FR-14..FR-18a).

No real CRM or billing system is involved. Each UI/CLI session works on its own deep
copy (``new_store``) so a simulated refund in one session never leaks into another.
"""
import copy

USERS = {
    # Demonstration user (FR-15, FR-18a): two recent recharges, ₹999 and ₹499.
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    },
    # Extra demo account for Scenario 3 (recharge outside the 7-day window).
    "user_789": {
        "name": "Priya",
        "plan": "Basic",
        "plan_amount": 299,
        "recharges": [
            {
                "recharge_id": "rch_901",
                "recharge_amount": 299,
                "recharge_status": "completed",
                "recharge_age_days": 15,
                "refund_status": "not_refunded",
            },
        ],
    },
}


def new_store() -> dict:
    """Fresh, mutable copy of the simulated account data."""
    return {"users": copy.deepcopy(USERS), "support_queue": []}
