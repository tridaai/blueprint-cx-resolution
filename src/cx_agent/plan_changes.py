"""Plan-change transactions: the quote, the fingerprint, the
reversal.

A plan change is the one conversation in this book that moves a
customer between price points, so it gets transaction discipline
the other desks don't need:

- The **quote** is computed from the account's real cycle calendar
  (cycle_start / cycle_end on the record, the shared clock for
  "today") by money.plan_change_for_cycle — credit for the unused
  days of the old plan, charge for the remaining days of the new
  one, the net, and what the next bill is projected to carry.
- **Effective dates are policy, not negotiation.** An upgrade (the
  new price is higher) starts now, prorated. A downgrade starts at
  the next cycle boundary with no proration either way — the
  customer keeps the plan they've paid for until then, and the
  change is scheduled on the account. A move to the same price
  starts now and moves no money.
- A proposal is a **versioned record stamped with a fingerprint**
  of the account state it was priced against — plan, cycle dates,
  balance, any scheduled change. Execution re-derives the
  fingerprint from the live account and refuses a mismatch: the
  world moved, so the customer confirms fresh figures, never
  stale ones.
- A completed change can be **reversed inside the policy window**
  (plan_change_reversal_days — the same day, by default) as a
  compensating pair of records: one restores the plan (or cancels
  the schedule), one unwinds the money. The original record is
  never edited; the pair points back at it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from .clock import today
from .money import plan_change_for_cycle
from .models import Account, ActionKind, ActionRecord, Plan


@dataclass
class ChangeQuote:
    direction: str  # upgrade | downgrade | lateral
    effective: str  # immediate | next_cycle
    effective_date: str  # ISO date the change takes effect
    old_plan_id: str
    new_plan_id: str
    old_price: Decimal  # the price the customer pays today
    new_price: Decimal  # the effective price of the target plan
    credit: Decimal  # for the unused days of the old plan
    charge: Decimal  # for the remaining days of the new plan
    net: Decimal  # charge - credit; + due, - credit balance
    days_remaining: int
    days_in_cycle: int
    cycle_start: str
    cycle_end: str
    # What the change puts on the next bill beyond what's already
    # owed there: the net adjustment plus the first full month at
    # the new price (for a downgrade, just the first month — the
    # adjustment is zero by the time it starts).
    next_bill_impact: Decimal


def quote_change(account: Account, current: Plan, target: Plan,
                 new_price: Decimal, on: date | None = None) -> ChangeQuote:
    """Price a change of plan against the account's cycle calendar.

    `new_price` is the effective price of the target (a loyalty
    price when one applies); direction is read off it against the
    price the customer actually pays on the current plan."""
    on = on or today()
    cycle_start = date.fromisoformat(account.cycle_start)
    cycle_end = date.fromisoformat(account.cycle_end)
    if new_price > current.monthly_price:
        direction = "upgrade"
    elif new_price < current.monthly_price:
        direction = "downgrade"
    else:
        direction = "lateral"
    if direction == "downgrade":
        # Next cycle boundary: no proration, nothing moves now.
        return ChangeQuote(
            direction=direction, effective="next_cycle",
            effective_date=(cycle_end + timedelta(days=1)).isoformat(),
            old_plan_id=current.plan_id, new_plan_id=target.plan_id,
            old_price=current.monthly_price, new_price=new_price,
            credit=Decimal("0"), charge=Decimal("0"), net=Decimal("0"),
            days_remaining=(cycle_end - on).days,
            days_in_cycle=(cycle_end - cycle_start).days + 1,
            cycle_start=account.cycle_start, cycle_end=account.cycle_end,
            next_bill_impact=new_price)
    story = plan_change_for_cycle(current.monthly_price, new_price,
                                  cycle_start, cycle_end, on)
    return ChangeQuote(
        direction=direction, effective="immediate",
        effective_date=on.isoformat(),
        old_plan_id=current.plan_id, new_plan_id=target.plan_id,
        old_price=current.monthly_price, new_price=new_price,
        credit=story["credit_for_old_plan"],
        charge=story["charge_for_new_plan"], net=story["net"],
        days_remaining=story["days_remaining"],
        days_in_cycle=story["days_in_cycle"],
        cycle_start=account.cycle_start, cycle_end=account.cycle_end,
        next_bill_impact=story["net"] + new_price)


# ---------------------------------------------------------------------------
# The fingerprint: what a proposal was priced against
# ---------------------------------------------------------------------------


def fingerprint_parts(account: Account) -> dict:
    """The account state a plan proposal's figures rest on. Every
    part is stamped on the proposal in plain sight; the fingerprint
    is their hash, and execution compares the two."""
    return {
        "plan_id": account.plan_id,
        "cycle_start": account.cycle_start,
        "cycle_end": account.cycle_end,
        "balance_due": str(account.balance_due),
        "pending_plan_id": account.pending_plan_id or "",
    }


def account_fingerprint(account: Account) -> str:
    parts = fingerprint_parts(account)
    joined = "|".join(f"{key}={parts[key]}" for key in sorted(parts))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def fingerprint_diff(account: Account, stamped: dict) -> list[str]:
    """What moved between the stamped state and the live account,
    in plain words — the stale explanation names these."""
    now = fingerprint_parts(account)
    labels = {
        "plan_id": "your plan changed",
        "cycle_start": "your billing cycle moved",
        "cycle_end": "your billing cycle moved",
        "balance_due": "your balance changed",
        "pending_plan_id": "a scheduled plan change appeared",
    }
    moved = []
    for key, label in labels.items():
        if stamped.get(key) != now.get(key) and label not in moved:
            moved.append(label)
    return moved


# ---------------------------------------------------------------------------
# Reading the change history: the reversal's raw material
# ---------------------------------------------------------------------------


def change_records(store, account_number: str) -> list[ActionRecord]:
    """Every plan-change record on the account, oldest first —
    proposals, executions, and reversal legs alike."""
    records = [r for r in store.list_actions()
               if r.kind == ActionKind.CHANGE_PLAN
               and r.account_number == account_number]
    return sorted(records, key=lambda r: (r.created_date, r.turn))


def latest_executed_change(store, account_number: str) -> ActionRecord | None:
    """The most recent executed plan change that is itself an
    original change — reversal legs point back at their original
    (payload reversal_of) and are never a reversal target."""
    originals = [r for r in change_records(store, account_number)
                 if r.status == "executed"
                 and not r.payload.get("reversal_of")]
    return originals[-1] if originals else None


def reversal_of(store, record: ActionRecord) -> list[ActionRecord]:
    """The executed compensating pair that reversed `record`, if
    any — two records, both stamped with its id."""
    return [r for r in change_records(store, record.account_number)
            if r.status == "executed"
            and r.payload.get("reversal_of") == record.action_id]


def within_reversal_window(record: ActionRecord, policy,
                           on: date | None = None) -> bool:
    """The reversal window, in code: the change's own day counts,
    plus plan_change_reversal_days either side of it the policy
    allows (zero — the same day only — by default)."""
    on = on or today()
    made = date.fromisoformat(record.created_date)
    return 0 <= (on - made).days <= policy.plan_change_reversal_days
