"""The roaming book: every roaming figure and decision, computed
here in code — the roaming sibling of the billing book.

- **Trip quotes** — a trip priced two ways from the rate table
  and the customer's own usage profile: pay-as-you-go, and the
  Roam Pass with its daily allowance and the overage beyond it.
  The recommendation is whichever total is smaller; the figures
  are Decimal end to end and no price ever comes from the model.
- **Bill-shock guard** — a quote whose pay-as-you-go projection
  passes the policy threshold. The specialist leads with the
  projection and withholds any chargeable proposal until the
  customer has seen it; the guard is a property of the quote,
  decided here.
- **Fair-use progress** — usage against the ceiling as a
  percentage, the pace the trip is running at, and the projected
  overage cost if that pace holds to the end of a standard
  10-day trip window (the Roam Pass's own length).
- **Notices** — at 80% and at 100% of the ceiling the warning
  becomes a stored RoamingNotice carrying the computed figures
  and the exact summary the customer is shown. A 100% crossing
  with no earlier warning on record is flagged on the notice
  itself: the crossing is a policy event, not just a state.
- **Dispute evidence** — a disputed roaming charge is decided
  from records: an active pass window covering the bill's cycle
  makes it stand, a fair-use warning given before the bill was
  issued makes it stand, and usage past the ceiling with no
  warning on record makes it creditable in full.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from .billing_book import RoamingVerdict, line_evidence
from .clock import today
from .config import Policy
from .money import fmt, money
from .models import (
    Account,
    ActionKind,
    AuditEvent,
    Bill,
    RoamingNotice,
)

ZERO = Decimal("0")

# Destination spellings customers use that are not the rate
# table's country names.
ALIASES = {
    "uae": "United Arab Emirates",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
}


def _num(value: Decimal) -> str:
    """A Decimal as a customer reads it: 7.2, 26, 104 — trailing
    zeros trimmed, never scientific notation."""
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def gb(value: Decimal) -> str:
    return f"{_num(value)} GB"


def rate_for(store, destination: str):
    """The rate row for a destination, by its table name or a
    known alias; None when Northbeam publishes no rate for it."""
    if not destination:
        return None
    row = store.get_roaming_rate(destination)
    if row is not None:
        return row
    alias = ALIASES.get(destination.strip().lower())
    return store.get_roaming_rate(alias) if alias else None


def find_destination(store, text: str):
    """The rate row a message names, if it names one: a table
    destination appearing in the text, or an alias for one."""
    lowered = text.lower()
    for row in store.list_roaming_rates():
        if row.destination.lower() in lowered:
            return row
    for alias, destination in ALIASES.items():
        if alias in lowered.split() or f" {alias} " in f" {lowered} ":
            return store.get_roaming_rate(destination)
    return None


# ---------------------------------------------------------------------------
# Trip quotes
# ---------------------------------------------------------------------------


@dataclass
class TripQuote:
    destination: str
    days: int
    per_gb: Decimal
    expected_gb: Decimal  # days × the customer's typical daily use
    payg_cost: Decimal
    pass_name: str = ""
    pass_price: Decimal = ZERO
    pass_allowance_gb: Decimal = ZERO  # included across the trip
    pass_overage_gb: Decimal = ZERO
    pass_overage_cost: Decimal = ZERO
    pass_cost: Decimal = ZERO  # price + overage, the honest pass total
    cheaper: str = "payg"  # pass | payg | equal
    saving: Decimal = ZERO
    shock: bool = False  # pay-as-you-go projection past the threshold

    @property
    def has_pass(self) -> bool:
        return bool(self.pass_name)


def _pass_offer(store, account: Account):
    """The roaming pass this account may buy, if one exists and
    the plan is eligible for it."""
    for offer in store.list_offers():
        if offer.kind == "roaming_pass" \
                and account.plan_id in offer.eligible_plan_ids:
            return offer
    return None


def quote_trip(store, account: Account, destination: str, days: int,
               policy: Policy) -> TripQuote | None:
    """Price a trip both ways, in Decimal, from the rate table
    and the account's usage profile. None when there is no rate
    for the destination — the desk says so rather than guessing."""
    rate = rate_for(store, destination)
    if rate is None or days <= 0:
        return None
    typical = Decimal(str(account.typical_daily_data_gb))
    expected = typical * Decimal(days)
    quote = TripQuote(
        destination=rate.destination, days=days, per_gb=rate.per_gb,
        expected_gb=expected, payg_cost=money(expected * rate.per_gb))
    offer = _pass_offer(store, account)
    if offer is not None:
        per_day = offer.included_gb_per_day or ZERO
        allowance = per_day * Decimal(days)
        overage = max(ZERO, expected - allowance)
        quote.pass_name = offer.name
        quote.pass_price = offer.one_off_price
        quote.pass_allowance_gb = allowance
        quote.pass_overage_gb = overage
        quote.pass_overage_cost = money(overage * rate.per_gb)
        quote.pass_cost = money(offer.one_off_price + quote.pass_overage_cost)
        if quote.pass_cost < quote.payg_cost:
            quote.cheaper = "pass"
        elif quote.pass_cost == quote.payg_cost:
            quote.cheaper = "equal"
        quote.saving = abs(quote.payg_cost - quote.pass_cost)
    quote.shock = quote.payg_cost > policy.roaming_bill_shock_amount
    return quote


# ---------------------------------------------------------------------------
# Fair-use progress, projection, notices
# ---------------------------------------------------------------------------


def fair_use_progress(account: Account, policy: Policy) -> dict:
    """Usage against the ceiling: the figures every fair-use
    statement is composed from."""
    used = Decimal(str(account.roaming_gb_used))
    cap = Decimal(policy.roaming_fair_use_gb)
    pct = (used / cap * 100).quantize(Decimal("0.1")) if cap > 0 else ZERO
    return {"used": used, "cap": cap, "pct": pct}


def days_roaming(account: Account, on: date | None = None) -> int:
    """Whole days the current trip has run, counting both ends;
    one when the start isn't on the record."""
    if not account.roaming_since:
        return 1
    try:
        start = date.fromisoformat(account.roaming_since)
    except ValueError:
        return 1
    return max(1, ((on or today()) - start).days + 1)


def projected_overage(store, account: Account,
                      policy: Policy) -> dict | None:
    """Where this trip lands if the pace holds: the pace so far
    (usage over days roaming) carried to the end of a standard
    trip window — the Roam Pass's duration — and the part of it
    past the fair-use ceiling, priced at the destination rate.
    None when there is no trip or no rate to price it with."""
    if not account.roaming_country:
        return None
    rate = rate_for(store, account.roaming_country)
    if rate is None:
        return None
    progress = fair_use_progress(account, policy)
    pace = progress["used"] / Decimal(days_roaming(account))
    window = 10
    offer = _pass_offer(store, account)
    if offer is not None and offer.duration_days:
        window = offer.duration_days
    projected_total = pace * Decimal(window)
    overage = max(ZERO, projected_total - progress["cap"])
    return {
        "pace_per_day": pace, "window_days": window,
        "projected_total_gb": projected_total,
        "overage_gb": overage,
        "overage_cost": money(overage * rate.per_gb),
        "per_gb": rate.per_gb, "destination": rate.destination,
    }


_MARKS = (("fair_use_80", Decimal("80")), ("fair_use_100", Decimal("100")))


def notice_summary(kind: str, progress: dict, projection: dict | None,
                   unwarned: bool = False) -> str:
    """The proactive summary text, composed from the computed
    figures — the same text the notice stores and the customer
    reads."""
    used, cap, pct = progress["used"], progress["cap"], progress["pct"]
    if kind == "fair_use_100":
        lead = (f"Fair-use warning: you have used {gb(used)} roaming this "
                f"trip, past the {gb(cap)} fair-use ceiling ({_num(pct)}%). "
                "Speeds are reduced now, and further data is charged per GB.")
    else:
        lead = (f"Fair-use heads-up: you have used {gb(used)} roaming this "
                f"trip — {_num(pct)}% of the {gb(cap)} fair-use ceiling.")
    if projection is not None and projection["overage_cost"] > 0:
        lead += (f" At the pace you're going ({_num(projection['pace_per_day'])} "
                 f"GB a day), you're projected to reach "
                 f"{gb(projection['projected_total_gb'])} over a "
                 f"{projection['window_days']}-day trip — about "
                 f"{gb(projection['overage_gb'])} past the ceiling, roughly "
                 f"{fmt(projection['overage_cost'])} at "
                 f"{projection['destination']} rates.")
    return lead


def sync_fair_use_notices(store, account: Account, policy: Policy,
                          conversation_id: str = "") -> list[RoamingNotice]:
    """Raise the fair-use notices the account's usage has earned
    and not yet been given — at 80% of the ceiling, and at 100%.
    Idempotent: a mark already on record for this trip is never
    raised twice. Returns the notices created by this call."""
    if not account.roaming_country:
        return []
    progress = fair_use_progress(account, policy)
    if progress["used"] <= 0:
        return []
    existing = [n for n in store.list_roaming_notices(account.account_number)
                if n.trip_start == account.roaming_since]
    prior_any = bool(existing)
    have = {n.kind for n in existing}
    projection = projected_overage(store, account, policy)
    created: list[RoamingNotice] = []
    for kind, mark in _MARKS:
        if progress["pct"] < mark or kind in have:
            continue
        unwarned = kind == "fair_use_100" and not prior_any
        notice = RoamingNotice(
            notice_id=f"NTF-{uuid.uuid4().hex[:10]}",
            account_number=account.account_number,
            kind=kind,
            destination=account.roaming_country,
            trip_start=account.roaming_since,
            used_gb=progress["used"], cap_gb=progress["cap"],
            pct=progress["pct"],
            projected_overage_cost=(projection["overage_cost"]
                                    if projection else None),
            summary=notice_summary(kind, progress, projection, unwarned),
            conversation_id=conversation_id,
            unwarned_crossing=unwarned,
            created_date=today().isoformat(),
        )
        store.save_roaming_notice(notice)
        have.add(kind)
        created.append(notice)
        if conversation_id:
            store.save_audit(AuditEvent(
                event_id=f"EVT-{uuid.uuid4().hex[:10]}",
                conversation_id=conversation_id, kind="roaming",
                detail={"event": "fair_use_notice", "kind": kind,
                        "notice_id": notice.notice_id,
                        "pct": str(progress["pct"]),
                        "unwarned_crossing": unwarned},
                turn=0))
    return created


def record_roaming_usage(store, account_number: str, gb_used: Decimal,
                         policy: Policy,
                         conversation_id: str = "") -> tuple[Account, list]:
    """The network-side write: usage accrues on the account
    record, and any fair-use mark the new total crosses raises
    its notice in the same act."""
    account = store.get_account(account_number)
    if account is None:
        return None, []
    account.roaming_gb_used = (Decimal(str(account.roaming_gb_used))
                                + Decimal(str(gb_used)))
    store.save_account(account)
    notices = sync_fair_use_notices(store, account, policy, conversation_id)
    return account, notices


# ---------------------------------------------------------------------------
# Dispute evidence — read from records, never from the bill's prose
# ---------------------------------------------------------------------------


def _pass_windows(store, account: Account) -> list[tuple[date, date, int]]:
    """(first day, last day, days) of every roaming pass window
    on the account's action log — executed activations only."""
    windows = []
    for record in store.list_actions():
        if record.account_number != account.account_number:
            continue
        if record.kind != ActionKind.ACTIVATE_OFFER \
                or record.status != "executed":
            continue
        offer = store.get_offer(record.payload.get("offer_id", ""))
        if offer is None or offer.kind != "roaming_pass":
            continue
        try:
            start = date.fromisoformat(record.created_date)
        except (TypeError, ValueError):
            continue
        days = offer.duration_days or 0
        windows.append((start, start + timedelta(days=days - 1), days))
    return windows


def roaming_dispute_verdicts(store, account: Account | None, bill: Bill,
                             policy: Policy,
                             conversation_id: str = "") -> dict[str, RoamingVerdict]:
    """Decide the bill's roaming lines from the records, per
    POL-ROAM-03. A line the bill's own markers already convict
    (a wrong rate, a duplicate) is left to them — record evidence
    never overrides the bill's own evidence. Warnings raised
    inside the conversation doing the disputing do not count as
    a prior warning: they were produced by this very contact,
    after the usage was billed."""
    verdicts: dict[str, RoamingVerdict] = {}
    if account is None or bill is None:
        return verdicts
    cycle_start = date.fromisoformat(bill.cycle_start)
    cycle_end = date.fromisoformat(bill.cycle_end)
    windows = _pass_windows(store, account)
    progress = fair_use_progress(account, policy)
    warnings = [n for n in store.list_roaming_notices(account.account_number)
                if n.conversation_id != conversation_id
                and n.created_date
                and date.fromisoformat(n.created_date) <= cycle_end]
    for charge in bill.charges:
        if charge.kind != "roaming":
            continue
        if line_evidence(charge)[0] > 0:
            continue  # the bill's own evidence decides this line
        covering = next(((s, e, d) for s, e, d in windows
                         if s <= cycle_end and e >= cycle_start), None)
        if covering is not None:
            start, _end, days = covering
            verdicts[charge.charge_id] = RoamingVerdict(
                charge_id=charge.charge_id, creditable=ZERO,
                evidence=(f"a Roam Pass was active for this trip "
                          f"(activated {start.isoformat()}, covering "
                          f"{days} days)"),
                decline_reason=(
                    "the charge falls inside the Roam Pass window on "
                    "your account — data past the pass's daily 1 GB is "
                    "billed per GB, so this one stands"))
            continue
        if warnings:
            latest = warnings[-1]
            verdicts[charge.charge_id] = RoamingVerdict(
                charge_id=charge.charge_id, creditable=ZERO,
                evidence=(f"a fair-use warning was given on "
                          f"{latest.created_date}, when usage reached "
                          f"{_num(latest.pct)}% of the "
                          f"{gb(latest.cap_gb)} fair-use ceiling"),
                decline_reason=(
                    f"we warned you on {latest.created_date} as the "
                    "usage passed the fair-use marks, before this "
                    "charge was billed, so it stands"))
            continue
        if progress["used"] > progress["cap"]:
            verdicts[charge.charge_id] = RoamingVerdict(
                charge_id=charge.charge_id, creditable=charge.amount,
                evidence=(f"roaming usage of {gb(progress['used'])} passed "
                          f"the {gb(progress['cap'])} fair-use ceiling "
                          "with no fair-use warning on record"))
    return verdicts
