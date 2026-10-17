"""The retention book: what happens when a customer says they
want to leave, written down as records and rules (round 14).

Three disciplines, all in code:

- **The case is a record.** The moment a customer expresses
  intent to cancel and the account is known, a RetentionCase is
  opened: the reason classified deterministically against the
  governed list below, the customer's words kept verbatim, and
  every later step — the offer, its answer, the cancellation,
  a fault routing — written onto that record. The report at the
  bottom of this module reads nothing but those records.
- **One save offer, built from real mechanisms.** At most one
  counter-offer per case, ever: the single-counter rule is
  checked when an offer is assembled and re-checked in
  actions.py at execution. The offer is never an invented
  discount — it is assembled from mechanisms already in the
  book: a plan downgrade priced by the plan-change machinery
  (plan_changes.quote_change, scheduled at the cycle boundary),
  a goodwill credit under the existing credit policy
  (billing_book.goodwill_offer, caps and rolling history
  included), or the loyalty upgrade price where the customer
  is eligible for it. Which mechanism answers depends on the
  recorded reason, in the order `assemble_save_offer` documents.
  A reason no mechanism honestly answers — moving house — gets
  no offer at all.
- **No dark patterns.** A customer who says they want no offers
  gets none: the instruction is recorded on the case and the
  flow goes straight to the cancellation. Declining the save
  offer does the same — the offer is never repeated, rephrased
  or replaced. A second explicit cancel request at any point
  completes the cancellation with no further retention steps.
  And the cancellation itself is governed: a confirmed
  cancel_account action, effective at the end of the current
  cycle (the customer keeps what they paid for), scheduled on
  the account the way a downgrade is scheduled.

A fault-blamed exit is not this desk's case to price: like a
fault-blamed downgrade (plan_changes / POL-PLAN-02), it follows
the billing desk's dispute path, and the case records that
routing and settles as escalated_retention_team — the outcome
for the cases policy puts with humans.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from . import billing_book, plan_changes
from .clock import today
from .money import fmt
from .models import (
    ActionKind,
    AuditEvent,
    Conversation,
    ProposedAction,
    RetentionCase,
)

# ---------------------------------------------------------------------------
# Governed vocabularies
# ---------------------------------------------------------------------------

# The reason list a case may carry, in the order the report
# presents them. Classification below never invents an eighth.
RETENTION_REASONS = (
    "price", "coverage", "service_quality", "moving_house",
    "competitor", "no_longer_needed", "other",
)

# Reason classes over the customer's words, in reading order:
# the first class that matches decides, so "too expensive, and
# I'm moving house anyway" is a price case (the words are kept
# verbatim either way; the class is what the offer reads).
_REASON_CLASSES: list[tuple[str, list[str]]] = [
    ("price", [
        "too expensive", "expensive", "too pricey", "costs too much",
        "cost too much", "paying too much", "too much money",
        "cheaper elsewhere", "the price",
    ]),
    ("moving_house", [
        "moving house", "moving home", "moving away", "relocating",
        "relocation", "emigrating", "leaving the country",
        "moving abroad", "moving overseas", "i'm moving",
        "i am moving", "we're moving", "we are moving",
    ]),
    ("competitor", [
        "another provider", "other provider", "competitor", "rival",
        "switching provider", "switched provider", "deal elsewhere",
        "offered me a deal", "better offer from",
    ]),
    ("coverage", [
        "coverage", "reception", "blackspot", "black spot",
    ]),
    ("service_quality", [
        "service quality", "poor service", "bad service",
        "terrible service", "awful service", "service is poor",
        "service is broken", "service is terrible",
        "unreliable", "let down", "fed up", "customer service",
    ]),
    ("no_longer_needed", [
        "no longer need", "don't need", "do not need", "not using",
        "no longer use", "don't use", "do not use", "no use for",
        "never use",
    ]),
]

# Words that say "cancel something else", not "cancel the
# account": autopay, a scheduled plan change. Those requests
# belong to their own desks and never open a retention case.
_CANCEL_EXCLUSIONS = (
    "autopay", "auto pay", "scheduled change", "cancel the change",
    "cancel that change", "cancel the reversal",
)

_CANCEL_PATTERNS = [
    r"\bcancel", r"\bclose (?:my|the) account\b",
    r"\bleave northbeam\b", r"\bleaving northbeam\b",
    r"\bwant to leave\b", r"\bi'?m leaving\b", r"\bi am leaving\b",
]

# The customer declining to be retained, in their own words.
# Any of these on the case means NO offer is assembled — the
# stated instruction is honoured, and recorded.
_NO_OFFER_PHRASES = [
    "just cancel", "no offers", "no offer", "don't try to keep me",
    "do not try to keep me", "don't try to sell", "do not try to sell",
    "no retention", "don't offer", "do not offer", "no deals",
    "no counter", "stop trying to keep me", "just close it",
    "just cancel it", "not interested in offers", "no sales pitch",
]


def classify_reason(text: str) -> str:
    """The governed reason for a wish to leave, from the
    customer's words alone: the first matching class in the
    governed order, or "other" when nothing in the list fits —
    the words themselves ride the case verbatim regardless."""
    lowered = f" {text.lower()} "
    for reason, phrases in _REASON_CLASSES:
        if any(p in lowered for p in phrases):
            return reason
    return "other"


def is_cancel_intent(text: str) -> bool:
    """The customer is expressing intent to cancel or leave —
    as opposed to cancelling an arrangement, a scheduled change
    or autopay, which are their own desks' transactions."""
    lowered = f" {text.lower()} "
    if any(w in lowered for w in _CANCEL_EXCLUSIONS):
        return False
    return any(re.search(p, lowered) for p in _CANCEL_PATTERNS)


def no_offers_requested(text: str) -> bool:
    """The customer has said they want no offers — "just cancel
    it", "don't try to keep me". The words are the instruction;
    the flow honours them exactly."""
    lowered = f" {text.lower()} "
    return any(p in lowered for p in _NO_OFFER_PHRASES)


# ---------------------------------------------------------------------------
# The case record
# ---------------------------------------------------------------------------


def _audit(store, conv: Conversation, detail: dict) -> None:
    store.save_audit(AuditEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id, kind="retention",
        detail=detail, turn=conv.turns))


def _touch(store, case: RetentionCase) -> RetentionCase:
    case.updated_date = today().isoformat()
    store.save_retention_case(case)
    return case


def open_case(store, conv: Conversation, text: str,
              account_number: str = "") -> RetentionCase:
    """Open the case the moment the intent is expressed: reason
    classified in code, the customer's words verbatim, and the
    no-offers instruction read off the same words."""
    case = RetentionCase(
        case_id=f"RET-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=account_number or conv.account_number or "",
        reason=classify_reason(text),
        reason_text=text,
        state="open",
        no_offers_requested=no_offers_requested(text),
        cancel_requests=1,
        created_date=today().isoformat(),
        updated_date=today().isoformat(),
    )
    store.save_retention_case(case)
    _audit(store, conv, {
        "event": "retention_case_opened", "case_id": case.case_id,
        "account_number": case.account_number, "reason": case.reason,
        "no_offers_requested": case.no_offers_requested})
    return case


def open_case_for(store, conversation_id: str) -> RetentionCase | None:
    """The conversation's case that is still live — opened and
    not yet settled into an outcome — if one is."""
    for case in store.list_retention_cases(
            conversation_id=conversation_id):
        if not case.outcome:
            return case
    return None


def note_cancel_request(store, conv: Conversation,
                        case: RetentionCase, text: str) -> RetentionCase:
    """A further explicit cancel request on a live case: the
    count rises on the record (a second request is what ends
    retention, in code), and a no-offers instruction in it is
    recorded the same way."""
    case.cancel_requests += 1
    if no_offers_requested(text):
        case.no_offers_requested = True
    _audit(store, conv, {
        "event": "retention_cancel_request", "case_id": case.case_id,
        "cancel_requests": case.cancel_requests,
        "no_offers_requested": case.no_offers_requested})
    return _touch(store, case)


def mark_escalated_fault(store, conv: Conversation,
                         case: RetentionCase) -> RetentionCase:
    """A fault-blamed exit settles as the human path: the case
    records that it was routed to the billing desk's dispute
    path — policy puts fault-driven exits with that team, not
    with a priced save offer."""
    case.state = "escalated_retention_team"
    case.outcome = "escalated_retention_team"
    case.routed = "billing_dispute_path"
    _audit(store, conv, {
        "event": "retention_escalated", "case_id": case.case_id,
        "routed": case.routed, "reason": case.reason})
    return _touch(store, case)


def mark_offer_proposed(store, conv: Conversation,
                        case: RetentionCase, action_id: str) -> RetentionCase:
    """The assembled offer became a recorded proposal: its
    action id lands on the case, so the record names the exact
    proposal the customer is answering."""
    case.save_offer_action_id = action_id
    case.save_offer_outcome = "proposed"
    case.state = "offer_proposed"
    _audit(store, conv, {
        "event": "retention_offer_proposed", "case_id": case.case_id,
        "action_id": action_id, "mechanism": case.save_offer_mechanism})
    return _touch(store, case)


def note_cancellation_proposed(store, conv: Conversation,
                               case: RetentionCase,
                               action_id: str) -> RetentionCase:
    """The governed cancellation was proposed: the case carries
    the proposal's action id until the customer's answer settles
    it (executed → mark_cancelled)."""
    case.cancellation_action_id = action_id
    _audit(store, conv, {
        "event": "retention_cancellation_proposed",
        "case_id": case.case_id, "action_id": action_id})
    return _touch(store, case)


def mark_offer_declined(store, conv: Conversation,
                        case: RetentionCase) -> RetentionCase:
    case.save_offer_outcome = "declined"
    case.state = "open"
    _audit(store, conv, {
        "event": "retention_offer_declined", "case_id": case.case_id,
        "mechanism": case.save_offer_mechanism})
    return _touch(store, case)


def mark_offer_blocked(store, conv: Conversation, case: RetentionCase,
                       block_reason: str) -> RetentionCase:
    case.save_offer_outcome = "blocked"
    case.state = "open"
    _audit(store, conv, {
        "event": "retention_offer_blocked", "case_id": case.case_id,
        "block_reason": block_reason})
    return _touch(store, case)


def mark_saved(store, conv: Conversation, case: RetentionCase,
               action_id: str) -> RetentionCase:
    """The save offer executed: the case is saved, with the
    executed action that saved it on the record."""
    case.save_offer_outcome = "accepted"
    case.save_offer_action_id = action_id or case.save_offer_action_id
    case.state = "saved"
    case.outcome = "saved_accepted_offer"
    _audit(store, conv, {
        "event": "retention_saved", "case_id": case.case_id,
        "action_id": action_id, "mechanism": case.save_offer_mechanism})
    return _touch(store, case)


def mark_cancelled(store, conv: Conversation, case: RetentionCase,
                   action_id: str, effective: str) -> RetentionCase:
    """The cancellation executed: the case closes as cancelled,
    carrying the action and the cycle-boundary date it takes
    effect — the same date the account record carries."""
    case.cancellation_action_id = action_id
    case.cancellation_effective = effective
    case.state = "cancelled"
    case.outcome = "cancelled"
    _audit(store, conv, {
        "event": "retention_cancelled", "case_id": case.case_id,
        "action_id": action_id, "effective": effective})
    return _touch(store, case)


# ---------------------------------------------------------------------------
# The save offer — one, policy-bound, real
# ---------------------------------------------------------------------------


@dataclass
class SaveOffer:
    """An assembled save offer: the typed proposal (which goes
    through the normal propose → confirm → execute gate), the
    mechanism it is built from, and the lines the reply must
    carry verbatim."""

    proposal: ProposedAction
    mechanism: str  # plan_downgrade | goodwill_credit | loyalty_upgrade
    intro: str = ""
    must_include: list[str] = field(default_factory=list)


def _proposal_version(store, conv: Conversation) -> int:
    """Plan-change proposals are versioned per conversation —
    the same counting the offers desk uses."""
    prior = [r for r in store.list_actions(conv.conversation_id)
             if r.kind == ActionKind.CHANGE_PLAN
             and r.status in ("proposed", "superseded", "declined")]
    return len(prior) + 1


def _change_proposal(store, conv: Conversation, case: RetentionCase,
                     account, target, new_price,
                     basis: str) -> tuple[ProposedAction, list[str]]:
    """A plan-change save offer, priced by the plan-change
    machinery against the account's real cycle calendar — the
    same payload shape (and fingerprint) the offers desk's own
    proposals carry, with the retention case stamped on it."""
    current = store.get_plan(account.plan_id)
    quote = plan_changes.quote_change(account, current, target, new_price)
    payload = {
        "new_plan_id": target.plan_id,
        "old_plan_id": current.plan_id,
        "new_price": str(new_price), "net": str(quote.net),
        "credit": str(quote.credit), "charge": str(quote.charge),
        "direction": quote.direction, "effective": quote.effective,
        "effective_date": quote.effective_date,
        "cycle_start": quote.cycle_start, "cycle_end": quote.cycle_end,
        "days_remaining": quote.days_remaining,
        "days_in_cycle": quote.days_in_cycle,
        "next_bill_impact": str(quote.next_bill_impact),
        "proposal_version": _proposal_version(store, conv),
        "issued_date": today().isoformat(),
        "fingerprint": plan_changes.account_fingerprint(account),
        "fingerprint_parts": plan_changes.fingerprint_parts(account),
        "retention_case_id": case.case_id,
        "retention_save_offer": True,
    }
    if quote.effective == "next_cycle":
        proposal = ProposedAction(
            kind=ActionKind.CHANGE_PLAN,
            summary=(f"change your plan to {target.name} at "
                     f"{fmt(new_price)}/month from "
                     f"{quote.effective_date}"),
            payload=payload, amount=Decimal("0"), basis=basis)
        must_include = [
            f"{target.name} takes effect on {quote.effective_date} "
            "(the start of your next cycle).",
            f"Until then you stay on {current.name}; from "
            f"{quote.effective_date}, {target.name} is "
            f"{fmt(new_price)}/month.",
            "No charge or credit applies to this change.",
        ]
        return proposal, must_include
    if quote.net > 0:
        net_line = f"Net adjustment on your next bill: {fmt(quote.net)} due."
    elif quote.net < 0:
        net_line = (f"Net adjustment on your next bill: "
                    f"a {fmt(-quote.net)} credit.")
    else:
        net_line = "Net adjustment on your next bill: $0.00."
    proposal = ProposedAction(
        kind=ActionKind.CHANGE_PLAN,
        summary=(f"change your plan to {target.name} at "
                 f"{fmt(new_price)}/month, effective now"),
        payload=payload, amount=quote.net, basis=basis)
    must_include = [
        f"Credit for the unused part of {current.name}: {fmt(quote.credit)}",
        f"Charge for {target.name} for the remaining "
        f"{quote.days_remaining} days: {fmt(quote.charge)}",
        net_line,
        f"Projected next bill: {fmt(quote.next_bill_impact)} — the net "
        f"adjustment above plus the first full month of {target.name} "
        f"at {fmt(new_price)}.",
        f"From the next cycle, {target.name} is {fmt(new_price)}/month.",
    ]
    return proposal, must_include


def _downgrade_offer(store, conv, case, account, current):
    cheaper = [p for p in store.list_plans()
               if p.monthly_price < current.monthly_price]
    if not cheaper:
        return None
    target = min(cheaper, key=lambda p: p.monthly_price)
    proposal, must_include = _change_proposal(
        store, conv, case, account, target, target.monthly_price,
        basis=("Retention save offer under POL-RET-01: a downgrade "
               "priced by the plan-change machinery (POL-PLAN-02 — "
               "next-cycle start, no proration)."))
    return SaveOffer(
        proposal=proposal, mechanism="plan_downgrade",
        intro=(f"Before I do that, there's one thing I can offer: "
               f"move you down to {target.name} at "
               f"{fmt(target.monthly_price)}/month instead of "
               f"{fmt(current.monthly_price)}. That's the one save "
               f"offer I can make — if it doesn't work for you, "
               f"I'll cancel exactly as you asked, and I won't "
               f"offer anything else."),
        must_include=must_include)


def _loyalty_offer(store, conv, case, account, current, customer):
    loyalty = store.get_offer("OFF-UPG-LOYAL")
    if loyalty is None or customer is None \
            or customer.tenure_months < 12 \
            or account.plan_id not in loyalty.eligible_plan_ids:
        return None
    target = store.get_plan(loyalty.target_plan_id or "")
    if target is None or target.plan_id == current.plan_id:
        return None
    basis = (f"Retention save offer under POL-RET-01: loyalty "
             f"upgrade OFF-UPG-LOYAL: {fmt(loyalty.monthly_price)}/month "
             f"instead of {fmt(target.monthly_price)} (tenure "
             f"{customer.tenure_months} months).")
    proposal, must_include = _change_proposal(
        store, conv, case, account, target, loyalty.monthly_price, basis)
    return SaveOffer(
        proposal=proposal, mechanism="loyalty_upgrade",
        intro=(f"Before I do that, there's one thing I can offer: "
               f"as a customer of {customer.tenure_months} months you "
               f"qualify for {target.name} at "
               f"{fmt(loyalty.monthly_price)}/month instead of "
               f"{fmt(target.monthly_price)} — the loyalty price. "
               f"That's the one save offer I can make — if it "
               f"doesn't work for you, I'll cancel exactly as you "
               f"asked, and I won't offer anything else."),
        must_include=must_include)


def _goodwill_offer(store, conv, case, account, customer, policy):
    if customer is None or customer.tenure_months < 12:
        return None
    offer, history = billing_book.goodwill_offer(store, conv, policy)
    if offer <= 0 or history.offered_here:
        return None
    basis = "Retention save offer under POL-RET-01: goodwill under POL-BILL-02."
    if offer < policy.goodwill_credit_cap:
        basis += (" Offer reduced to the headroom left under the "
                  "rolling goodwill cap.")
    proposal = ProposedAction(
        kind=ActionKind.APPLY_CREDIT,
        summary=f"apply a goodwill credit of {fmt(offer)}",
        payload={"credit_class": "goodwill",
                 "retention_case_id": case.case_id,
                 "retention_save_offer": True},
        amount=offer, basis=basis)
    return SaveOffer(
        proposal=proposal, mechanism="goodwill_credit",
        intro=(f"Before I do that, there's one thing I can offer: "
               f"a goodwill credit of {fmt(offer)} on your account, "
               f"for how this has gone. That's the one save offer I "
               f"can make — if it doesn't work for you, I'll cancel "
               f"exactly as you asked, and I won't offer anything "
               f"else."),
        must_include=[])


def assemble_save_offer(store, conv: Conversation, case: RetentionCase,
                        policy) -> SaveOffer | None:
    """Assemble the case's ONE save offer from the mechanisms
    already in the book — or None, honestly, when there is none
    to make. The single-counter rule is enforced here, at
    proposal time (and again in actions.py at execution): a
    case that already carries an offer — proposed, answered, in
    any state — never gets a second one assembled. A customer
    who asked for no offers gets none. Which mechanism answers
    is read off the recorded reason:

    - price → a downgrade to the cheapest cheaper plan, else a
      goodwill credit;
    - service_quality / coverage → a goodwill credit, else the
      loyalty upgrade price;
    - competitor / other → the loyalty upgrade price, else a
      downgrade;
    - no_longer_needed → a downgrade, else the loyalty price;
    - moving_house → none: no mechanism in the book answers a
      move, and the desk does not invent one.
    """
    if case.no_offers_requested or case.outcome:
        return None
    if case.save_offer_summary or case.save_offer_action_id \
            or case.save_offer_outcome:
        return None  # the single counter has already been spent
    account = store.get_account(
        case.account_number or conv.account_number or "")
    if account is None:
        return None
    current = store.get_plan(account.plan_id)
    if current is None:
        return None
    customer = store.get_customer(conv.customer_id) \
        if conv.customer_id else None
    if customer is None:
        customer = store.get_customer_by_account(account.account_number)
    reason = case.reason
    if reason == "moving_house":
        return None
    builders = {
        "price": (_downgrade_offer, _goodwill_offer),
        "service_quality": (_goodwill_offer, _loyalty_offer),
        "coverage": (_goodwill_offer, _loyalty_offer),
        "competitor": (_loyalty_offer, _downgrade_offer),
        "other": (_loyalty_offer, _downgrade_offer),
        "no_longer_needed": (_downgrade_offer, _loyalty_offer),
    }
    for builder in builders.get(reason, ()):
        if builder is _goodwill_offer:
            offer = builder(store, conv, case, account, customer, policy)
        elif builder is _loyalty_offer:
            offer = builder(store, conv, case, account, current, customer)
        else:
            offer = builder(store, conv, case, account, current)
        if offer is not None:
            case.save_offer_kind = offer.proposal.kind.value
            case.save_offer_mechanism = offer.mechanism
            case.save_offer_summary = offer.proposal.summary
            case.save_offer_outcome = "proposed"
            _touch(store, case)
            return offer
    return None


# ---------------------------------------------------------------------------
# The cancellation itself
# ---------------------------------------------------------------------------


def cancellation_effective_date(account) -> str:
    """When a cancellation confirmed today takes effect: the day
    after the current cycle ends — the same boundary the
    pending-plan machinery schedules a downgrade at."""
    try:
        end = date.fromisoformat(account.cycle_end)
    except (TypeError, ValueError):
        return today().isoformat()
    return (end + timedelta(days=1)).isoformat()


def cancellation_proposal(conv: Conversation, case: RetentionCase,
                          account) -> ProposedAction:
    """The governed cancellation, as a typed proposal for the
    normal confirmation gate: effective at the cycle's end, so
    the customer keeps the service they have paid for until
    then. Executing it (actions.py, kind cancel_account) writes
    that date onto the account as its pending cancellation."""
    effective = cancellation_effective_date(account)
    return ProposedAction(
        kind=ActionKind.CANCEL_ACCOUNT,
        summary=(f"cancel your Northbeam account on {effective}, "
                 f"at the end of this billing cycle"),
        payload={"retention_case_id": case.case_id,
                 "effective": "next_cycle",
                 "effective_date": effective,
                 "cycle_end": account.cycle_end},
        amount=Decimal("0"),
        basis=("Cancellation under POL-RET-01: effective at the "
               "end of the current cycle; the customer keeps the "
               "service paid for until then."))


# ---------------------------------------------------------------------------
# The report — counts from records only
# ---------------------------------------------------------------------------

OUTCOMES = ("saved_accepted_offer", "cancelled",
            "escalated_retention_team")


def retention_report(store) -> dict:
    """The retention report over every stored case: cases by
    reason, offers made and accepted, the save rate over decided
    cases (an outcome in OUTCOMES), and cancellations completed.
    Every figure is a count over the case records — nothing is
    read from conversations or prose."""
    cases = store.list_retention_cases()
    by_reason = {reason: 0 for reason in RETENTION_REASONS}
    by_outcome = {outcome: 0 for outcome in OUTCOMES}
    offers_made = offers_accepted = offers_declined = 0
    no_offers = 0
    for case in cases:
        if case.reason in by_reason:
            by_reason[case.reason] += 1
        if case.outcome in by_outcome:
            by_outcome[case.outcome] += 1
        if case.no_offers_requested:
            no_offers += 1
        if case.save_offer_summary or case.save_offer_action_id:
            offers_made += 1
        if case.save_offer_outcome == "accepted":
            offers_accepted += 1
        elif case.save_offer_outcome == "declined":
            offers_declined += 1
    decided = sum(by_outcome.values())
    saved = by_outcome["saved_accepted_offer"]
    return {
        "cases": len(cases),
        "by_reason": by_reason,
        "by_outcome": by_outcome,
        "offers_made": offers_made,
        "offers_accepted": offers_accepted,
        "offers_declined": offers_declined,
        "no_offers_requested": no_offers,
        "decided_cases": decided,
        "save_rate": (saved / decided) if decided else None,
        "cancellations_completed": by_outcome["cancelled"],
        "escalated_retention_team": by_outcome["escalated_retention_team"],
    }
