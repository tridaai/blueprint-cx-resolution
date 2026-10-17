"""The billing book: every figure and decision the billing desk
stands on, computed here in code.

- **Decomposition** — a bill grouped into charge families with
  per-family totals, and deltas against the previous bill. The
  narrative the customer hears may only use these figures.
- **Bill shock** — when the total jumps past the policy threshold
  over the trailing average, the driver is chosen here from the
  deltas (largest positive family delta, a new family, a one-off),
  never by the model.
- **Dispute assessment** — each disputed line decided on its own
  evidence: evidenced lines credit (in full, or the excess on a
  wrong rate), named-but-clean lines decline with a plain-words
  reason, and an evidenced total above the tier cap escalates with
  the case record intact.
- **Credit history** — what an account has actually been credited,
  read across conversations over the trailing window. Goodwill is
  offered only out of the headroom the history leaves; the
  evidenced tier never reads it.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .clock import today
from .config import Policy
from .money import fmt
from .models import (
    ActionKind,
    AuditEvent,
    Bill,
    BillCharge,
    Conversation,
    DisputeCase,
    DisputeLine,
)

ZERO = Decimal("0")

# Charge families, in the order an explanation walks them. Charge
# kinds on the bill map onto exactly one family each.
FAMILIES: list[tuple[str, str, tuple[str, ...]]] = [
    ("plan", "Plan", ("plan",)),
    ("device", "Device", ("device",)),
    ("usage", "Usage", ("usage",)),
    ("roaming", "Roaming", ("roaming",)),
    ("one_off", "One-offs", ("one_off",)),
    ("taxes_fees", "Taxes & fees", ("tax", "fee")),
]
_FAMILY_OF_KIND = {kind: key for key, _label, kinds in FAMILIES for kind in kinds}
_FAMILY_LABEL = {key: label for key, label, _kinds in FAMILIES}


def family_of(charge: BillCharge) -> str:
    return _FAMILY_OF_KIND.get(charge.kind, "one_off")


def decompose_bill(bill: Bill) -> list[dict]:
    """The bill grouped by family: per-family totals computed from
    the lines, families in walk order, empty families omitted."""
    groups: list[dict] = []
    for key, label, _kinds in FAMILIES:
        lines = [c for c in bill.charges if family_of(c) == key]
        if not lines:
            continue
        groups.append({
            "family": key, "label": label, "lines": lines,
            "total": sum((c.amount for c in lines), ZERO),
        })
    return groups


def family_totals(bill: Bill) -> dict[str, Decimal]:
    return {g["family"]: g["total"] for g in decompose_bill(bill)}


def compare_bills(current: Bill, previous: Bill) -> list[dict]:
    """Per-family deltas of the current bill against the previous
    one — every family present on either bill, in walk order."""
    cur, prev = family_totals(current), family_totals(previous)
    rows = []
    for key, label, _kinds in FAMILIES:
        if key not in cur and key not in prev:
            continue
        c, p = cur.get(key, ZERO), prev.get(key, ZERO)
        rows.append({"family": key, "label": label, "current": c,
                     "previous": p, "delta": c - p})
    return rows


def trailing_average(bills: list[Bill]) -> Decimal:
    """Mean total of the account's earlier bills — the baseline a
    bill-shock jump is measured against."""
    if not bills:
        return ZERO
    return sum((b.total for b in bills), ZERO) / Decimal(len(bills))


def bill_shock(current: Bill, previous_bills: list[Bill],
               policy: Policy) -> dict | None:
    """The bill-shock read, or None when the bill is unremarkable.

    Fires when the current total exceeds the trailing average by
    at least the policy threshold. The driver is the family with
    the largest positive delta against the previous bill; a driver
    family the previous bill didn't have is a new charge, and a
    driver in the one-off family is named by its largest line."""
    if not previous_bills:
        return None
    average = trailing_average(previous_bills)
    if average <= 0 or current.total <= average:
        return None
    pct = ((current.total - average) / average * 100).quantize(Decimal("0.1"))
    if pct < policy.bill_shock_threshold_pct:
        return None
    previous = previous_bills[-1]
    rows = [r for r in compare_bills(current, previous) if r["delta"] > 0]
    if not rows:
        return None
    # compare_bills walks families in explanation order, and max()
    # keeps the first of equals — ties go to the earlier family.
    driver = max(rows, key=lambda r: r["delta"])
    top_line = None
    if driver["family"] == "one_off":
        lines = [c for c in current.charges if family_of(c) == "one_off"]
        top_line = max(lines, key=lambda c: c.amount) if lines else None
    return {
        "average": average,
        "increase": current.total - average,
        "pct": pct,
        "driver": {
            "family": driver["family"], "label": driver["label"],
            "current": driver["current"], "previous": driver["previous"],
            "delta": driver["delta"],
            "is_new": driver["previous"] == 0,
            "top_line": top_line,
        },
    }


# ---------------------------------------------------------------------------
# Dispute assessment
# ---------------------------------------------------------------------------


def line_evidence(charge: BillCharge) -> tuple[Decimal, str]:
    """What the bill itself proves about one charge: the amount
    that shouldn't be there and the evidence in words, or
    (0, "") when the charge is clean."""
    if charge.duplicate_of:
        return charge.amount, (
            f"duplicate charge '{charge.description}' of {fmt(charge.amount)}")
    if charge.after_cancellation:
        return charge.amount, (
            f"'{charge.description}' of {fmt(charge.amount)} "
            "charged after cancellation")
    if charge.expected_amount is not None and charge.amount > charge.expected_amount:
        return charge.amount - charge.expected_amount, (
            f"'{charge.description}' billed at {fmt(charge.amount)} "
            f"against a {fmt(charge.expected_amount)} rate")
    return ZERO, ""


_STOPWORDS = {"monthly", "the", "and", "for", "with", "your", "this", "that"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 4 and w not in _STOPWORDS}


def _named_charges(bill: Bill, text: str) -> list[BillCharge]:
    """Charges the customer points at by name. A charge is named
    when two of its distinctive words appear in the message, or
    when a word unique to it on this bill does — one shared word
    ('plan', 'data') is a topic, not a pointer."""
    said = _tokens(text)
    token_sets = {c.charge_id: _tokens(c.description) for c in bill.charges}
    named = []
    for charge in bill.charges:
        hits = token_sets[charge.charge_id] & said
        unique_hit = any(
            sum(1 for other in token_sets.values() if tok in other) == 1
            for tok in hits)
        if len(hits) >= 2 or unique_hit:
            named.append(charge)
    return named


_INSTALMENT_RE = re.compile(r"instalment\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)


def decline_reason(charge: BillCharge) -> str:
    """Why a clean charge stands, in plain words a customer can
    check against their own paperwork."""
    match = _INSTALMENT_RE.search(charge.description)
    if charge.kind == "device" and match:
        return (f"instalment {match.group(1)} of {match.group(2)} on the "
                "handset agreement for your current phone — it matches "
                "that schedule")
    if charge.kind == "plan":
        return "the monthly price of the plan on your account — it matches"
    if charge.kind == "device":
        return "the instalment schedule on your account — it matches"
    if charge.kind == "tax":
        return "taxes and regulatory fees worked out on the rest of the bill"
    if charge.kind == "fee":
        return "the set fee for this on your account — it matches"
    if charge.kind == "roaming":
        return "roaming usage recorded on your account for this trip"
    if charge.kind == "one_off":
        return "a one-off charge recorded as confirmed on your account"
    return "the rates and dates on your account — it matches them"


@dataclass
class RoamingVerdict:
    """One roaming line decided from the account's records rather
    than the bill's own markers (roaming_book computes these): a
    creditable amount with its evidence, or zero with the evidence
    and the plain-words reason the charge stands."""

    charge_id: str
    creditable: Decimal = ZERO
    evidence: str = ""
    decline_reason: str = ""


@dataclass
class Assessment:
    lines: list[DisputeLine] = field(default_factory=list)
    credit_total: Decimal = ZERO  # sum of credited lines
    evidenced_total: Decimal = ZERO  # all evidenced money, credited or not
    escalated: bool = False  # evidenced total above the tier cap

    @property
    def credited(self) -> list[DisputeLine]:
        return [l for l in self.lines if l.decision == "credited"]

    @property
    def declined(self) -> list[DisputeLine]:
        return [l for l in self.lines if l.decision == "declined"]


def assess_dispute(bill: Bill, text: str, policy: Policy,
                   roaming_verdicts: dict | None = None) -> Assessment:
    """Decide every disputed line on its own evidence.

    The disputed set is every line the bill itself proves wrong,
    plus every clean line the customer names. The original of a
    duplicate pair is not a separate dispute — the pair is one
    wrong charge — so it is never declined alongside its duplicate.

    Roaming lines may also arrive decided from the account's
    records (`roaming_verdicts`, computed by roaming_book): a
    creditable verdict joins the evidenced set on equal footing
    with the bill's own markers; a stands verdict declines the
    line with its recorded reason when the customer disputes the
    roaming charge, whether they named the line or the topic.
    """
    verdicts = roaming_verdicts or {}
    evidenced: dict[str, tuple[Decimal, str]] = {}
    pair_originals: set[str] = set()
    for charge in bill.charges:
        amount, evidence = line_evidence(charge)
        if amount > 0:
            evidenced[charge.charge_id] = (amount, evidence)
            if charge.duplicate_of:
                pair_originals.add(charge.duplicate_of)
    for charge_id, verdict in verdicts.items():
        if verdict.creditable > 0 and charge_id not in evidenced:
            evidenced[charge_id] = (verdict.creditable, verdict.evidence)
    evidenced_total = sum((a for a, _e in evidenced.values()), ZERO)
    escalated = evidenced_total > policy.dispute_credit_cap

    named_ids = {c.charge_id for c in _named_charges(bill, text)}
    roaming_named = "roaming" in _tokens(text) or "roam" in _tokens(text)
    lines: list[DisputeLine] = []
    for charge in bill.charges:
        if charge.charge_id in evidenced:
            amount, evidence = evidenced[charge.charge_id]
            lines.append(DisputeLine(
                charge_id=charge.charge_id, description=charge.description,
                amount=charge.amount, disputed_amount=amount,
                evidence=evidence,
                decision="escalated" if escalated else "credited"))
        elif charge.charge_id in verdicts \
                and (charge.charge_id in named_ids or roaming_named):
            verdict = verdicts[charge.charge_id]
            lines.append(DisputeLine(
                charge_id=charge.charge_id, description=charge.description,
                amount=charge.amount, disputed_amount=charge.amount,
                evidence=verdict.evidence,
                decision="declined", reason=verdict.decline_reason))
        elif charge.charge_id in named_ids and charge.charge_id not in pair_originals:
            lines.append(DisputeLine(
                charge_id=charge.charge_id, description=charge.description,
                amount=charge.amount, disputed_amount=charge.amount,
                decision="declined", reason=decline_reason(charge)))
    credit_total = sum((l.disputed_amount for l in lines
                        if l.decision == "credited"), ZERO)
    return Assessment(lines=lines, credit_total=credit_total,
                      evidenced_total=evidenced_total, escalated=escalated)


# ---------------------------------------------------------------------------
# Dispute cases — the record
# ---------------------------------------------------------------------------


def _audit(store, conv: Conversation, detail: dict) -> None:
    store.save_audit(AuditEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id, kind="billing",
        detail=detail, turn=conv.turns))


def open_dispute_case(store, conv: Conversation, bill: Bill,
                      assessment: Assessment, basis: str) -> DisputeCase:
    """Persist the assessment as a case and write the audit link.
    The state at open is final for declines and escalations; a case
    with credited lines stays `opened` until the customer answers
    the proposed credit."""
    if assessment.escalated:
        state = "escalated_above_cap"
    elif assessment.credited:
        state = "opened"
    else:
        state = "declined_with_reason"
    case = DisputeCase(
        case_id=f"DSP-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=conv.account_number or "",
        bill_id=bill.bill_id,
        state=state,
        lines=assessment.lines,
        decision_basis=basis,
        created_date=today().isoformat(),
        updated_date=today().isoformat(),
    )
    store.save_dispute_case(case)
    _audit(store, conv, {
        "event": "dispute_opened", "case_id": case.case_id,
        "bill_id": bill.bill_id, "state": state,
        "lines": [{"charge_id": l.charge_id, "decision": l.decision,
                   "disputed_amount": str(l.disputed_amount)}
                  for l in assessment.lines],
        "credit_total": str(assessment.credit_total),
        "evidenced_total": str(assessment.evidenced_total)})
    return case


def settle_dispute_case(store, conv: Conversation, case_id: str,
                        outcome: str, amount: Decimal | None,
                        note: str = "") -> DisputeCase | None:
    """Move a case when its proposed credit is answered: executed
    settles it (fully, or partially when lines were declined);
    declined or blocked leaves it opened with the answer on its
    basis, so the record never claims a decision that didn't land."""
    case = store.get_dispute_case(case_id)
    if case is None:
        return None
    if outcome == "executed":
        case.state = "decided_partial" if any(
            l.decision == "declined" for l in case.lines) else "decided_credit"
        case.credit_applied = amount or ZERO
    if note:
        case.decision_basis = f"{case.decision_basis} {note}".strip()
    case.updated_date = today().isoformat()
    store.save_dispute_case(case)
    _audit(store, conv, {
        "event": "dispute_decided", "case_id": case.case_id,
        "outcome": outcome, "state": case.state,
        "credit_applied": str(case.credit_applied)})
    return case


def note_goodwill_refusal(store, conv: Conversation,
                          case: DisputeCase | None, reason: str,
                          detail: dict) -> None:
    """A goodwill refusal is recorded, never just said: on the
    dispute case when one is open in this conversation, and always
    on the conversation's audit trail."""
    if case is not None:
        case.goodwill_refused_reason = reason
        case.decision_basis = (
            f"{case.decision_basis} Goodwill was considered and refused: "
            f"{reason}").strip()
        case.updated_date = today().isoformat()
        store.save_dispute_case(case)
    _audit(store, conv, {"event": "goodwill_refused",
                         "case_id": case.case_id if case else None,
                         "reason": reason, **detail})


# ---------------------------------------------------------------------------
# Credit history — the policy's memory
# ---------------------------------------------------------------------------


@dataclass
class CreditHistory:
    executed_total: Decimal = ZERO  # goodwill actually applied, in window
    pending_total: Decimal = ZERO  # goodwill proposed in this conversation
    entries: list = field(default_factory=list)
    offered_here: bool = False  # goodwill already offered in this conversation

    @property
    def total(self) -> Decimal:
        return self.executed_total + self.pending_total


def _in_window(created: str, window_days: int, on: date) -> bool:
    try:
        made = date.fromisoformat(created)
    except (TypeError, ValueError):
        return False
    return 0 <= (on - made).days <= window_days


def credit_history(store, account_number: str, policy: Policy,
                   conversation_id: str | None = None) -> CreditHistory:
    """Goodwill credited to an account inside the trailing window,
    read across every conversation the account has had. Pending
    goodwill in the current conversation counts too — an unanswered
    offer must not let a second one spend the same headroom — but a
    proposal that already executed is counted once, as executed."""
    on = today()
    history = CreditHistory()
    if not account_number:
        return history
    goodwill = [
        r for r in store.list_actions()
        if r.account_number == account_number
        and r.kind == ActionKind.APPLY_CREDIT
        and r.payload.get("credit_class") == "goodwill"
    ]
    executed_here = False
    for record in goodwill:
        if conversation_id and record.conversation_id == conversation_id \
                and record.status in ("proposed", "executed"):
            history.offered_here = True
        if record.status == "executed" and _in_window(
                record.created_date, policy.credit_history_days, on):
            history.executed_total += record.amount or ZERO
            history.entries.append(record)
            if conversation_id and record.conversation_id == conversation_id:
                executed_here = True
    if conversation_id and not executed_here:
        for record in goodwill:
            if (record.status == "proposed"
                    and record.conversation_id == conversation_id):
                history.pending_total += record.amount or ZERO
                history.entries.append(record)
    return history


def goodwill_offer(store, conv: Conversation,
                   policy: Policy) -> tuple[Decimal, CreditHistory]:
    """The goodwill the policy can still offer this account: the
    smaller of the per-offer cap and the rolling headroom the
    history leaves. Zero (or less) means goodwill is refused."""
    history = credit_history(store, conv.account_number or "", policy,
                             conversation_id=conv.conversation_id)
    headroom = policy.goodwill_rolling_cap - history.total
    return min(policy.goodwill_credit_cap, headroom), history
