"""Handoff quality (round 17) — the packet is graded, the
pickup is recorded, and the customer is told only what the
packet actually holds.

COMPLETENESS, IN CODE. Every packet is graded the moment it is
built, against the elements its class of escalation requires:

- identity — the verification state and the attempts spent
  reaching it. Carried by every packet by construction.
- issue_in_customers_words — the customer's own first message.
  Carried whenever the customer said anything at all.
- vulnerability — the flag determination. The field IS the
  determination, so it is carried by construction; it is named
  here because a packet that could not say would be a defect.
- signals — the customer's experience of the conversation
  (round 11). Required of every live transfer; the complaint
  watch's queue entries are graded without it — there is no
  live conversation experience to summarise there.
- tried — procedure steps with their outcomes, dispute lines
  as decided, actions attempted. Required of a standard
  handoff: a human continuing work needs to know what the work
  already was. Immediate escalations (threat, fraud, legal),
  identity lockouts and vulnerability transfers are not graded
  on it — they leave before trying is possible or proper.
- figures — the money the case turns on. Required whenever
  the case is a billing case, by intent or by queue, whichever
  door it leaves by.
- open_promises — the account's open promises and complaints,
  read from the round 16 ledger. Required whenever the account
  is known and the handoff is not an immediate routing: the
  human inherits the commitments too.

A packet missing a required element is INCOMPLETE, with the
gaps named on the packet, in the escalation's audit event, and
in the daily digest. The transfer itself is never blocked on
paperwork — a customer who needs a person gets a person; the
grade is for the humans and the record.

ACCEPTANCE IS AN ACT. A handoff sits in its queue until a
person picks it up: acceptance stamps the receiver and the
day on the escalation record and writes the audit event.
Handoffs unaccepted past the governed threshold
(`handoff_acceptance_days`) are named by the digest and the
queue views — visible, never silently reassigned; nothing
here moves a handoff on its own.

THE LINE TO THE CUSTOMER. The transfer reply's account of
what was passed on is generated from the packet's contents,
element by element: the reply claims a step was tried, a
figure named, a fault discussed only when the packet holds
it. An incomplete packet produces a shorter line, never a
false one.
"""

from __future__ import annotations

import uuid
from datetime import date

from .clock import today
from .config import Policy
from .models import AuditEvent, Escalation, HandoffPacket

# The escalation reasons that leave before any solving is
# possible or proper: a threat, a fraud report, a legal
# matter. Their packets are graded on what a first-turn
# handoff can honestly carry.
IMMEDIATE_REASONS = {"safety_threat", "fraud_report", "legal_matter"}


def packet_class(reason: str) -> str:
    """The class an escalation's packet is graded as, from the
    reason it was raised: immediate | identity | care | watch |
    standard (everything else — the customer asked twice, the
    conversation stalled, a desk reached its limit)."""
    if reason in IMMEDIATE_REASONS:
        return "immediate"
    if reason == "identity_verification_failed":
        return "identity"
    if reason == "vulnerable_customer":
        return "care"
    if reason == "complaint_sla_breach":
        return "watch"
    return "standard"


def grade_packet(packet: HandoffPacket, intent: str = "") -> HandoffPacket:
    """Grade a built packet in place: name its class, and the
    required elements it is missing. Deterministic — the same
    packet always earns the same grade."""
    cls = packet_class(packet.reason)
    packet.handoff_class = cls
    missing: list[str] = []
    if not packet.identity_state:
        missing.append("identity")
    if not packet.issue:
        missing.append("issue_in_customers_words")
    if cls != "watch" and not packet.signals:
        missing.append("signals")
    if cls == "standard" and not packet.tried:
        missing.append("tried")
    if (intent == "billing" or packet.queue == "billing") \
            and not packet.figures:
        missing.append("figures")
    if packet.account_number and cls != "immediate" \
            and not packet.open_state_known:
        missing.append("open_promises")
    packet.missing = missing
    packet.complete = not missing
    return packet


def continuity_line(packet: HandoffPacket) -> str:
    """What the customer is told the receiving agent will have —
    generated from the packet, element by element, so the line
    claims only what the packet holds. The vulnerability flag
    is deliberately not announced (the round 16 discipline: it
    rides the packet for the human, not the reply)."""
    parts = ["your identity and verification status"]
    if packet.issue:
        parts.append("what you told me about the issue "
                     "in your own words")
    if packet.tried:
        parts.append("the steps we tried and what happened")
    if packet.figures:
        parts.append("the amounts involved")
    if packet.known_incidents:
        parts.append("the known fault we discussed")
    if packet.open_items:
        parts.append("what is still open")
    joined = parts[0] if len(parts) == 1 else \
        ", ".join(parts[:-1]) + " and " + parts[-1]
    return (f"I've passed on {joined} — so you won't need to "
            "repeat yourself.")


def accept_handoff(store, escalation_id: str, receiver: str,
                   on: date | None = None) -> Escalation:
    """Record a person picking a handoff up: receiver and day
    stamped on the escalation, the act in the conversation's
    audit trail. Acceptance is singular — an accepted handoff
    keeps its first receiver; a second acceptance is refused,
    so the record of who took the case never rewrites itself."""
    receiver = (receiver or "").strip()
    if not receiver:
        raise ValueError("a receiver name is required")
    esc = next((e for e in store.list_escalations()
                if e.escalation_id == escalation_id), None)
    if esc is None:
        raise KeyError(escalation_id)
    if esc.accepted_by:
        raise ValueError(
            f"{escalation_id} was already accepted by {esc.accepted_by}")
    esc.accepted_by = receiver
    esc.accepted_date = (on or today()).isoformat()
    store.save_escalation(esc)
    store.save_audit(AuditEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:10]}",
        conversation_id=esc.conversation_id, kind="handoff",
        detail={"event": "handoff_accepted",
                "escalation_id": esc.escalation_id,
                "receiver": receiver, "queue": esc.queue},
        turn=esc.turn))
    return esc


def unaccepted_overdue(store, policy: Policy,
                       on: date | None = None) -> list[dict]:
    """The handoffs nobody has picked up past the governed
    threshold — oldest first, each with its age and its
    packet's grade. Escalations raised before creation dates
    were recorded cannot be aged and are never reported here.
    A read, like the watches: it names, a person acts."""
    on = on or today()
    rows: list[dict] = []
    for esc in store.list_escalations():
        if esc.accepted_by or not esc.created_date:
            continue
        age = (on - date.fromisoformat(esc.created_date)).days
        if age > policy.handoff_acceptance_days:
            rows.append({
                "escalation_id": esc.escalation_id,
                "conversation_id": esc.conversation_id,
                "queue": esc.queue,
                "reason": esc.reason,
                "created_date": esc.created_date,
                "age_days": age,
                "complete": esc.packet.complete,
                "missing": list(esc.packet.missing),
            })
    rows.sort(key=lambda r: (-r["age_days"], r["escalation_id"]))
    return rows
