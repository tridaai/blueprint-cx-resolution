"""Escalation discipline — the rules live here, in code, and are
evaluated on every turn before anything else composes a reply.

The rules, in the order they are checked:

1. A threat of harm escalates immediately (queue: general).
2. A fraud report escalates immediately (queue: fraud) — a human
   owns fraud, always.
3. Legal signals escalate immediately (queue: general).
4. Vulnerability — a flagged account, or the customer disclosing
   hardship/vulnerability — escalates to the care queue. A plainly
   stated circumstance (the governed classes in screening) also
   SETS the persistent flag on the account record, in the
   supervisor's screen node, before this rule routes the
   conversation; the broad keyword screen alone never sets it.
5. A locked-out identity verification escalates (queue: general);
   a human can complete verification on their side.
6. The customer asking for a human: the FIRST ask earns one genuine
   solve attempt with a clear offer; the SECOND ask escalates
   (queue: general). No dark patterns, no third ask.
7. No progress for `max_no_progress_turns` consecutive turns
   escalates to the queue for the active intent.

Every escalation carries a HandoffPacket built here from the
record — what was tried, what was done, what is still open — so the
customer never repeats themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import continuity
from . import handoffs
from . import incidents as incidents_mod
from . import signals as signals_mod
from .config import Policy
from .money import fmt
from .models import Conversation, HandoffPacket
from .screening import Screening

_QUEUE_BY_INTENT = {
    "billing": "billing",
    "roaming": "general",
    "offers": "general",
    "technical": "general",
    "account": "general",
    "care": "general",
    "unknown": "general",
}


@dataclass
class EscalationDecision:
    escalate: bool = False
    reason: str = ""
    queue: str = "general"


def evaluate(conv: Conversation, screening: Screening, policy: Policy,
             account_vulnerable: bool = False) -> EscalationDecision:
    if screening.threat:
        return EscalationDecision(True, "safety_threat", "general")
    if screening.fraud:
        return EscalationDecision(True, "fraud_report", "fraud")
    if screening.legal:
        return EscalationDecision(True, "legal_matter", "general")
    if screening.vulnerable or screening.vulnerable_statement \
            or account_vulnerable:
        return EscalationDecision(True, "vulnerable_customer", "vulnerable_care")
    if conv.identity_state.value == "failed":
        return EscalationDecision(True, "identity_verification_failed", "general")
    if screening.asks_for_human:
        conv.human_requests += 1
        if conv.human_requests >= policy.human_requests_before_escalation:
            return EscalationDecision(True, "customer_requested_human", "general")
    if conv.no_progress_turns >= policy.max_no_progress_turns:
        return EscalationDecision(
            True, "no_progress", _QUEUE_BY_INTENT.get(conv.intent.value, "general"))
    return EscalationDecision()


def build_packet(conv: Conversation, store, reason: str, queue: str) -> HandoffPacket:
    customer = store.get_customer(conv.customer_id) if conv.customer_id else None
    actions = store.list_actions(conv.conversation_id)
    taken = [f"{a.kind.value}: {a.summary}" for a in actions if a.status == "executed"]
    attempts = [f"{a.kind.value} proposed ({a.status})" for a in actions
                if a.status in ("proposed", "declined", "blocked")]
    last_customer = next(
        (m.text for m in reversed(conv.messages) if m.role == "customer"), "")
    # The issue in the customer's own words (round 17): their
    # first message — the problem as they stated it, before the
    # conversation reframed it.
    first_customer = next(
        (m.text for m in conv.messages if m.role == "customer"), "")
    open_items: list[str] = []
    if conv.pending_action is not None:
        open_items.append(f"Unconfirmed action: {conv.pending_action.summary}")
    if conv.unhandled_intent:
        open_items.append(f"Unhandled need logged: {conv.unhandled_intent}")
    vulnerable = bool(customer and customer.vulnerable)
    if vulnerable:
        set_here = any(
            e.kind == "vulnerability"
            and e.detail.get("event") == "flag_set"
            for e in store.list_audit(conv.conversation_id))
        open_items.append(
            "Vulnerability flag set on the account record in this "
            "conversation, from the customer's own statement."
            if set_here else
            "Vulnerability flag on the account record.")
    if conv.status != "resolved" and not taken:
        open_items.append("The customer's issue is not resolved.")
    summary = (
        f"{conv.turns} turn(s); intent={conv.intent.value}; "
        f"last customer message: {last_customer[:160]}"
    )
    # The customer's experience rides the packet too (round 11):
    # the effort the conversation cost them, how their frustration
    # moved, and which signal interventions were already tried and
    # how they were answered — computed live from the record, so
    # it is true at the moment of handoff. When the experience
    # itself needs the human's attention, an open item says so.
    sig_summary = signals_mod.packet_summary(store, conv)
    answered = [i for i in sig_summary["interventions"]
                if i["outcome"] in ("accepted", "declined", "lapsed")]
    if sig_summary["frustration_trend"] == "rising" or answered:
        open_items.append(
            f"Customer experience: effort {sig_summary['effort_score']} "
            f"over {conv.turns} turn(s), frustration "
            f"{sig_summary['frustration_trend']}; "
            f"{len(sig_summary['interventions'])} signal intervention(s) "
            "already tried — see the packet's signal summary.")
    # A matched known issue rides the packet (round 15): the
    # incidents this conversation matched that are still live on
    # the feed, as the records state them — the human sees the
    # customer was already told about a known fault, and exactly
    # what they were told.
    known_incidents = incidents_mod.packet_incidents(store, conv)
    for known in known_incidents:
        open_items.append(
            f"Known issue {known['incident_id']}: {known['title']} "
            f"({known['status']}) — acknowledged to the customer "
            f"from the incident record.")
    # What was tried, at continuation level (round 17): the
    # procedure walks' steps with their outcomes, the dispute
    # lines as they were decided, and the actions attempted —
    # grouped by the record they came from, each line traceable
    # to it. A walk that was only opened contributes nothing:
    # nothing was tried yet, and the grade will say so.
    tried: list[str] = []
    for run in store.list_procedure_runs(conv.conversation_id):
        for step in run.history:
            tried.append(
                f"{run.procedure_id} step {step.step_id}: "
                f"{step.result or step.detail}")
        if run.outcome and run.state != "active":
            tried.append(f"{run.procedure_id}: {run.outcome}")
    cases = store.list_dispute_cases(conversation_id=conv.conversation_id)
    for case in cases:
        for line in case.lines:
            if line.decision:
                tried.append(
                    f"Dispute {case.case_id}: {line.description} "
                    f"— {line.decision}")
    tried += attempts
    # The figures the case turns on (round 17): action amounts
    # and disputed charges, each named with its record.
    figures: list[str] = []
    for a in actions:
        if a.amount is not None:
            figures.append(f"{a.kind.value} {fmt(a.amount)} ({a.status})")
    for case in cases:
        for line in case.lines:
            figures.append(
                f"Disputed: {line.description} — {fmt(line.disputed_amount)} "
                f"of {fmt(line.amount)} billed (dispute {case.case_id})")
    figures = list(dict.fromkeys(figures))
    # The account's open promises and complaints (round 17),
    # from the round 16 ledger: the human inherits the
    # commitments too. Computable only once the account is
    # known — `open_state_known` records which case this is,
    # so an empty list is never mistaken for "none open".
    open_promises: list[str] = []
    open_state_known = bool(conv.account_number)
    if conv.account_number:
        for entry in continuity.promise_ledger(store, conv.account_number):
            if entry.state in ("pending", "broken"):
                suffix = " (overdue)" if entry.state == "broken" else ""
                open_promises.append(
                    f"{entry.summary} — due {entry.due_date}{suffix}")
    packet = HandoffPacket(
        conversation_id=conv.conversation_id,
        queue=queue,
        reason=reason,
        identity_state=conv.identity_state.value,
        identity_attempts=conv.identity_attempts,
        issue=first_customer[:200],
        vulnerable=vulnerable,
        customer_name=customer.full_name if customer else "",
        account_number=conv.account_number or "",
        summary=summary,
        attempts=attempts,
        actions_taken=taken,
        open_items=open_items,
        tried=tried,
        figures=figures,
        open_promises=open_promises,
        open_state_known=open_state_known,
        signals=sig_summary,
        known_incidents=known_incidents,
    )
    # The completeness grade (round 17): computed here, at the
    # one place packets are built, so every route to the queue
    # is graded by the same rules.
    return handoffs.grade_packet(packet, conv.intent.value)
