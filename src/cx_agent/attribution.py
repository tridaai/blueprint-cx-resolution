"""Failure attribution + the annotation queue.

Every conversation ends somehow. This module records how — an
outcome, and for the endings that weren't a resolution, one
primary cause from a governed taxonomy — computed deterministically
from the conversation's own audit trail and sibling records. No
model judges anything: each cause is a rule over recorded events,
the rule that fired is named on the record, and the signals it
read ride along as evidence, so any attribution can be checked
against the trail by a human who disagrees with it.

The rules, in the order they are read (first match wins — the
order is the argument: a machinery failure outranks everything,
a hard identity failure outranks the desk-level stories, and a
designed escalation is only called designed when nothing more
specific fired first):

1. `system_error` — the trail carries an internal-error event
   (the service writes one when a turn crashes).
2. `identity_friction` (hard) — verification locked out: the
   escalation reason, or the identity state itself, says failed.
3. `policy_limit` — a cap or refusal rule decided the outcome:
   a dispute over the agent's cap, a reversal outside its
   window, a broken arrangement (a human's case by policy), a
   fault-driven exit routed to billing by policy, a goodwill
   refusal, or an action blocked on a policy reason.
4. `procedure_dead_end` — a troubleshooting walk exhausted
   (callback proposed or not), a procedure escalation (safety
   step refused, a step failed, a field visit required), or a
   no-progress escalation with a walk in flight.
5. Designed escalations — threat, fraud, legal, vulnerability,
   the customer's second ask for a human, a conversation budget
   spent (round 18), a model call that timed out twice
   (round 18). These are the system working as designed, not
   failing: cause is None and the rule says so
   (`escalated_by_design:<reason>`).
6. `identity_friction` (soft) — the conversation ended still
   waiting at the verification step: verification abandoned.
7. `routing_error` — rerouted two or more times and nothing
   executed: the conversation was bounced, not served.
8. `missing_capability` — the care desk logged a need it cannot
   serve, or the router fell back to care on low confidence:
   there is no desk for what was asked.
9. `customer_left` — work was left pending: a proposal awaiting
   its answer, or a procedure walk mid-step.
10. Otherwise the trail gives no rule anything true to say:
    cause None, rule `no_progress_escalation` for a bare
    no-progress escalation, else `ended_without_pending_work`.

Attribution runs two ways, both operator acts in the repo's
pattern (like the complaint watch): a pass attributes every
conversation that has closed (resolved / escalated) and doesn't
have a record yet; a single-conversation attribution covers the
ones that simply stopped — a still-open record the operator
declares ended, which is how `customer_left` gets its record.
A record, once written, stands: later passes never rewrite it.

The annotation queue is the human loop over these records.
Every escalated or unresolved conversation is queued, plus a
deterministic sample of the resolved ones. An annotator marks
the attributed cause right or corrects it from the same
taxonomy; a second annotator, never the same person, does the
same; agreement between the first two is measured (simple
agreement and Cohen's kappa over the cause labels), and where
they disagree a third person settles it with a verdict. The
report closes the loop back to design: the top unhandled-intent
mining candidates carry the attributed-cause counts of their
member conversations, so "what should we build next" reads off
evidence, not impressions.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from datetime import timedelta

from . import mining
from .clock import today
from .models import AnnotationRecord, AttributionRecord

# The governed taxonomy. One primary cause per non-resolved
# conversation, from this list or none.
CAUSES = (
    "routing_error",
    "identity_friction",
    "policy_limit",
    "missing_capability",
    "procedure_dead_end",
    "customer_left",
    "system_error",
)

OUTCOMES = (
    "resolved",
    "resolved_with_action",
    "escalated",
    "unresolved_ended",
)

# Escalation reasons that are the design working, not a failure:
# the rules demanded a human whatever the desks could have done.
# Round 18 adds the cost closings: a budget spent and a model
# that timed out twice both hand the conversation to a person
# by design — the discipline working, not the agent failing.
DESIGNED_ESCALATION_REASONS = {
    "safety_threat",
    "fraud_report",
    "legal_matter",
    "vulnerable_customer",
    "customer_requested_human",
    "budget_exhausted",
    "model_timeout",
}

# Escalation reasons a policy rule owns outright: the outcome was
# decided by the cap, the window, or the routing policy — the
# desk never had a move.
POLICY_ESCALATION_RULES = {
    "dispute_above_agent_cap": "dispute_above_cap",
    "plan_reversal_outside_window": "reversal_outside_window",
    "broken_payment_arrangement": "broken_arrangement_human_only",
    "fault_cancellation_dispute_path": "fault_exit_routed_by_policy",
}

# Escalation reasons a procedure owns: the walk itself ended the
# conversation.
PROCEDURE_ESCALATION_REASONS = {
    "safety_step_refused",
    "procedure_step_failed",
    "field_visit_required",
}

# Action block reasons that are policy speaking (caps, windows,
# source-only money rules) — as opposed to state (no account, no
# balance) or the identity/confirmation gates, which have their
# own causes.
POLICY_BLOCK_REASONS = {
    "credit_above_cap",
    "goodwill_rolling_cap_reached",
    "hard_decline_retry_refused",
    "refund_destination_not_on_file",
    "balance_below_arrangement_minimum",
    "arrangement_broken_human_only",
    "arrangement_already_active",
    "reversal_window_closed",
    "plan_not_eligible",
}


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _signals(store, conv) -> dict:
    """Everything the rules read, gathered once from the record:
    the audit trail, the action log, the dispute cases, the
    procedure runs, the escalation that closed the conversation."""
    cid = conv.conversation_id
    audit = store.list_audit(cid)
    actions = store.list_actions(cid)
    own_escalations = [e for e in store.list_escalations()
                       if e.conversation_id == cid
                       and e.reason != "complaint_sla_breach"]
    runs = store.list_procedure_runs(conversation_id=cid)
    return {
        "status": conv.status,
        "turns": conv.turns,
        "executed_kinds": sorted(a.kind.value for a in actions
                                 if a.status == "executed"),
        "pending_action": conv.pending_action is not None,
        "awaiting_identity": conv.specialist_state.get("awaiting"),
        "identity_state": conv.identity_state.value,
        "unhandled_intent": bool(conv.unhandled_intent),
        "escalation_reason": (own_escalations[-1].reason
                              if own_escalations else ""),
        "reroutes": sum(1 for e in audit if e.kind == "reroute"),
        "low_confidence_turns": sum(
            1 for e in audit if e.kind == "turn"
            and (e.detail.get("routing") or {}).get("decision")
            == "low_confidence_care"),
        "internal_errors": sum(
            1 for e in audit if e.kind == "system"
            and e.detail.get("event") == "internal_error"),
        "goodwill_refused": any(
            e.kind == "billing" and e.detail.get("event") == "goodwill_refused"
            for e in audit),
        "policy_blocks": [a.block_reason for a in actions
                          if a.status == "blocked"
                          and a.block_reason in POLICY_BLOCK_REASONS],
        "dispute_states": [d.state for d in
                           store.list_dispute_cases(conversation_id=cid)],
        "run_states": [r.state for r in runs],
    }


def compute_attribution(store, conv) -> AttributionRecord:
    """The outcome and the cause, from the signals. Pure: reads
    the store, writes nothing."""
    s = _signals(store, conv)
    desk = conv.specialist_state.get("desk") or conv.intent.value
    evidence = {k: v for k, v in s.items() if k != "status"}

    if s["status"] == "escalated":
        outcome = "escalated"
    elif s["pending_action"]:
        # The record may say resolved — an earlier need was met —
        # but the conversation's last word is a proposal nobody
        # answered. Its business is unfinished.
        outcome = "unresolved_ended"
    elif s["status"] == "resolved" and s["executed_kinds"]:
        outcome = "resolved_with_action"
    elif s["status"] == "resolved":
        outcome = "resolved"
    else:
        outcome = "unresolved_ended"

    if outcome in ("resolved", "resolved_with_action"):
        return AttributionRecord(
            conversation_id=conv.conversation_id, outcome=outcome,
            cause=None, rule=outcome, desk=desk, evidence=evidence,
            attributed_date=today().isoformat())

    reason = s["escalation_reason"]
    cause: str | None = None
    rule = ""
    if s["internal_errors"]:
        cause, rule = "system_error", "internal_error_event"
    elif reason == "identity_verification_failed" \
            or s["identity_state"] == "failed":
        cause, rule = "identity_friction", "identity_lockout"
    elif reason in POLICY_ESCALATION_RULES:
        cause, rule = "policy_limit", POLICY_ESCALATION_RULES[reason]
    elif "escalated_above_cap" in s["dispute_states"]:
        cause, rule = "policy_limit", "dispute_above_cap"
    elif s["goodwill_refused"]:
        cause, rule = "policy_limit", "goodwill_refused"
    elif s["policy_blocks"]:
        cause, rule = "policy_limit", f"action_blocked:{s['policy_blocks'][0]}"
    elif "dead_ended" in s["run_states"]:
        cause, rule = "procedure_dead_end", "procedure_exhausted"
    elif reason in PROCEDURE_ESCALATION_REASONS:
        cause, rule = "procedure_dead_end", reason
    elif reason == "no_progress" and s["run_states"]:
        cause, rule = "procedure_dead_end", "procedure_stalled_no_progress"
    elif reason in DESIGNED_ESCALATION_REASONS:
        rule = f"escalated_by_design:{reason}"
    elif s["awaiting_identity"] and s["identity_state"] != "verified":
        cause, rule = "identity_friction", "verification_abandoned"
    elif s["reroutes"] >= 2 and not s["executed_kinds"]:
        cause, rule = "routing_error", "rerouted_without_progress"
    elif s["unhandled_intent"]:
        cause, rule = "missing_capability", "unhandled_intent_logged"
    elif s["low_confidence_turns"]:
        cause, rule = "missing_capability", "low_confidence_care_fallback"
    elif s["pending_action"]:
        cause, rule = "customer_left", "left_with_action_pending"
    elif "active" in s["run_states"]:
        cause, rule = "customer_left", "left_mid_procedure"
    elif reason == "no_progress":
        cause, rule = "missing_capability", "no_progress_escalation"
    else:
        rule = "ended_without_pending_work"

    return AttributionRecord(
        conversation_id=conv.conversation_id, outcome=outcome,
        cause=cause, rule=rule, desk=desk, evidence=evidence,
        attributed_date=today().isoformat())


def attribute_conversation(store, conversation_id: str) -> AttributionRecord | None:
    """Attribute one conversation, whatever state it ended in —
    the operator declaring it over. An existing record stands:
    attribution is a snapshot at close, never rewritten."""
    existing = store.get_attribution(conversation_id)
    if existing is not None:
        return existing
    conv = store.get_conversation(conversation_id)
    if conv is None:
        return None
    record = compute_attribution(store, conv)
    store.save_attribution(record)
    return record


def run_attribution(store) -> dict:
    """The pass: attribute every conversation that has closed
    (resolved or escalated) and has no record yet. Conversations
    still open are left alone — they haven't ended; one that
    merely stopped is attributed singly, by name."""
    attributed, already, still_open = [], 0, 0
    for conv in sorted(store.list_conversations(),
                       key=lambda c: c.conversation_id):
        if store.get_attribution(conv.conversation_id) is not None:
            already += 1
        elif conv.status == "active":
            still_open += 1
        else:
            record = compute_attribution(store, conv)
            store.save_attribution(record)
            attributed.append(conv.conversation_id)
    return {"attributed": attributed, "already_attributed": already,
            "still_open": still_open}


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _window_bounds(days: int) -> tuple[tuple[str, str], tuple[str, str]]:
    end = today()
    cur_start = end - timedelta(days=days - 1)
    prev_end = cur_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return (cur_start.isoformat(), end.isoformat()), \
        (prev_start.isoformat(), prev_end.isoformat())


def _bucket(records: list[AttributionRecord]) -> dict:
    by_outcome = Counter(r.outcome for r in records)
    by_cause = Counter(r.cause for r in records if r.cause)
    by_cause_desk: dict[str, Counter] = {}
    designed = no_cause = 0
    for r in records:
        if r.cause:
            by_cause_desk.setdefault(r.cause, Counter())[r.desk] += 1
        elif r.outcome in ("escalated", "unresolved_ended"):
            if r.rule.startswith("escalated_by_design"):
                designed += 1
            else:
                no_cause += 1
    return {
        "total": len(records),
        "by_outcome": {o: by_outcome.get(o, 0) for o in OUTCOMES},
        "by_cause": {c: by_cause.get(c, 0) for c in CAUSES},
        "by_cause_desk": {cause: dict(sorted(desks.items()))
                          for cause, desks in sorted(by_cause_desk.items())},
        "by_design": designed,
        "no_cause": no_cause,
    }


def attribution_report(store, policy, days: int = 7) -> dict:
    """Counts by cause × desk over a window of days, with the
    previous window beside them — and the mining join: the top
    unhandled-intent candidates with the attributed causes of
    their member conversations attached."""
    (cur_lo, cur_hi), (prev_lo, prev_hi) = _window_bounds(days)
    records = store.list_attributions()
    current = [r for r in records if cur_lo <= r.attributed_date <= cur_hi]
    previous = [r for r in records if prev_lo <= r.attributed_date <= prev_hi]
    cur, prev = _bucket(current), _bucket(previous)

    pairs = set()
    for bucket in (cur, prev):
        for cause, desks in bucket["by_cause_desk"].items():
            pairs.update((cause, desk) for desk in desks)
    rows = [{
        "cause": cause, "desk": desk,
        "current": cur["by_cause_desk"].get(cause, {}).get(desk, 0),
        "previous": prev["by_cause_desk"].get(cause, {}).get(desk, 0),
        "delta": cur["by_cause_desk"].get(cause, {}).get(desk, 0)
        - prev["by_cause_desk"].get(cause, {}).get(desk, 0),
    } for cause, desk in sorted(pairs)]
    rows.sort(key=lambda r: (-r["current"], r["cause"], r["desk"]))

    by_id = {r.conversation_id: r for r in records}
    mining_evidence = []
    for candidate in mining.mine_unhandled_intents(store, policy)["candidates"]:
        counts = Counter()
        for cid in candidate["conversation_ids"]:
            record = by_id.get(cid)
            if record is None:
                counts["unattributed"] += 1
            elif record.cause in ("missing_capability", "routing_error"):
                counts[record.cause] += 1
            elif record.cause:
                counts["other_cause"] += 1
            else:
                counts["no_cause"] += 1
        mining_evidence.append({
            "label": candidate["label"],
            "count": candidate["count"],
            "missing_capability": counts.get("missing_capability", 0),
            "routing_error": counts.get("routing_error", 0),
            "other_cause": counts.get("other_cause", 0),
            "no_cause": counts.get("no_cause", 0),
            "unattributed": counts.get("unattributed", 0),
        })

    return {
        "window_days": days,
        "current_window": [cur_lo, cur_hi],
        "previous_window": [prev_lo, prev_hi],
        "current": cur,
        "previous": prev,
        "cause_desk_rows": rows,
        "mining_evidence": mining_evidence,
    }


# ---------------------------------------------------------------------------
# The annotation queue
# ---------------------------------------------------------------------------


def _sampled(conversation_id: str, modulus: int) -> bool:
    if modulus <= 1:
        return True
    digest = hashlib.sha256(conversation_id.encode()).hexdigest()
    return int(digest, 16) % modulus == 0


def _effective_cause(annotation: AnnotationRecord,
                     attributed_cause: str | None) -> str | None:
    """The cause an annotation stands for: the attributed one
    when it says the attribution is right, its correction when
    it doesn't."""
    return attributed_cause if annotation.cause_correct \
        else annotation.corrected_cause


def _pair(store, conversation_id: str) -> list[AnnotationRecord]:
    """The first two non-verdict annotations, in creation
    order — the pair agreement is measured on."""
    rows = [a for a in store.list_annotations(conversation_id)
            if not a.is_verdict]
    return rows[:2]


def _queue_state(store, record: AttributionRecord) -> str:
    anns = store.list_annotations(record.conversation_id)
    if any(a.is_verdict for a in anns):
        return "settled"
    pair = _pair(store, record.conversation_id)
    if len(pair) == 2 and _effective_cause(pair[0], record.cause) \
            != _effective_cause(pair[1], record.cause):
        return "disputed"
    return "annotated" if anns else "awaiting"


def annotation_queue(store, policy) -> list[dict]:
    """Who reviews what: every escalated or unresolved
    conversation, plus the deterministic sample of resolved
    ones, each with its attribution and its review state."""
    rows = []
    for record in store.list_attributions():
        needs_review = record.outcome in ("escalated", "unresolved_ended")
        sampled = record.outcome in ("resolved", "resolved_with_action") \
            and _sampled(record.conversation_id,
                         policy.annotation_sample_modulus)
        if not (needs_review or sampled):
            continue
        anns = store.list_annotations(record.conversation_id)
        rows.append({
            "conversation_id": record.conversation_id,
            "outcome": record.outcome,
            "cause": record.cause,
            "rule": record.rule,
            "desk": record.desk,
            "attributed_date": record.attributed_date,
            "sampled": sampled and not needs_review,
            "annotations": len(anns),
            "state": _queue_state(store, record),
        })
    return rows


def annotate(store, conversation_id: str, annotator: str,
             cause_correct: bool, corrected_cause: str | None = None,
             note: str = "", is_verdict: bool = False) -> AnnotationRecord:
    """Write one annotation, enforcing the queue's rules in
    code: the conversation must be attributed; a correction is
    required exactly when the attribution is marked wrong, and
    must come from the taxonomy; a second opinion comes from a
    different person; a verdict needs a live disagreement and a
    third person, and closes the record."""
    record = store.get_attribution(conversation_id)
    if record is None:
        raise ValueError(
            f"conversation {conversation_id} has no attribution record — "
            "attribute it before annotating it")
    annotator = (annotator or "").strip()
    if not annotator:
        raise ValueError("an annotation names its annotator")
    if cause_correct and corrected_cause:
        raise ValueError(
            "a corrected cause is only for an annotation that marks "
            "the attributed cause wrong")
    if not cause_correct:
        if not corrected_cause:
            raise ValueError(
                "marking the attributed cause wrong requires the "
                "corrected cause")
        if corrected_cause not in CAUSES:
            raise ValueError(
                f"corrected cause {corrected_cause!r} is not in the "
                f"taxonomy: {', '.join(CAUSES)}")
    existing = store.list_annotations(conversation_id)
    if any(a.is_verdict for a in existing):
        raise ValueError("this attribution is settled — the verdict stands")
    prior_names = {a.annotator for a in existing if not a.is_verdict}
    if is_verdict:
        pair = _pair(store, conversation_id)
        if len(pair) < 2:
            raise ValueError("a verdict settles a disagreement — it "
                             "needs two annotations first")
        if _effective_cause(pair[0], record.cause) \
                == _effective_cause(pair[1], record.cause):
            raise ValueError("the first two annotations agree — there "
                             "is no disagreement to settle")
        if annotator in prior_names:
            raise ValueError("the verdict comes from a third person, "
                             "not from either annotator")
    elif annotator in prior_names:
        raise ValueError("a second annotation must come from a "
                         "different annotator")
    annotation = AnnotationRecord(
        annotation_id=f"ANN-{uuid.uuid4().hex[:10]}",
        conversation_id=conversation_id,
        annotator=annotator,
        cause_correct=cause_correct,
        corrected_cause=corrected_cause,
        note=note,
        is_verdict=is_verdict,
        seq=len(store.list_annotations()) + 1,
        created_date=today().isoformat(),
    )
    store.save_annotation(annotation)
    return annotation


def annotation_agreement(store) -> dict:
    """Agreement between the first two annotators, over every
    conversation that has a pair: the simple rate and Cohen's
    kappa over the cause labels (the attributed cause itself
    counts when an annotator marks it right; 'none' is a label
    like any other). Disagreements still awaiting a verdict are
    listed — that list is the lead's work queue."""
    labels = list(CAUSES) + ["none"]
    pairs: list[tuple[str, str, str]] = []  # (conversation, label1, label2)
    disagreements, settled = [], []
    for record in store.list_attributions():
        pair = _pair(store, record.conversation_id)
        if len(pair) < 2:
            continue
        first = _effective_cause(pair[0], record.cause)
        second = _effective_cause(pair[1], record.cause)
        pairs.append((record.conversation_id,
                      first or "none", second or "none"))
        has_verdict = any(a.is_verdict for a in
                           store.list_annotations(record.conversation_id))
        if first != second:
            if has_verdict:
                settled.append(record.conversation_id)
            else:
                disagreements.append({
                    "conversation_id": record.conversation_id,
                    "attributed_cause": record.cause,
                    "first": {"annotator": pair[0].annotator,
                              "cause": first},
                    "second": {"annotator": pair[1].annotator,
                               "cause": second},
                })
        elif has_verdict:
            settled.append(record.conversation_id)

    total = len(pairs)
    agreements = sum(1 for _c, a, b in pairs if a == b)
    kappa = None
    if total:
        observed = agreements / total
        first_marginal = Counter(a for _c, a, _b in pairs)
        second_marginal = Counter(b for _c, _a, b in pairs)
        expected = sum((first_marginal.get(label, 0) / total)
                       * (second_marginal.get(label, 0) / total)
                       for label in labels)
        kappa = 1.0 if expected >= 1.0 else round(
            (observed - expected) / (1 - expected), 4)
    return {
        "pairs": total,
        "agreements": agreements,
        "agreement_rate": round(agreements / total, 4) if total else None,
        "kappa": kappa,
        "disagreements": disagreements,
        "settled": settled,
    }
