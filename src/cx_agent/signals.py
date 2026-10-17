"""Conversation signals: what the conversation itself says about
how the customer is experiencing it, computed in code from the
conversation's own records after every customer turn — never
judged by a model.

Four signals, each a pure function of records:

- **Frustration.** Every customer message is scored against a
  governed negative-affect lexicon (below; each entry counts once
  per message, weighted by strength). The series of scores is the
  trend. The series is RISING when the latest message scored at
  least 2, beat the message before it (which itself scored), and
  beat every score before those two — the customer is at their
  most frustrated right now, and it built over at least two
  turns. A plateau is not a rise, and a peak that already passed
  is not a rise either.
- **Effort.** A customer-effort measure from records: one point
  per turn taken, one per identity attempt (each code the
  customer had to submit, right or wrong), one per
  reroute, and one per repeated ask — a customer message whose
  token overlap with their previous message reaches the policy
  threshold. Everything the customer had to do beyond speaking
  once costs exactly one; nothing is weighted twice. The
  components ride the record beside the score, so the number is
  explainable.
- **Progress.** Whether this turn landed, read off the turn's own
  result: findings returned, and whether an action executed.
- **Confusion.** Explicit not-understanding phrases, or a
  repeated question (the same overlap measure), pinned to the
  topic of the last explanation a desk gave — the topic the
  supervisor tracks from the desks' citations, or the bill
  explanation the billing desk marks itself.

Signals persist per turn as SignalRecords: the latest record is
the conversation's snapshot, the list is its series. The
interventions that act on signals — offering a person early,
serving a second explanation form — are rules in the supervisor
over these functions; every firing and every answer to an offer
is an audit event (kind "signal"), and this module folds that
trail into the handoff summary and the operator report. This
module computes; it does not act.
"""

from __future__ import annotations

import re

from .config import Policy
from .documents import _tokens
from .models import Conversation, SignalRecord

# ---------------------------------------------------------------------------
# Frustration: the governed negative-affect lexicon
# ---------------------------------------------------------------------------

_NEGATIVE_AFFECT: list[tuple[str, int]] = [
    # (pattern, weight) — weight 3 is the strongest language,
    # 1 is mild displeasure. Each pattern counts once per message.
    (r"\bfurious\b", 3),
    (r"\bdisgraceful\b", 3),
    (r"\bappalling\b", 3),
    (r"\bfrustrat\w*\b", 2),
    (r"\bangry\b", 2),
    (r"\bridiculous\b", 2),
    (r"\bunacceptable\b", 2),
    (r"\buseless\b", 2),
    (r"\bterrible\b", 2),
    (r"\bawful\b", 2),
    (r"\bfed up\b", 2),
    (r"\bsick of\b", 2),
    (r"\bwaste of time\b", 2),
    (r"\bincompetent\b", 2),
    (r"\bthis is a joke\b", 2),
    (r"\bpathetic\b", 2),
    (r"\boutrageous\b", 2),
    (r"\bannoy\w*\b", 1),
    (r"\bupset\b", 1),
    (r"\bunhappy\b", 1),
    (r"\bdisappointed\b", 1),
    (r"\bnot happy\b", 1),
    (r"\bhopeless\b", 1),
]

_LEXICON = [(re.compile(p), w) for p, w in _NEGATIVE_AFFECT]


def frustration_score(text: str) -> int:
    """One customer message against the lexicon."""
    lowered = text.lower()
    return sum(w for pattern, w in _LEXICON if pattern.search(lowered))


def customer_texts(conv: Conversation) -> list[str]:
    """The customer's own words, in turn order."""
    return [m.text for m in sorted(
        (m for m in conv.messages if m.role == "customer"),
        key=lambda m: m.turn)]


def frustration_series(conv: Conversation) -> list[int]:
    """The lexicon score of every customer message, in order —
    the raw material of the trend."""
    return [frustration_score(t) for t in customer_texts(conv)]


def is_rising(scores: list[int]) -> bool:
    """The governed rise: the latest score is real (>= 2), the
    message before it scored too, the latest beat it, and the
    latest beats everything earlier — frustration that is
    building now, not a plateau and not a peak already past."""
    if len(scores) < 2:
        return False
    last, prev = scores[-1], scores[-2]
    if last < 2 or prev < 1 or last <= prev:
        return False
    return last > max(scores[:-2], default=0)


def frustration_trend(scores: list[int]) -> str:
    """The one-word read of the series for humans: rising, easing
    (a peak behind us, quieter now), steady (present, unmoving),
    or calm (nothing scored at all)."""
    if not scores or all(s == 0 for s in scores):
        return "calm"
    if is_rising(scores):
        return "rising"
    if scores[-1] < max(scores):
        return "easing"
    return "steady"


# ---------------------------------------------------------------------------
# Confusion: said outright, or the same question again
# ---------------------------------------------------------------------------

_CONFUSION_PHRASES = [
    r"\bdon'?t understand\b",
    r"\bdo not understand\b",
    r"\bnot sure (?:what|i)\b",
    r"\bwhat do you mean\b",
    r"\bwhat does (?:that|this) mean\b",
    r"\bdoesn'?t make sense\b",
    r"\bdoes not make sense\b",
    r"\bi'?m confused\b",
    r"\bso confused\b",
    r"\bconfusing\b",
    r"\bplain english\b",
    r"\bplain language\b",
    r"\bsay that again\b",
    r"\bexplain (?:that|this|it) again\b",
    r"\bcan you explain\b",
    r"\byou'?ve lost me\b",
    r"\btoo technical\b",
    r"\bjargon\b",
]

_CONFUSION = [re.compile(p) for p in _CONFUSION_PHRASES]


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap over the repo's own content tokens — the
    repeat measure for both confusion and effort."""
    left, right = set(_tokens(a)), set(_tokens(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def detect_confusion(text: str, conv: Conversation,
                     policy: Policy) -> tuple[bool, str, str]:
    """(confused, topic, via) for the customer's latest message.

    `via` is "explicit" (a not-understanding phrase), "repeat"
    (the same question as their previous message, by the overlap
    measure), or "". The topic is the last explanation's — from
    the supervisor's signal state — and is "" when nothing has
    been explained yet: confusion with nothing explained is real,
    but there is no second form to serve for it."""
    lowered = text.lower()
    sig = conv.specialist_state.get("signals") or {}
    topic = sig.get("explained_topic", "")
    if any(p.search(lowered) for p in _CONFUSION):
        return True, topic, "explicit"
    texts = customer_texts(conv)
    if len(texts) >= 2 and texts[-1] == text:
        if token_overlap(texts[-1], texts[-2]) >= policy.confusion_repeat_overlap:
            return True, topic, "repeat"
    return False, topic, ""


# ---------------------------------------------------------------------------
# Effort: everything the customer had to do, counted once each
# ---------------------------------------------------------------------------


def repeat_count(conv: Conversation, policy: Policy) -> int:
    """Adjacent customer messages that ask the same thing again,
    by the overlap measure."""
    texts = customer_texts(conv)
    return sum(
        1 for a, b in zip(texts, texts[1:])
        if token_overlap(a, b) >= policy.confusion_repeat_overlap)


def effort_breakdown(conv: Conversation, audit: list,
                     policy: Policy) -> dict:
    """The effort score and the components it is made of. The
    reroute count comes from the audit trail — a reroute is an
    event, not a state — everything else is the conversation
    record itself."""
    parts = {
        "turns": conv.turns,
        "identity_attempts": conv.identity_attempts,
        "reroutes": sum(1 for e in audit if e.kind == "reroute"),
        "repeats": repeat_count(conv, policy),
    }
    parts["score"] = (parts["turns"] + parts["identity_attempts"]
                      + parts["reroutes"] + parts["repeats"])
    return parts


# ---------------------------------------------------------------------------
# The per-turn record, the handoff summary, the operator report
# ---------------------------------------------------------------------------


def snapshot(store, conv: Conversation, policy: Policy, *,
             progress: bool, findings_count: int, action_executed: bool,
             confusion: tuple[bool, str, str],
             interventions: list[dict]) -> SignalRecord:
    """The SignalRecord for the turn that just closed, computed
    from the conversation as it now stands."""
    scores = frustration_series(conv)
    audit = store.list_audit(conv.conversation_id)
    eff = effort_breakdown(conv, audit, policy)
    confused, topic, _via = confusion
    return SignalRecord(
        record_id=f"SIG-{conv.conversation_id}-{conv.turns}",
        conversation_id=conv.conversation_id,
        turn=conv.turns,
        desk=(conv.specialist_state.get("desk") or conv.intent.value),
        frustration_score=scores[-1] if scores else 0,
        frustration_rising=is_rising(scores),
        effort_score=eff["score"],
        effort_turns=eff["turns"],
        effort_identity_attempts=eff["identity_attempts"],
        effort_reroutes=eff["reroutes"],
        effort_repeats=eff["repeats"],
        progress=progress,
        findings_count=findings_count,
        action_executed=action_executed,
        confusion=confused,
        confusion_topic=topic if confused else "",
        interventions=interventions,
    )


def intervention_history(audit: list) -> list[dict]:
    """The intervention trail, folded from the signal audit
    events: each offer by kind with its latest outcome (offered,
    then accepted / declined / lapsed), each second form as it
    was applied — chronological by first firing."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for event in audit:
        if event.kind != "signal":
            continue
        kind = event.detail.get("kind", "")
        if not kind:
            continue
        if kind not in merged:
            merged[kind] = {"kind": kind,
                            "outcome": event.detail.get("outcome", ""),
                            "turn": event.detail.get("turn", 0)}
            order.append(kind)
        elif event.detail.get("event") == "intervention":
            merged[kind]["outcome"] = event.detail.get("outcome", "")
    return [merged[k] for k in order]


def packet_summary(store, conv: Conversation,
                   policy: Policy | None = None) -> dict:
    """The signal summary the handoff packet carries: how much
    this conversation cost the customer, how their frustration
    moved, and what the signal rules already tried — computed
    live from the record, so it is true at the moment of
    escalation, not only after the turn's bookkeeping."""
    policy = policy or Policy()
    scores = frustration_series(conv)
    audit = store.list_audit(conv.conversation_id)
    eff = effort_breakdown(conv, audit, policy)
    history = intervention_history(audit)
    return {
        "turns": conv.turns,
        "effort_score": eff["score"],
        "effort": {k: eff[k] for k in
                   ("turns", "identity_attempts", "reroutes", "repeats")},
        "frustration_trend": frustration_trend(scores),
        "frustration_latest": scores[-1] if scores else 0,
        "interventions": [
            {"kind": h["kind"], "outcome": h["outcome"]} for h in history],
        "second_forms_used": [
            h["kind"].split(":", 1)[1] for h in history
            if h["kind"].startswith("second_form:")],
    }


def signal_report(store, policy: Policy) -> dict:
    """Aggregates over every stored conversation's signal
    records, for the operators watching the experience, not just
    the outcomes: average effort by the desk a conversation
    entered at, the share of conversations whose frustration
    rose at any point, what the interventions did, and the
    frustration/escalation table — counts side by side,
    descriptive only: the table shows the two together, it does
    not claim one caused the other."""
    records = store.list_signal_records()
    by_conv: dict[str, list[SignalRecord]] = {}
    for record in records:
        by_conv.setdefault(record.conversation_id, []).append(record)
    escalated = {e.conversation_id for e in store.list_escalations()}

    effort_by_desk: dict[str, list[int]] = {}
    rising_convs = 0
    offered = accepted = declined = lapsed = second_forms = 0
    cells = {"rising_and_escalated": 0, "rising_not_escalated": 0,
             "not_rising_and_escalated": 0, "not_rising_not_escalated": 0}
    total_effort = 0
    for conv_id, rows in by_conv.items():
        rows.sort(key=lambda r: r.turn)
        effort_by_desk.setdefault(rows[0].desk, []).append(rows[-1].effort_score)
        total_effort += rows[-1].effort_score
        rose = any(r.frustration_rising for r in rows)
        if rose:
            rising_convs += 1
        key = ("rising" if rose else "not_rising") + \
            ("_and_escalated" if conv_id in escalated else "_not_escalated")
        cells[key] += 1
        for row in rows:
            for entry in row.interventions:
                kind, outcome = entry.get("kind", ""), entry.get("outcome", "")
                if kind.startswith("second_form:"):
                    second_forms += 1
                elif outcome == "offered":
                    offered += 1
                elif outcome == "accepted":
                    accepted += 1
                elif outcome == "declined":
                    declined += 1
                elif outcome == "lapsed":
                    lapsed += 1
    n = len(by_conv)
    decided = accepted + declined
    return {
        "conversations": n,
        "avg_effort": round(total_effort / n, 2) if n else 0.0,
        "avg_effort_by_entry_desk": {
            desk: round(sum(v) / len(v), 2)
            for desk, v in sorted(effort_by_desk.items())},
        "rising_frustration_conversations": rising_convs,
        "rising_frustration_share": round(rising_convs / n, 3) if n else 0.0,
        "interventions": {
            "offered": offered, "accepted": accepted,
            "declined": declined, "lapsed": lapsed,
            "second_forms": second_forms,
            "acceptance_rate": (round(accepted / decided, 3)
                                if decided else None),
        },
        "frustration_escalation": cells,
        "note": ("Descriptive counts over stored conversations. The "
                 "frustration/escalation table places the two side by "
                 "side; it is not evidence that one caused the other."),
    }
