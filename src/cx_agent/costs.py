"""Cost & latency discipline (round 18) — the accounting.

Every model call the supervisor makes is a record before it is
a number. The boundary (llm.py) logs each attempt — its kind
(`decision`: the compose call that produces the turn's answer;
`advisory`: the router's tie-break hint), its tokens with their
basis (`measured` where the provider reported usage, `estimated`
where the deterministic estimate stood in — the offline backend
always estimates), its outcome (ok, error, timeout) and its
wall-clock latency — and the supervisor writes those records
onto the conversation's audit trail, stamped with the day and
the desk that owned the turn.

This module is the fold over those records, and the only one:
per-conversation usage (which the budget check reads at the
start of every turn), the operator report (overall, per desk,
per day), and the daily digest's cost section. The figures
reconcile by construction — the conversation totals are the
record sums, and the report totals are the conversation sums —
because nothing anywhere keeps a second set of books: no
in-memory counters, no parallel tallies. A conversation the
records say nothing about costs nothing here, however long it
ran before this accounting existed.
"""

from __future__ import annotations

import math

from .config import Policy

CALL_AUDIT_KIND = "model_call"
BUDGET_AUDIT_KIND = "budget"
TIMEOUT_REASON = "model_timeout"


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------

def _records_for(store, conversation_id: str) -> list[dict]:
    return [{"conversation_id": conversation_id, **event.detail}
            for event in store.list_audit(conversation_id)
            if event.kind == CALL_AUDIT_KIND]


def call_records(store, conversation_id: str | None = None) -> list[dict]:
    """Every recorded model call — one conversation's, or the
    whole book's in conversation order."""
    if conversation_id is not None:
        return _records_for(store, conversation_id)
    rows: list[dict] = []
    for conv in store.list_conversations():
        rows.extend(_records_for(store, conv.conversation_id))
    return rows


def _budget_exhaustions(store, conversation_ids) -> int:
    count = 0
    for cid in conversation_ids:
        count += sum(
            1 for event in store.list_audit(cid)
            if event.kind == BUDGET_AUDIT_KIND
            and event.detail.get("event") == "budget_exhausted")
    return count


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------

def _fold(records: list[dict]) -> dict:
    """One set of call records → its accounting: the call counts
    by kind, the tokens split by basis (a token is never counted
    twice — its record carries exactly one basis), the latency
    mean and p95 over every attempt, and the outcome counts."""
    n = len(records)
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "calls": n,
        "decision_calls": sum(
            1 for r in records if r["call_kind"] == "decision"),
        "advisory_calls": sum(
            1 for r in records if r["call_kind"] == "advisory"),
        "tokens": sum(r["tokens"] for r in records),
        "tokens_estimated": sum(
            r["tokens"] for r in records
            if r["tokens_basis"] == "estimated"),
        "tokens_measured": sum(
            r["tokens"] for r in records
            if r["tokens_basis"] == "measured"),
        "mean_latency_ms": (round(sum(latencies) / n, 2) if n else None),
        "p95_latency_ms": (latencies[max(0, math.ceil(0.95 * n) - 1)]
                           if n else None),
        "timeouts": sum(1 for r in records if r["outcome"] == "timeout"),
        "errors": sum(1 for r in records if r["outcome"] == "error"),
    }


def conversation_usage(store, conversation_id: str) -> dict:
    """One conversation's spend, folded from its call records —
    the numbers the budget check reads at the start of a turn."""
    folded = _fold(_records_for(store, conversation_id))
    return {key: folded[key] for key in (
        "calls", "decision_calls", "advisory_calls", "tokens",
        "tokens_estimated", "tokens_measured")}


# ---------------------------------------------------------------------------
# The report and the digest section
# ---------------------------------------------------------------------------

def cost_report(store, policy: Policy) -> dict:
    """The operator report over the whole book: the overall
    fold, the same fold per desk and per day, the closings the
    discipline produced (budget exhaustions read from their
    audit events, timeout closings from the escalation
    records), and the governed caps the spend ran under."""
    records = call_records(store)
    by_desk: dict[str, list[dict]] = {}
    by_day: dict[str, list[dict]] = {}
    for record in records:
        by_desk.setdefault(record.get("desk", ""), []).append(record)
        by_day.setdefault(record.get("date", ""), []).append(record)
    conversation_ids = [c.conversation_id
                        for c in store.list_conversations()]
    return {
        "overall": _fold(records),
        "by_desk": {desk: _fold(rows)
                    for desk, rows in sorted(by_desk.items())},
        "by_day": {day: _fold(rows)
                   for day, rows in sorted(by_day.items())},
        "budget_exhaustions": _budget_exhaustions(store, conversation_ids),
        "timeout_escalations": sum(
            1 for e in store.list_escalations()
            if e.reason == TIMEOUT_REASON),
        "caps": {
            "max_decision_calls_per_conversation":
                policy.max_decision_calls_per_conversation,
            "max_estimated_tokens_per_conversation":
                policy.max_estimated_tokens_per_conversation,
            "model_call_timeout_seconds":
                policy.model_call_timeout_seconds,
        },
        "note": ("Folded from the model-call records on the "
                 "conversations' audit trails. Tokens are split "
                 "by basis: measured where the provider reported "
                 "usage, estimated by the deterministic estimate "
                 "everywhere else — the offline backend always "
                 "estimates."),
    }


def digest_section(store, conversation_ids: set) -> dict:
    """The daily digest's cost section: the day's conversations'
    call records folded with the report's own rules, plus the
    day's closings inside the same scope — budget exhaustions
    from their audit events, timeout closings from the day's
    escalation records."""
    rows: list[dict] = []
    for cid in sorted(conversation_ids):
        rows.extend(_records_for(store, cid))
    folded = _fold(rows)
    folded["budget_exhaustions"] = _budget_exhaustions(
        store, sorted(conversation_ids))
    folded["timeout_escalations"] = sum(
        1 for e in store.list_escalations()
        if e.conversation_id in conversation_ids
        and e.reason == TIMEOUT_REASON)
    return folded
