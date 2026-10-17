"""Rubric trajectory gate for the CX Resolution Agent.

Plays scripted multi-turn conversations — persona drivers, in the
production guides' sense: the refund seeker, the human-demander,
the fragment writer, the AI skeptic — against the real service on
the offline fallback, and grades each trajectory against the
behavior rubrics (escalation, education, grounding, transaction,
identity, care) with pass/fail assertions, never vibes.

Gate: every zero-tolerance assertion passes (the safety invariants:
escalation decisions, whether an action executed, exact money and
plan state, identity state) and the overall pass rate is >= 95%.

Run:  uv run python evals/run_evals.py
Writes evals/results.json with the full detail.

The clock is pinned (CX_TODAY) so cycle dates — and every figure
priced off them — are exact on any day the gate runs. A turn may
also carry a "poke": the world moving under a live conversation
(a payment landing through another channel, days passing over
the action log), applied to the store before the message posts.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("CX_TODAY", "2026-10-01")

from cx_agent import care_book, mining, payments_book  # noqa: E402
from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.service import ConversationService  # noqa: E402
from cx_agent.store import InMemoryStore  # noqa: E402

# Assertion keys whose failure is never tolerable, wherever they
# appear: they encode "the agent escalated (or didn't) when it
# must", "money moved (or didn't) exactly", "identity was (or
# wasn't) established".
ZT_KEYS = {"escalated", "action_status", "balance_due", "plan_id",
           "identity_state", "payment_method_last4", "executed_kinds",
           "dispute_credit",
           # Round 5: a scheduled downgrade and a reversal pair are
           # execution state, held to the same standard.
           "pending_plan_id", "reversal_count",
           # Round 4: the procedure engine's record-keeping is held
           # to the same standard — step order, callback presence
           # and transcript contents are never a matter of degree.
           "callbacks_count", "callback_status", "callback_transcript_steps",
           "run_state", "run_current_step", "run_history_steps",
           "run_safety_refusals",
           # Round 6: a quoted figure, the guard's ordering, and the
           # fair-use record are never a matter of degree.
           "reply_ordered", "roaming_notices_count", "notice_kinds",
           "notice_unwarned", "dispute_evidence_includes",
           # Round 7: attempt outcomes and codes, arrangement
           # schedules and derived states, credit balances and
           # autopay notices are execution state, exact or failed.
           "attempt_outcomes", "attempt_codes", "arrangement_state",
           "arrangement_total", "arrangement_installments",
           "credit_balance", "payment_notices",
           # Round 8: a complaint's fields and deadline, a feedback
           # record's kind and words, the account's accessibility
           # preference, the vulnerability flag and its place in
           # the handoff, and the mined cluster's count are record
           # state — exact or failed.
           "complaints_count", "complaint_category", "complaint_status",
           "complaint_due", "complaint_breached_count",
           "feedback_count", "feedback_kind", "feedback_text",
           "customer_vulnerable", "packet_vulnerable",
           "escalation_reasons", "accessibility_preferred",
           "mining_candidates", "mining_top_count"}

_CODE = re.compile(r"\{code:(NB-\d+)\}")


def _substitute(text: str) -> str:
    return _CODE.sub(lambda m: expected_code(m.group(1)), text)


def check_turn(result, expect: dict, failures: list, label: str,
               scenario_zt: bool) -> tuple[int, int, int, int]:
    """Returns (passed, total, zt_passed, zt_total) for one turn."""
    passed = total = zt_passed = zt_total = 0
    values = {
        "intent": result.intent,
        "escalated": result.escalated,
        "escalation_reason": result.escalation_reason,
        "status": result.status,
        "identity_state": result.identity_state,
        "action_status": result.action.status if result.action else "none",
        "action_kind": result.action.kind.value if result.action else "none",
    }
    for key, want in expect.items():
        total += 1
        is_zt = scenario_zt or key in ZT_KEYS
        if is_zt:
            zt_total += 1
        if key == "reply_contains":
            ok = want in result.reply
            got = f"reply lacks {want!r}"
        elif key == "reply_contains_all":
            missing = [w for w in want if w not in result.reply]
            ok = not missing
            got = f"reply lacks {missing!r}"
        elif key == "reply_not_contains":
            ok = want not in result.reply
            got = f"reply contains {want!r}"
        elif key == "reply_ordered":
            # Each fragment present, at strictly increasing
            # positions — the projection really does lead.
            positions = [result.reply.find(w) for w in want]
            ok = all(p >= 0 for p in positions) and \
                positions == sorted(positions) and \
                len(set(positions)) == len(positions)
            got = f"positions={positions}"
        elif key == "citations_include":
            ok = want in result.citations
            got = f"citations={result.citations}"
        else:
            ok = values.get(key) == want
            got = f"{key}={values.get(key)!r}"
        if ok:
            passed += 1
            if is_zt:
                zt_passed += 1
        else:
            failures.append(f"{label}: expected {key}={want!r}, got {got}\n"
                            f"    reply: {result.reply[:200]}")
    return passed, total, zt_passed, zt_total


def check_final(service, scenario: dict, conv_id: str,
                failures: list, label: str,
                scenario_zt: bool) -> tuple[int, int, int, int]:
    passed = total = zt_passed = zt_total = 0
    final = scenario.get("final") or {}
    if not final:
        return passed, total, zt_passed, zt_total
    store = service.store
    conv = store.get_conversation(conv_id)
    account = store.get_account(scenario["account"]) if scenario.get("account") else None
    escalations = [e for e in store.list_escalations()
                   if e.conversation_id == conv_id]
    executed = sorted(a.kind.value for a in store.list_actions(conv_id)
                      if a.status == "executed")
    audit = store.list_audit(conv_id)
    cases = store.list_dispute_cases(conversation_id=conv_id)
    case = cases[-1] if cases else None
    callbacks = store.list_callbacks(conversation_id=conv_id)
    callback = callbacks[-1] if callbacks else None
    runs = store.list_procedure_runs(conversation_id=conv_id)
    run = runs[-1] if runs else None
    values = {
        "status": conv.status if conv else None,
        "unhandled_logged": bool(conv and conv.unhandled_intent),
        "escalation_queue": escalations[-1].queue if escalations else None,
        "reroute_count": sum(1 for e in audit if e.kind == "reroute"),
        "action_statuses": [a.status for a in store.list_actions(conv_id)],
        "action_statuses_sorted": sorted(
            a.status for a in store.list_actions(conv_id)),
        "executed_kinds": executed,
        "actions_count": len(store.list_actions(conv_id)),
        "balance_due": str(account.balance_due) if account else None,
        "plan_id": account.plan_id if account else None,
        "pending_plan_id": account.pending_plan_id if account else None,
        "pending_plan_effective": (account.pending_plan_effective
                                   if account else None),
        "reversal_count": sum(
            1 for a in store.list_actions(conv_id)
            if a.status == "executed" and a.payload.get("reversal_of")),
        "payment_method_last4": account.payment_method_last4 if account else None,
        "dispute_state": case.state if case else None,
        "dispute_credit": str(case.credit_applied) if case else None,
        "dispute_decisions": ([line.decision for line in case.lines]
                              if case else []),
        "callbacks_count": len(callbacks),
        "callback_status": callback.status if callback else None,
        "callback_procedure": callback.procedure_id if callback else None,
        "callback_transcript_steps": ([s.step_id for s in callback.transcript]
                                      if callback else []),
        "run_state": run.state if run else None,
        "run_current_step": run.current_step_id if run else None,
        "run_history_steps": ([h.step_id for h in run.history]
                               if run else []),
        "run_safety_refusals": run.safety_refusals if run else 0,
    }
    if scenario.get("account"):
        notices = store.list_roaming_notices(scenario["account"])
        values["roaming_notices_count"] = len(notices)
        values["notice_kinds"] = sorted(n.kind for n in notices)
        values["notice_unwarned"] = any(n.unwarned_crossing for n in notices)
        attempts = store.list_payment_attempts(scenario["account"])
        values["attempt_outcomes"] = [t.outcome for t in attempts]
        values["attempt_codes"] = [t.reason_code for t in attempts]
        arrangements = store.list_payment_arrangements(scenario["account"])
        arr = arrangements[-1] if arrangements else None
        values["arrangements_count"] = len(arrangements)
        values["arrangement_state"] = (
            payments_book.arrangement_state(arr) if arr else None)
        values["arrangement_total"] = str(arr.total) if arr else None
        values["arrangement_installments"] = (
            [f"{i.amount}@{i.due_date}" for i in arr.installments]
            if arr else [])
        values["credit_balance"] = (
            str(account.credit_balance) if account else None)
        values["autopay_enabled"] = (
            account.autopay_enabled if account else None)
        pnotices = store.list_payment_notices(scenario["account"])
        values["payment_notices"] = len(pnotices)
        values["payment_notice_amount"] = (
            str(pnotices[-1].amount) if pnotices else None)
    values["dispute_evidence"] = (" | ".join(
        line.evidence for line in case.lines) if case else "")
    # Round 8: the care desk's records and the mining pass.
    complaints = store.list_complaints(conversation_id=conv_id)
    complaint = complaints[-1] if complaints else None
    values["complaints_count"] = len(complaints)
    values["complaint_category"] = complaint.category if complaint else None
    values["complaint_status"] = complaint.status if complaint else None
    values["complaint_due"] = complaint.response_due if complaint else None
    watch = care_book.complaint_watch(store, service.policy)
    values["complaint_breached_count"] = len(watch["breached"])
    values["complaint_due_soon_count"] = len(watch["due_soon"])
    feedback_rows = store.list_feedback(conversation_id=conv_id)
    values["feedback_count"] = len(feedback_rows)
    values["feedback_kind"] = feedback_rows[-1].kind if feedback_rows else None
    values["feedback_text"] = feedback_rows[-1].text if feedback_rows else None
    values["escalations_count"] = len(escalations)
    values["escalation_reasons"] = [e.reason for e in escalations]
    values["packet_vulnerable"] = (
        escalations[-1].packet.vulnerable if escalations else None)
    if scenario.get("account"):
        customer = store.get_customer_by_account(scenario["account"])
        values["customer_vulnerable"] = (
            customer.vulnerable if customer else None)
        values["accessibility_preferred"] = (
            account.preferred_contact if account else None)
        values["accessibility_note"] = (
            account.accessibility_note if account else None)
    report = mining.mine_unhandled_intents(store, service.policy)
    values["mining_candidates"] = len(report["candidates"])
    values["mining_top_count"] = max(
        (c["count"] for c in report["candidates"]), default=0)
    values["mining_labels"] = [c["label"] for c in report["candidates"]]
    for key, want in final.items():
        total += 1
        is_zt = scenario_zt or key in ZT_KEYS
        if is_zt:
            zt_total += 1
        if key == "dispute_evidence_includes":
            ok = want in values["dispute_evidence"]
            got = values["dispute_evidence"]
        elif key == "mining_label_includes":
            ok = any(want in label for label in values["mining_labels"])
            got = values["mining_labels"]
        else:
            ok = values.get(key) == want
            got = values.get(key)
        if ok:
            passed += 1
            if is_zt:
                zt_passed += 1
        else:
            failures.append(f"{label}: final expected {key}={want!r}, "
                            f"got {got!r}")
    return passed, total, zt_passed, zt_total


def apply_poke(service, poke: dict) -> None:
    """Move the world under a live conversation, before the turn's
    message posts: set fields on an account (a payment landing
    through another channel changes the balance the same way), or
    age every action record by whole days (time passing)."""
    store = service.store
    days = int(poke.get("age_actions_days", 0))
    if days:
        for record in store.list_actions():
            made = date.fromisoformat(record.created_date)
            record.created_date = (made - timedelta(days=days)).isoformat()
            store.save_action(record)
    complaint_days = int(poke.get("age_complaints_days", 0))
    if complaint_days:
        # Time passes over the complaint book: records and their
        # deadlines age together, so a deadline that was ahead of
        # the pinned today falls behind it.
        for record in store.list_complaints():
            record.created_date = (
                date.fromisoformat(record.created_date)
                - timedelta(days=complaint_days)).isoformat()
            record.response_due = (
                date.fromisoformat(record.response_due)
                - timedelta(days=complaint_days)).isoformat()
            store.save_complaint(record)
    if poke.get("run_complaint_watch"):
        # The operator runs the SLA watch between contacts: a
        # breached complaint lands in the queue before the
        # customer's next message.
        service.run_complaint_watch()
    account_number = poke.get("account")
    if account_number:
        account = store.get_account(account_number)
        for key, value in poke.items():
            if key in ("account", "age_actions_days", "run_autopay",
                       "break_arrangements", "age_complaints_days",
                       "run_complaint_watch"):
                continue
            if key in ("balance_due", "roaming_gb_used", "credit_balance"):
                value = Decimal(str(value))
            setattr(account, key, value)
        store.save_account(account)
        if poke.get("run_autopay"):
            # The billing-side run happens between contacts: the
            # attempt and any failure notice land on the record
            # before the customer's next message.
            from cx_agent.config import Policy

            payments_book.run_autopay(store, account_number, Policy())
        if poke.get("break_arrangements"):
            # Time passes over an arrangement: its first
            # instalment's due date moves into the past, unpaid —
            # the state the calendar derives is broken.
            arrangements = store.list_payment_arrangements(account_number)
            if arrangements:
                arr = arrangements[-1]
                arr.created_date = (
                    date.fromisoformat(arr.created_date)
                    - timedelta(days=40)).isoformat()
                arr.installments[0].due_date = (
                    date.fromisoformat(os.environ["CX_TODAY"])
                    - timedelta(days=1)).isoformat()
                arr.state = "active"
                store.save_payment_arrangement(arr)


def main() -> int:
    scenarios = [json.loads(line) for line in
                 (ROOT / "evals" / "scenarios.jsonl").read_text().splitlines()
                 if line.strip()]

    failures: list[str] = []
    per_rubric: dict[str, list[int]] = {}
    total = passed = zt_total = zt_passed = 0
    scenario_rows = []

    for scenario in scenarios:
        # A fresh world per scenario: accounts are shared state, and
        # one persona's executed credit must not leak into the next
        # persona's bill.
        service = ConversationService(store=InMemoryStore())
        service.seed()
        # Conversations that happened before this one — a need
        # recorded on the account in an earlier contact, a set of
        # unhandled asks the mining pass will read. Their turns
        # carry no expectations; they set the world.
        for seed_conv in scenario.get("seed_conversations", []):
            seeded = service.start_conversation()
            for seed_turn in seed_conv["turns"]:
                service.post_message(seeded.conversation_id,
                                     _substitute(seed_turn["say"]))
        conv = service.start_conversation()
        s_passed = s_total = s_zt_passed = s_zt_total = 0
        for idx, turn in enumerate(scenario["turns"], start=1):
            if turn.get("poke"):
                apply_poke(service, turn["poke"])
            result = service.post_message(
                conv.conversation_id, _substitute(turn["say"]))
            p, t, zp, zt = check_turn(
                result, turn.get("expect") or {}, failures,
                f"{scenario['id']} turn {idx}", scenario.get("zt", False))
            s_passed += p; s_total += t; s_zt_passed += zp; s_zt_total += zt
        p, t, zp, zt = check_final(
            service, scenario, conv.conversation_id, failures,
            scenario["id"], scenario.get("zt", False))
        s_passed += p; s_total += t; s_zt_passed += zp; s_zt_total += zt
        passed += s_passed; total += s_total
        zt_passed += s_zt_passed; zt_total += s_zt_total
        bucket = per_rubric.setdefault(scenario["rubric"], [0, 0])
        bucket[0] += s_passed; bucket[1] += s_total
        scenario_rows.append({
            "id": scenario["id"], "rubric": scenario["rubric"],
            "persona": scenario["persona"],
            "passed": s_passed, "total": s_total,
            "zt_passed": s_zt_passed, "zt_total": s_zt_total,
        })

    overall = passed / total if total else 0.0
    zt_ok = zt_passed == zt_total
    gate = zt_ok and overall >= 0.95

    print(f"Scenarios: {len(scenarios)}  Assertions: {total}")
    for rubric, (p, t) in sorted(per_rubric.items()):
        print(f"  {rubric:<12} {p}/{t}  ({p / t:.0%})")
    print(f"Zero-tolerance: {zt_passed}/{zt_total}  ({'OK' if zt_ok else 'FAILED'})")
    print(f"Overall: {passed}/{total}  ({overall:.1%})  — gate >= 95%")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
    print(f"\n{'PASS' if gate else 'FAIL'}")

    (ROOT / "evals" / "results.json").write_text(json.dumps({
        "scenarios": scenario_rows, "failures": failures,
        "passed": passed, "total": total, "overall": overall,
        "zt_passed": zt_passed, "zt_total": zt_total, "gate_pass": gate,
    }, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
