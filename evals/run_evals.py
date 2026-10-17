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
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.service import ConversationService  # noqa: E402
from cx_agent.store import InMemoryStore  # noqa: E402

# Assertion keys whose failure is never tolerable, wherever they
# appear: they encode "the agent escalated (or didn't) when it
# must", "money moved (or didn't) exactly", "identity was (or
# wasn't) established".
ZT_KEYS = {"escalated", "action_status", "balance_due", "plan_id",
           "identity_state", "payment_method_last4", "executed_kinds"}

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
        elif key == "reply_not_contains":
            ok = want not in result.reply
            got = f"reply contains {want!r}"
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
    values = {
        "status": conv.status if conv else None,
        "unhandled_logged": bool(conv and conv.unhandled_intent),
        "escalation_queue": escalations[-1].queue if escalations else None,
        "reroute_count": sum(1 for e in audit if e.kind == "reroute"),
        "action_statuses": [a.status for a in store.list_actions(conv_id)],
        "executed_kinds": executed,
        "actions_count": len(store.list_actions(conv_id)),
        "balance_due": str(account.balance_due) if account else None,
        "plan_id": account.plan_id if account else None,
        "payment_method_last4": account.payment_method_last4 if account else None,
    }
    for key, want in final.items():
        total += 1
        is_zt = scenario_zt or key in ZT_KEYS
        if is_zt:
            zt_total += 1
        ok = values.get(key) == want
        if ok:
            passed += 1
            if is_zt:
                zt_passed += 1
        else:
            failures.append(f"{label}: final expected {key}={want!r}, "
                            f"got {values.get(key)!r}")
    return passed, total, zt_passed, zt_total


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
        conv = service.start_conversation()
        s_passed = s_total = s_zt_passed = s_zt_total = 0
        for idx, turn in enumerate(scenario["turns"], start=1):
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
