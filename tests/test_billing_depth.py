"""Round 3 — billing depth.

The billing book's arithmetic (decomposition, comparison, shock),
the per-line dispute decisions and the cases they persist, the
rolling goodwill history, and the flows end to end: a partial
dispute credits exactly the evidenced line, an over-cap dispute
escalates with its case intact, and a customer who has used up the
rolling goodwill allowance is refused — on record — while the
evidenced path stays open.
"""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from decimal import Decimal

from cx_agent import actions
from cx_agent.billing_book import (
    assess_dispute,
    bill_shock,
    compare_bills,
    credit_history,
    decompose_bill,
    goodwill_offer,
    trailing_average,
)
from cx_agent.clock import today
from cx_agent.config import Policy
from cx_agent.dataset import build_bills
from cx_agent.identity import expected_code
from cx_agent.models import (
    ActionKind,
    ActionRecord,
    Bill,
    BillCharge,
    Conversation,
    IdentityState,
    ProposedAction,
)

POLICY = Policy()
D = Decimal


def _bills(account: str) -> list[Bill]:
    rows = [b for b in build_bills() if b.account_number == account]
    return sorted(rows, key=lambda b: b.issued)


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _verified_conversation(service, account: str, postcode: str) -> str:
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "why is my bill so high")
    _say(service, conv.conversation_id, f"{account} postcode {postcode}")
    _say(service, conv.conversation_id, expected_code(account))
    return conv.conversation_id


# ---------------------------------------------------------------------------
# Decomposition, comparison, bill shock
# ---------------------------------------------------------------------------


def test_decomposition_groups_by_family_with_code_totals():
    current = _bills("NB-1001")[-1]
    groups = {g["family"]: g for g in decompose_bill(current)}
    assert groups["plan"]["total"] == D("45.00")
    assert groups["device"]["total"] == D("22.50")
    assert groups["one_off"]["total"] == D("19.98")  # the duplicate pair
    assert groups["taxes_fees"]["total"] == D("6.87")
    assert sum(g["total"] for g in groups.values()) == current.total


def test_comparison_deltas_against_the_previous_bill():
    previous, current = _bills("NB-1002")
    rows = {r["family"]: r for r in compare_bills(current, previous)}
    assert rows["roaming"]["delta"] == D("68.40")
    assert rows["roaming"]["previous"] == D("0")
    assert rows["plan"]["delta"] == D("0")
    assert trailing_average([previous]) == D("84.12")


def test_bill_shock_names_the_computed_driver():
    previous, current = _bills("NB-1002")
    shock = bill_shock(current, [previous], POLICY)
    assert shock is not None
    assert shock["average"] == D("84.12")
    assert shock["increase"] == D("68.40")
    assert shock["driver"]["family"] == "roaming"
    assert shock["driver"]["is_new"] is True


def test_bill_shock_stays_quiet_below_the_threshold():
    # Amara is +11.9% and Marcus +24.4% over their previous bills —
    # under the 25% threshold, so neither explanation leads with a
    # shock. No previous bills at all also means no shock claim.
    for account in ("NB-1001", "NB-1006"):
        previous, current = _bills(account)
        assert bill_shock(current, [previous], POLICY) is None
    assert bill_shock(_bills("NB-1002")[-1], [], POLICY) is None


def test_bill_shock_driver_can_be_a_one_off():
    previous = Bill(bill_id="B-P", account_number="NB-T",
                    cycle_start="2026-08-13", cycle_end="2026-09-11",
                    issued="2026-09-12", due="2026-10-03",
                    charges=[BillCharge(charge_id="P1", kind="plan",
                                        description="Plan", amount=D("40.00"))])
    current = Bill(bill_id="B-C", account_number="NB-T",
                   cycle_start="2026-09-12", cycle_end="2026-10-11",
                   issued="2026-10-12", due="2026-11-02",
                   charges=[
                       BillCharge(charge_id="C1", kind="plan",
                                  description="Plan", amount=D("40.00")),
                       BillCharge(charge_id="C2", kind="one_off",
                                  description="Engineer visit", amount=D("60.00")),
                   ])
    shock = bill_shock(current, [previous], POLICY)
    assert shock["driver"]["family"] == "one_off"
    assert shock["driver"]["top_line"].description == "Engineer visit"


# ---------------------------------------------------------------------------
# Per-line dispute assessment
# ---------------------------------------------------------------------------


def test_partial_assessment_decides_each_line_on_its_own_evidence():
    bill = _bills("NB-1001")[-1]
    assessment = assess_dispute(
        bill,
        "I dispute two charges: you charged me twice for device "
        "protection, and the handset instalment — I don't recognise it",
        POLICY)
    by_id = {l.charge_id: l for l in assessment.lines}
    # The duplicate is credited for exactly its amount…
    assert by_id["CH-1001-3"].decision == "credited"
    assert by_id["CH-1001-3"].disputed_amount == D("9.99")
    # …the named handset line is declined with a checkable reason…
    assert by_id["CH-1001-4"].decision == "declined"
    assert "instalment 14 of 24" in by_id["CH-1001-4"].reason
    # …and the original of the duplicate pair is not a dispute line
    # of its own — the pair is one wrong charge.
    assert "CH-1001-2" not in by_id
    assert assessment.credit_total == D("9.99")
    assert assessment.escalated is False


def test_wrong_rate_line_disputes_only_the_excess():
    bill = Bill(bill_id="B-T", account_number="NB-T",
                cycle_start="2026-09-12", cycle_end="2026-10-11",
                issued="2026-10-12", due="2026-11-02",
                charges=[BillCharge(charge_id="T1", kind="usage",
                                    description="Data overage",
                                    amount=D("30.00"),
                                    expected_amount=D("22.00"))])
    assessment = assess_dispute(bill, "the data overage is wrong", POLICY)
    assert assessment.lines[0].decision == "credited"
    assert assessment.lines[0].disputed_amount == D("8.00")
    assert assessment.credit_total == D("8.00")


def test_over_cap_assessment_escalates_every_evidenced_line():
    bill = _bills("NB-1012")[-1]
    assessment = assess_dispute(bill, "the roaming charge is wrong", POLICY)
    assert assessment.evidenced_total == D("58.00")
    assert assessment.escalated is True
    assert assessment.credit_total == D("0")
    assert all(l.decision == "escalated" for l in assessment.lines)


# ---------------------------------------------------------------------------
# Credit history and the goodwill tiers
# ---------------------------------------------------------------------------


def _credit_record(account: str, amount: str, days_ago: int,
                   credit_class: str = "goodwill",
                   status: str = "executed",
                   conversation_id: str = "CONV-HIST") -> ActionRecord:
    return ActionRecord(
        action_id=f"ACT-T-{account}-{days_ago}-{status}-{credit_class}",
        conversation_id=conversation_id, kind=ActionKind.APPLY_CREDIT,
        summary="credit", payload={"credit_class": credit_class},
        amount=D(amount), status=status, account_number=account,
        created_date=(today() - timedelta(days=days_ago)).isoformat())


def test_credit_history_reads_the_trailing_window_only(seeded_store):
    seeded_store.save_action(_credit_record("NB-1003", "25.00", 10))
    seeded_store.save_action(_credit_record("NB-1003", "25.00", 120))  # too old
    seeded_store.save_action(_credit_record("NB-1003", "25.00", 5,
                                            credit_class="dispute"))  # wrong tier
    seeded_store.save_action(_credit_record("NB-1004", "25.00", 5))  # other account
    history = credit_history(seeded_store, "NB-1003", POLICY)
    assert history.executed_total == D("25.00")


def test_goodwill_offer_is_the_headroom_under_the_rolling_cap(seeded_store):
    # Ruth's seeded history uses the full $50 rolling allowance.
    conv = Conversation(conversation_id="CONV-T", account_number="NB-1011",
                        customer_id="CUST-011")
    offer, history = goodwill_offer(seeded_store, conv, POLICY)
    assert history.executed_total == D("50.00")
    assert offer <= 0
    # June has no history: the full per-offer cap is available.
    conv = Conversation(conversation_id="CONV-T2", account_number="NB-1009",
                        customer_id="CUST-009")
    offer, _history = goodwill_offer(seeded_store, conv, POLICY)
    assert offer == D("25.00")


def test_execution_rereads_the_rolling_cap(seeded_store):
    conv = Conversation(conversation_id="CONV-T", account_number="NB-1011",
                        customer_id="CUST-011",
                        identity_state=IdentityState.VERIFIED)
    seeded_store.save_conversation(conv)
    record = actions.execute(
        seeded_store, conv,
        ProposedAction(kind=ActionKind.APPLY_CREDIT, summary="goodwill",
                       payload={"credit_class": "goodwill"},
                       amount=D("25.00")),
        POLICY, confirmed=True)
    assert record.status == "blocked"
    assert record.block_reason == "goodwill_rolling_cap_reached"
    assert seeded_store.get_account("NB-1011").balance_due == D("72.61")


# ---------------------------------------------------------------------------
# Flows: cases through real conversations
# ---------------------------------------------------------------------------


def test_partial_dispute_credits_exactly_the_evidenced_line(service):
    conv_id = _verified_conversation(service, "NB-1001", "02134")
    result = _say(service, conv_id,
                  "I dispute two charges: you charged me twice for device "
                  "protection, and the handset instalment — I don't "
                  "recognise it")
    assert result.action is not None and result.action.status == "proposed"
    assert result.action.amount == D("9.99")
    result = _say(service, conv_id, "yes")
    assert result.action.status == "executed"
    assert service.store.get_account("NB-1001").balance_due == D("84.36")

    cases = service.store.list_dispute_cases(conversation_id=conv_id)
    assert len(cases) == 1
    case = cases[0]
    assert case.state == "decided_partial"
    assert case.credit_applied == D("9.99")
    assert {l.charge_id: l.decision for l in case.lines} == {
        "CH-1001-3": "credited", "CH-1001-4": "declined"}
    # The audit trail links the conversation to the case both ways.
    kinds = [(e.kind, e.detail.get("event")) for e in
             service.store.list_audit(conv_id)]
    assert ("billing", "dispute_opened") in kinds
    assert ("billing", "dispute_decided") in kinds


def test_declined_credit_leaves_the_case_opened(service):
    conv_id = _verified_conversation(service, "NB-1001", "02134")
    _say(service, conv_id, "I was charged twice for device protection")
    result = _say(service, conv_id, "no, leave it")
    assert result.action.status == "declined"
    case = service.store.list_dispute_cases(conversation_id=conv_id)[0]
    assert case.state == "opened"
    assert case.credit_applied == D("0")
    assert "declined the offered credit" in case.decision_basis


def test_repeat_goodwill_is_refused_on_record(service):
    conv_id = _verified_conversation(service, "NB-1011", "55401")
    result = _say(service, conv_id,
                  "I dispute the handset instalment — that charge is wrong, "
                  "I never agreed to it, and honestly this is ridiculous, "
                  "I've been loyal for years")
    assert result.action is None
    assert "already credited $50.00" in result.reply
    assert service.store.get_account("NB-1011").balance_due == D("72.61")
    case = service.store.list_dispute_cases(conversation_id=conv_id)[0]
    assert case.state == "declined_with_reason"
    assert case.goodwill_refused_reason.startswith("rolling goodwill cap")
    assert [l.decision for l in case.lines] == ["declined"]


def test_evidenced_path_is_untouched_by_goodwill_history(service):
    # Same store, same day: Ruth's exhausted goodwill history does
    # not stop Amara's evidenced dispute crediting in full.
    _verified_conversation(service, "NB-1011", "55401")
    conv_id = _verified_conversation(service, "NB-1001", "02134")
    result = _say(service, conv_id,
                  "I was charged twice for device protection, that's wrong")
    assert result.action is not None and result.action.amount == D("9.99")


def test_over_cap_dispute_escalates_with_the_case_intact(service):
    conv_id = _verified_conversation(service, "NB-1012", "80202")
    result = _say(service, conv_id,
                  "I dispute the roaming charge on my bill — this is the "
                  "wrong charge, I was quoted $20 for Greece, not $78")
    assert result.escalated is True
    assert result.escalation_reason == "dispute_above_agent_cap"
    case = service.store.list_dispute_cases(conversation_id=conv_id)[0]
    assert case.state == "escalated_above_cap"
    assert case.lines[0].decision == "escalated"
    assert case.lines[0].disputed_amount == D("58.00")
    assert service.store.get_account("NB-1012").balance_due == D("130.38")


def test_bill_shock_explanation_leads_with_the_driver(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "my bill is much higher than usual")
    _say(service, conv.conversation_id, "NB-1002 postcode 90210")
    result = _say(service, conv.conversation_id, expected_code("NB-1002"))
    assert "recent average of $84.12" in result.reply
    assert "main driver is roaming" in result.reply
    assert "$68.40" in result.reply


def test_explanation_groups_families_and_compares(service):
    conv_id = _verified_conversation(service, "NB-1001", "02134")
    conv = service.store.get_conversation(conv_id)
    last_reply = conv.messages[-1].text
    assert "Plan total: $45.00" in last_reply
    assert "One-offs total: $19.98" in last_reply
    assert "One-offs: $19.98 (was $9.99)" in last_reply


def test_goodwill_offer_shrinks_to_the_remaining_headroom(service):
    service.store.save_action(_credit_record(
        "NB-1009", "40.00", 3, conversation_id="CONV-EARLIER"))
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "my bill feels far too expensive and I've been a loyal customer "
         "for years")
    _say(service, conv.conversation_id, "NB-1009 postcode 98101")
    result = _say(service, conv.conversation_id, expected_code("NB-1009"))
    assert result.action is not None
    assert result.action.amount == D("10.00")


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

os.environ["SQLITE_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="cx-api-r3-"), "cx.db")
os.environ.pop("DATABASE_URL", None)

from fastapi.testclient import TestClient  # noqa: E402

from cx_agent.api import app  # noqa: E402


def test_disputes_api_and_audit_endpoint():
    client = TestClient(app)
    client.post("/seed")
    conv_id = client.post("/conversations").json()["conversation_id"]
    for text in ("why is my bill so high", "NB-1001 postcode 02134",
                 expected_code("NB-1001"),
                 "I dispute two charges: you charged me twice for device "
                 "protection, and the handset instalment — I don't "
                 "recognise it", "yes"):
        client.post(f"/conversations/{conv_id}/messages", json={"text": text})
    disputes = client.get("/disputes",
                          params={"conversation_id": conv_id}).json()["disputes"]
    assert len(disputes) == 1
    case = disputes[0]
    assert case["state"] == "decided_partial"
    assert case["credit_applied"] == "9.99"
    one = client.get(f"/disputes/{case['case_id']}")
    assert one.status_code == 200
    assert one.json()["account_number"] == "NB-1001"
    assert client.get("/disputes/DSP-NOPE").status_code == 404
    audit = client.get(f"/conversations/{conv_id}/audit").json()["audit"]
    billing_events = [e for e in audit if e["kind"] == "billing"]
    assert {e["detail"]["event"] for e in billing_events} == {
        "dispute_opened", "dispute_decided"}
    assert all(e["detail"]["case_id"] == case["case_id"]
               for e in billing_events)
