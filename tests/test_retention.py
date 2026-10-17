"""Round 14: the retention flow.

A wish to leave becomes a case record — reason classified in
code against the governed list, the customer's words verbatim —
and the flow around it is deliberately un-pushy: at most one
save offer per case, built from a real mechanism already in
the book (a priced downgrade, a goodwill credit under its
policy, the loyalty upgrade price); no offer at all when the
customer says they want none; a declined offer or a second
cancel request goes straight to the governed cancellation,
which executes through the normal action gate and takes
effect at the cycle boundary. A fault-blamed exit follows the
existing billing dispute path, with the routing on the case.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from cx_agent import actions as action_mod
from cx_agent import retention_book as rb
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import (
    ActionKind,
    Conversation,
    IdentityState,
    ProposedAction,
    RetentionCase,
)


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-10-01")


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _drive(service, account_number, postcode, opener):
    """Open with the wish to leave, answer the identity flow,
    and let the code turn resume the request at the offers desk.
    Returns (conversation, result of the code turn)."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, opener)
    _say(service, conv.conversation_id,
         f"{account_number} postcode {postcode}")
    result = _say(service, conv.conversation_id,
                  expected_code(account_number))
    return conv, result


def _case(service, conv):
    cases = service.store.list_retention_cases(
        conversation_id=conv.conversation_id)
    assert len(cases) == 1
    return cases[0]


def _save_offer_actions(service, conv):
    return [a for a in service.store.list_actions(conv.conversation_id)
            if a.payload.get("retention_save_offer")]


def _save_offer_proposals(service, conv):
    """The save-offer PROPOSALS made — as opposed to the answer
    records (declined / executed) the confirmation protocol
    writes beside each proposal, in the book's usual pattern.
    A proposal record is the one that starts life "proposed"
    (it may later be superseded by a reroute)."""
    return [a for a in _save_offer_actions(service, conv)
            if a.status in ("proposed", "superseded")]


def _boundary(account) -> str:
    return (date.fromisoformat(account.cycle_end)
            + timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# Classification: reasons, intent, the no-offers instruction
# ---------------------------------------------------------------------------


def test_reasons_are_governed_and_deterministic():
    assert rb.classify_reason(
        "I want to cancel, it's too expensive") == "price"
    assert rb.classify_reason(
        "cancel please — the coverage at my house is awful") == "coverage"
    assert rb.classify_reason(
        "I want to leave, the service quality is terrible") \
        == "service_quality"
    assert rb.classify_reason(
        "please cancel, I'm moving house next month") == "moving_house"
    assert rb.classify_reason(
        "cancel my account, another provider offered me a deal") \
        == "competitor"
    assert rb.classify_reason(
        "I want to cancel — I no longer need a second line") \
        == "no_longer_needed"
    assert rb.classify_reason("I just want to cancel") == "other"
    # Every answer the classifier can give is on the governed list.
    for text in ("cancel", "leave because reasons", "cancel, pricey"):
        assert rb.classify_reason(text) in rb.RETENTION_REASONS


def test_cancel_intent_is_not_cancelling_something_else():
    assert rb.is_cancel_intent("I want to cancel my account")
    assert rb.is_cancel_intent("please cancel my service")
    assert rb.is_cancel_intent("just cancel it")
    # Other desks' cancellations never open a retention case.
    assert not rb.is_cancel_intent("turn off autopay, cancel autopay")
    assert not rb.is_cancel_intent("cancel the scheduled change")
    assert not rb.is_cancel_intent("cancel that change, put me back")


def test_no_offers_instruction_is_read_off_the_words():
    assert rb.no_offers_requested("just cancel it, no offers")
    assert rb.no_offers_requested("don't try to keep me, just cancel")
    assert not rb.no_offers_requested("I want to cancel my account")


# ---------------------------------------------------------------------------
# The case record and the save offer, accepted
# ---------------------------------------------------------------------------


def test_price_case_opens_and_the_downgrade_offer_saves_it(service):
    conv, result = _drive(
        service, "NB-1001", "02134",
        "I want to cancel my account, it's too expensive")
    assert result.intent == "offers"
    assert result.action is not None
    assert result.action.kind == ActionKind.CHANGE_PLAN
    assert result.action.status == "proposed"
    case = _case(service, conv)
    assert case.account_number == "NB-1001"
    assert case.reason == "price"
    assert case.reason_text == \
        "I want to cancel my account, it's too expensive"
    assert case.state == "offer_proposed"
    assert case.save_offer_mechanism == "plan_downgrade"
    assert case.save_offer_action_id == result.action.action_id
    # The offer is the plan-change machinery's downgrade: the
    # cheapest cheaper plan, scheduled at the cycle boundary,
    # no money moved today.
    account = service.store.get_account("NB-1001")
    assert result.action.payload["new_plan_id"] == "PLAN-BAS"
    assert result.action.payload["effective"] == "next_cycle"
    assert result.action.payload["effective_date"] == _boundary(account)

    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1001")
    assert account.plan_id == "PLAN-ESS"  # the live plan has not moved
    assert account.pending_plan_id == "PLAN-BAS"
    assert account.pending_plan_effective == _boundary(account)
    assert account.balance_due == Decimal("94.35")  # nothing charged
    assert account.pending_cancellation_effective == ""
    case = _case(service, conv)
    assert case.state == "saved"
    assert case.outcome == "saved_accepted_offer"
    assert case.save_offer_outcome == "accepted"


def test_loyalty_price_is_the_competitor_save(service):
    conv, result = _drive(
        service, "NB-1001", "02134",
        "I want to cancel — another provider offered me a deal")
    case = _case(service, conv)
    assert case.reason == "competitor"
    assert case.save_offer_mechanism == "loyalty_upgrade"
    # The loyalty mechanism exactly: Unlimited Max at $60, not
    # $75, prorated off the cycle calendar — the figures the
    # plan-change machinery produces for that price.
    assert result.action.payload["new_price"] == "60.00"
    assert result.action.payload["net"] == "5.50"
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1001")
    assert account.plan_id == "PLAN-UNL"
    assert account.balance_due == Decimal("99.85")  # 94.35 + 5.50 net
    assert _case(service, conv).outcome == "saved_accepted_offer"


def test_goodwill_credit_is_the_service_quality_save(service):
    conv, result = _drive(
        service, "NB-1009", "98101",
        "I want to cancel my account — the service quality "
        "has been terrible")
    case = _case(service, conv)
    assert case.reason == "service_quality"
    assert case.save_offer_mechanism == "goodwill_credit"
    assert result.action.kind == ActionKind.APPLY_CREDIT
    assert result.action.payload["credit_class"] == "goodwill"
    # The goodwill policy's own cap, not an invented discount.
    assert result.action.amount == Policy().goodwill_credit_cap
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1009")
    assert account.balance_due == Decimal("50.00")  # 75.00 - 25.00
    assert _case(service, conv).outcome == "saved_accepted_offer"


def test_no_case_opens_before_the_account_is_known(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "I want to cancel my account, it's too expensive")
    assert service.store.list_retention_cases() == []


# ---------------------------------------------------------------------------
# No dark patterns
# ---------------------------------------------------------------------------


def test_declining_the_offer_goes_straight_to_cancellation(service):
    conv, _ = _drive(
        service, "NB-1003", "10001",
        "Please cancel my account — too expensive")
    declined = _say(service, conv.conversation_id, "no, I don't want that")
    # The same reply proposes the cancellation — and nothing
    # else: the declined offer is not repeated, rephrased or
    # replaced by a second one.
    assert declined.action is not None
    assert declined.action.kind == ActionKind.CANCEL_ACCOUNT
    assert declined.action.status == "proposed"
    assert "Basic Talk" not in declined.reply
    assert len(_save_offer_proposals(service, conv)) == 1
    assert any(a.status == "declined"
               for a in _save_offer_actions(service, conv))
    case = _case(service, conv)
    assert case.save_offer_outcome == "declined"
    assert case.outcome == ""  # not decided until the cancellation lands

    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1003")
    assert account.pending_cancellation_effective == _boundary(account)
    assert account.plan_id == "PLAN-FAM"  # active until the boundary
    case = _case(service, conv)
    assert case.state == "cancelled"
    assert case.outcome == "cancelled"
    assert case.cancellation_effective == _boundary(account)
    assert case.cancellation_action_id == done.action.action_id
    # Still exactly one save offer anywhere on the books.
    assert len(_save_offer_proposals(service, conv)) == 1


def test_no_offers_means_no_offer_at_all(service):
    conv, result = _drive(
        service, "NB-1005", "94107",
        "Just cancel it, no offers — I want to cancel my account")
    assert result.action is not None
    assert result.action.kind == ActionKind.CANCEL_ACCOUNT
    assert _save_offer_actions(service, conv) == []
    case = _case(service, conv)
    assert case.no_offers_requested is True
    assert case.save_offer_summary == ""
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    assert _case(service, conv).outcome == "cancelled"
    assert _save_offer_actions(service, conv) == []


def test_second_insistence_at_the_offer_completes_it(service):
    conv, _ = _drive(
        service, "NB-1007", "02139",
        "I want to cancel my account, it's too expensive")
    insisted = _say(service, conv.conversation_id,
                    "No — just cancel my account")
    # No second offer, no re-ask: the cancellation is proposed.
    assert insisted.action.kind == ActionKind.CANCEL_ACCOUNT
    assert len(_save_offer_proposals(service, conv)) == 1
    # And saying cancel to the cancellation proposal confirms
    # it — it does not decline the customer's own request.
    done = _say(service, conv.conversation_id, "cancel it now")
    assert done.action.status == "executed"
    assert done.action.kind == ActionKind.CANCEL_ACCOUNT
    assert _case(service, conv).outcome == "cancelled"


def test_second_request_reaching_the_desk_gets_no_new_offer(service):
    conv, _ = _drive(
        service, "NB-1011", "55401",
        "I want to cancel my account, it's too expensive")
    assert len(_save_offer_proposals(service, conv)) == 1
    # A billing detour retires the unanswered offer (the round-2
    # rule), and the customer comes back still wanting to leave.
    _say(service, conv.conversation_id, "actually, why is my bill so high?")
    again = _say(service, conv.conversation_id,
                 "I still want to cancel my account")
    assert again.action is not None
    assert again.action.kind == ActionKind.CANCEL_ACCOUNT
    assert len(_save_offer_proposals(service, conv)) == 1
    case = _case(service, conv)
    assert case.cancel_requests == 2


def test_moving_house_gets_no_invented_offer(service):
    conv, result = _drive(
        service, "NB-1002", "90210",
        "Please cancel my account — I'm moving house next month")
    case = _case(service, conv)
    assert case.reason == "moving_house"
    assert case.save_offer_summary == ""
    assert result.action.kind == ActionKind.CANCEL_ACCOUNT
    assert _save_offer_actions(service, conv) == []


# ---------------------------------------------------------------------------
# The fault path, and the execution-time rules
# ---------------------------------------------------------------------------


def test_fault_blamed_exit_follows_the_billing_path(service):
    conv, result = _drive(
        service, "NB-1009", "98101",
        "I want to cancel because of the fault — the service "
        "quality is terrible and it keeps breaking")
    assert result.escalated
    assert result.escalation_reason == "fault_cancellation_dispute_path"
    assert service.list_escalations()[-1].queue == "billing"
    case = _case(service, conv)
    assert case.reason == "service_quality"
    assert case.routed == "billing_dispute_path"
    assert case.state == "escalated_retention_team"
    assert case.outcome == "escalated_retention_team"
    assert case.save_offer_summary == ""
    account = service.store.get_account("NB-1009")
    assert account.plan_id == "PLAN-UNL"
    assert account.pending_cancellation_effective == ""


def test_single_counter_rule_is_rechecked_at_execution(seeded_store):
    account = seeded_store.get_account("NB-1001")
    case = RetentionCase(
        case_id="RET-T1", conversation_id="CONV-T",
        account_number="NB-1001", reason="service_quality",
        reason_text="cancel, the service quality is terrible",
        created_date="2026-10-01", updated_date="2026-10-01")
    seeded_store.save_retention_case(case)
    conv = Conversation(conversation_id="CONV-T",
                        account_number="NB-1001",
                        identity_state=IdentityState.VERIFIED)
    proposal = ProposedAction(
        kind=ActionKind.APPLY_CREDIT,
        summary="apply a goodwill credit of $25.00",
        payload={"credit_class": "goodwill",
                 "retention_case_id": "RET-T1",
                 "retention_save_offer": True},
        amount=Decimal("25.00"), basis="test")
    first = action_mod.execute(seeded_store, conv, proposal, Policy(),
                               confirmed=True)
    assert first.status == "executed"
    # The same offer proposed again — through any path — may
    # not run a second time.
    second = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                confirmed=True)
    assert second.status == "blocked"
    assert second.block_reason == "retention_single_counter_used"
    assert seeded_store.get_account("NB-1001").balance_due \
        == Decimal("69.35")  # credited exactly once


def test_cancel_account_execution_schedules_at_the_boundary(seeded_store):
    account = seeded_store.get_account("NB-1001")
    account.pending_plan_id = "PLAN-BAS"
    account.pending_plan_effective = "2026-10-13"
    seeded_store.save_account(account)
    conv = Conversation(conversation_id="CONV-T",
                        account_number="NB-1001",
                        identity_state=IdentityState.VERIFIED)
    proposal = ProposedAction(
        kind=ActionKind.CANCEL_ACCOUNT, summary="cancel the account",
        payload={"retention_case_id": "RET-T1",
                 "effective": "next_cycle",
                 "effective_date": _boundary(account)},
        amount=Decimal("0"), basis="test")
    record = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                confirmed=True)
    assert record.status == "executed"
    after = seeded_store.get_account("NB-1001")
    assert after.pending_cancellation_effective == _boundary(account)
    assert after.plan_id == "PLAN-ESS"  # still active today
    # A scheduled plan change is superseded by leaving.
    assert after.pending_plan_id is None
    assert after.pending_plan_effective == ""
    again = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                 confirmed=True)
    assert again.status == "blocked"
    assert again.block_reason == "cancellation_already_scheduled"


def test_cancellation_needs_verification_and_confirmation(seeded_store):
    conv = Conversation(conversation_id="CONV-T",
                        account_number="NB-1001")
    proposal = ProposedAction(
        kind=ActionKind.CANCEL_ACCOUNT, summary="cancel the account",
        payload={"effective_date": "2026-10-13"},
        amount=Decimal("0"), basis="test")
    blocked = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                 confirmed=True)
    assert blocked.status == "blocked"
    assert blocked.block_reason == "identity_not_verified"
    conv.identity_state = IdentityState.VERIFIED
    blocked = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                 confirmed=False)
    assert blocked.block_reason == "confirmation_missing"


# ---------------------------------------------------------------------------
# The report — counts from records only
# ---------------------------------------------------------------------------


def test_report_counts_the_records(service):
    _drive(service, "NB-1001", "02134",
           "I want to cancel my account, it's too expensive")
    conv1 = service.list_conversations()[-1]
    _say(service, conv1.conversation_id, "yes")  # saved
    conv2, _ = _drive(service, "NB-1003", "10001",
                      "Please cancel my account — too expensive")
    _say(service, conv2.conversation_id, "no")  # offer declined
    _say(service, conv2.conversation_id, "yes")  # cancelled
    conv3, _ = _drive(
        service, "NB-1009", "98101",
        "I want to cancel because of the fault — the service "
        "quality is terrible and it keeps breaking")

    report = rb.retention_report(service.store)
    assert report["cases"] == 3
    assert report["by_reason"]["price"] == 2
    assert report["by_reason"]["service_quality"] == 1
    assert report["offers_made"] == 2
    assert report["offers_accepted"] == 1
    assert report["offers_declined"] == 1
    assert report["decided_cases"] == 3
    assert report["save_rate"] == pytest.approx(1 / 3)
    assert report["cancellations_completed"] == 1
    assert report["escalated_retention_team"] == 1
    # The service and the module read the same records.
    assert service.retention_report() == report
    assert len(service.list_retention_cases()) == 3
