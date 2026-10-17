from decimal import Decimal

from cx_agent import actions
from cx_agent.config import Policy
from cx_agent.models import (
    ActionKind,
    Conversation,
    IdentityState,
    ProposedAction,
)

POLICY = Policy()


def _conv(store, verified=True, account="NB-1001"):
    conv = Conversation(conversation_id="CONV-T", account_number=account,
                        customer_id="CUST-001")
    if verified:
        conv.identity_state = IdentityState.VERIFIED
    store.save_conversation(conv)
    return conv


def _credit(amount, credit_class="dispute"):
    return ProposedAction(kind=ActionKind.APPLY_CREDIT, summary=f"credit {amount}",
                          payload={"credit_class": credit_class},
                          amount=Decimal(amount))


def test_consequential_action_blocked_unverified(seeded_store):
    conv = _conv(seeded_store, verified=False)
    record = actions.execute(seeded_store, conv, _credit("9.99"), POLICY,
                             confirmed=True)
    assert record.status == "blocked"
    assert record.block_reason == "identity_not_verified"
    assert seeded_store.get_account("NB-1001").balance_due == Decimal("94.35")


def test_consequential_action_blocked_without_confirmation(seeded_store):
    conv = _conv(seeded_store)
    record = actions.execute(seeded_store, conv, _credit("9.99"), POLICY,
                             confirmed=False)
    assert record.status == "blocked"
    assert record.block_reason == "confirmation_missing"


def test_dispute_credit_executes_and_reduces_balance(seeded_store):
    conv = _conv(seeded_store)
    record = actions.execute(seeded_store, conv, _credit("9.99"), POLICY,
                             confirmed=True)
    assert record.status == "executed"
    assert seeded_store.get_account("NB-1001").balance_due == Decimal("84.36")


def test_credit_caps_are_per_class(seeded_store):
    conv = _conv(seeded_store)
    over_dispute = actions.execute(seeded_store, conv, _credit("50.01"), POLICY,
                                   confirmed=True)
    assert over_dispute.block_reason == "credit_above_cap"
    over_goodwill = actions.execute(
        seeded_store, conv, _credit("25.01", "goodwill"), POLICY, confirmed=True)
    assert over_goodwill.block_reason == "credit_above_cap"
    at_cap = actions.execute(
        seeded_store, conv, _credit("25.00", "goodwill"), POLICY, confirmed=True)
    assert at_cap.status == "executed"


def test_plan_change_applies_proration_net(seeded_store):
    conv = _conv(seeded_store, account="NB-1003")
    conv.customer_id = "CUST-003"
    proposal = ProposedAction(
        kind=ActionKind.CHANGE_PLAN, summary="change plan",
        payload={"new_plan_id": "PLAN-UNL", "net": "-22.17"})
    record = actions.execute(seeded_store, conv, proposal, POLICY, confirmed=True)
    assert record.status == "executed"
    account = seeded_store.get_account("NB-1003")
    assert account.plan_id == "PLAN-UNL"
    assert account.balance_due == Decimal("96.43")


def test_card_update_stores_last4_only(seeded_store):
    conv = _conv(seeded_store, account="NB-1005")
    proposal = ProposedAction(kind=ActionKind.UPDATE_PAYMENT_METHOD,
                              summary="new card", payload={"last4": "7788"})
    record = actions.execute(seeded_store, conv, proposal, POLICY, confirmed=True)
    assert record.status == "executed"
    assert seeded_store.get_account("NB-1005").payment_method_last4 == "7788"
    bad = ProposedAction(kind=ActionKind.UPDATE_PAYMENT_METHOD,
                         summary="bad", payload={"last4": "12345678"})
    assert actions.execute(seeded_store, conv, bad, POLICY,
                           confirmed=True).block_reason == "bad_card_last4"


def test_arrangement_requires_minimum_balance(seeded_store):
    conv = _conv(seeded_store, account="NB-1005")  # balance 76.77 < 100
    proposal = ProposedAction(kind=ActionKind.PAYMENT_ARRANGEMENT,
                              summary="split", payload={"instalments": 3,
                                                        "each": "25.59"})
    record = actions.execute(seeded_store, conv, proposal, POLICY, confirmed=True)
    assert record.block_reason == "balance_below_arrangement_minimum"


def test_callback_is_not_consequential_but_needs_an_account(seeded_store):
    conv = Conversation(conversation_id="CONV-T2")  # no account at all
    proposal = ProposedAction(kind=ActionKind.BOOK_CALLBACK, summary="callback",
                              payload={"reason": "PROC-NOSIGNAL"})
    record = actions.execute(seeded_store, conv, proposal, POLICY, confirmed=False)
    assert record.block_reason == "no_account"
    conv2 = _conv(seeded_store)
    record2 = actions.execute(seeded_store, conv2, proposal, POLICY, confirmed=False)
    assert record2.status == "executed"


def test_every_execution_is_audited(seeded_store):
    conv = _conv(seeded_store)
    actions.execute(seeded_store, conv, _credit("9.99"), POLICY, confirmed=True)
    events = seeded_store.list_audit(conv.conversation_id)
    assert any(e.kind == "action" and e.detail["status"] == "executed"
               for e in events)
