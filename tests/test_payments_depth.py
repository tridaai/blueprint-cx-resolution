"""Round 7: payments depth.

A payment is an attempt on the record: approved, or declined with
the processor's code and the class the code falls into. SOFT
declines may be retried once the cause is fixed; HARD declines
may not — the desk refuses, and execution refuses again, so a
dead card is never run twice. Arrangements are records with a
computed schedule and a state the calendar derives; a broken one
goes to a human with its record. Refunds go to the method on
file, from the credit balance on the record, and nowhere else.
Autopay failures raise a notice the next contact opens with.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from cx_agent import actions, payments_book
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import (
    ActionKind,
    ArrangementInstallment,
    Conversation,
    IdentityState,
    PaymentArrangement,
    ProposedAction,
)

D = Decimal
POLICY = Policy()


@pytest.fixture()
def policy():
    return Policy()


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _drive(service, account_number, postcode, opener):
    """Open with a request that needs the account, answer the
    identity flow, and let the code turn resume the request.
    Returns (conversation, result of the code turn)."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, opener)
    _say(service, conv.conversation_id,
         f"{account_number} postcode {postcode}")
    result = _say(service, conv.conversation_id,
                  expected_code(account_number))
    return conv, result


def _conv(store, account, customer="CUST-001"):
    conv = Conversation(conversation_id="CONV-T", account_number=account,
                        customer_id=customer)
    conv.identity_state = IdentityState.VERIFIED
    store.save_conversation(conv)
    return conv


def _payment(amount):
    return ProposedAction(
        kind=ActionKind.TAKE_PAYMENT, summary=f"pay {amount}",
        payload={"amount": amount}, amount=D(amount))


# ---------------------------------------------------------------------------
# Decline classes and the processor double
# ---------------------------------------------------------------------------


def test_decline_classes(policy):
    assert payments_book.decline_class("insufficient_funds") == "soft"
    assert payments_book.decline_class("processor_timeout") == "soft"
    assert payments_book.decline_class("card_expired") == "hard"
    assert payments_book.decline_class("card_reported_lost") == "hard"
    assert payments_book.decline_class("account_closed") == "hard"
    # A code the table predates reads as soft — the class that
    # does not forbid a retry on evidence the book doesn't have.
    assert payments_book.decline_class("some_future_code") == "soft"


def test_every_attempt_lands_on_the_log(seeded_store):
    account = seeded_store.get_account("NB-1003")
    account.processor_code = "insufficient_funds"
    seeded_store.save_account(account)
    attempt = payments_book.process_payment(
        seeded_store, account, D("118.60"), source="manual")
    assert attempt.outcome == "declined"
    assert attempt.reason_code == "insufficient_funds"
    assert attempt.decline_class == "soft"
    assert attempt.method_last4 == "2210"
    account.processor_code = ""
    seeded_store.save_account(account)
    attempt = payments_book.process_payment(
        seeded_store, account, D("118.60"), source="manual")
    assert attempt.outcome == "approved"
    assert [t.outcome for t in payments_book.attempts_for(
        seeded_store, "NB-1003")] == ["declined", "approved"]


# ---------------------------------------------------------------------------
# Execution: the retry rules are enforced at the moment of the attempt
# ---------------------------------------------------------------------------


def test_hard_decline_retry_never_executes(seeded_store):
    conv = _conv(seeded_store, "NB-1009", "CUST-009")
    account = seeded_store.get_account("NB-1009")
    account.processor_code = "card_expired"
    seeded_store.save_account(account)

    first = actions.execute(seeded_store, conv, _payment("75.00"),
                            POLICY, confirmed=True)
    assert first.status == "blocked"
    assert first.block_reason == "payment_declined_card_expired"
    assert seeded_store.get_account("NB-1009").balance_due == D("75.00")
    assert len(payments_book.attempts_for(seeded_store, "NB-1009")) == 1

    # The retry is refused before the processor is ever reached:
    # no second attempt exists, and no money moved.
    retry = actions.execute(seeded_store, conv, _payment("75.00"),
                            POLICY, confirmed=True)
    assert retry.status == "blocked"
    assert retry.block_reason == "hard_decline_retry_refused"
    assert len(payments_book.attempts_for(seeded_store, "NB-1009")) == 1
    assert seeded_store.get_account("NB-1009").balance_due == D("75.00")

    # A new card clears the standing decline: the block was a fact
    # about the old method, not about the customer.
    account = seeded_store.get_account("NB-1009")
    account.payment_method_last4 = "4321"
    account.processor_code = ""
    seeded_store.save_account(account)
    assert payments_book.hard_decline_block(seeded_store, account) is None
    paid = actions.execute(seeded_store, conv, _payment("75.00"),
                           POLICY, confirmed=True)
    assert paid.status == "executed"
    assert seeded_store.get_account("NB-1009").balance_due == D("0")


def test_soft_decline_retries_after_the_fix(seeded_store):
    conv = _conv(seeded_store, "NB-1003", "CUST-003")
    account = seeded_store.get_account("NB-1003")
    account.processor_code = "insufficient_funds"
    seeded_store.save_account(account)
    declined = actions.execute(seeded_store, conv, _payment("118.60"),
                               POLICY, confirmed=True)
    assert declined.block_reason == "payment_declined_insufficient_funds"
    assert seeded_store.get_account("NB-1003").balance_due == D("118.60")

    account = seeded_store.get_account("NB-1003")
    account.processor_code = ""  # the customer topped up
    seeded_store.save_account(account)
    paid = actions.execute(seeded_store, conv, _payment("118.60"),
                           POLICY, confirmed=True)
    assert paid.status == "executed"
    assert seeded_store.get_account("NB-1003").balance_due == D("0")
    assert [t.outcome for t in payments_book.attempts_for(
        seeded_store, "NB-1003")] == ["declined", "approved"]


# ---------------------------------------------------------------------------
# Arrangement maths: exact splits, policy dates, derived state
# ---------------------------------------------------------------------------


def test_split_amounts_sum_to_the_cent(policy):
    assert payments_book.split_amounts(D("212.40"), policy) == \
        [D("70.80"), D("70.80"), D("70.80")]
    assert payments_book.split_amounts(D("100.00"), policy) == \
        [D("33.34"), D("33.33"), D("33.33")]
    assert payments_book.split_amounts(D("110.00"), policy) == \
        [D("36.67"), D("36.67"), D("36.66")]
    for total in ("100.00", "100.01", "187.50", "212.40", "999.99"):
        parts = payments_book.split_amounts(D(total), policy)
        assert sum(parts) == D(total)
        assert len(parts) <= policy.arrangement_max_installments
        assert all(p >= policy.arrangement_min_installment for p in parts)


def test_installment_count_respects_the_minimum(policy):
    # $60 cannot carry three instalments of $25: the count falls
    # until each slice clears the policy minimum.
    assert payments_book.split_amounts(D("60.00"), policy) == \
        [D("30.00"), D("30.00")]
    assert payments_book.split_amounts(D("45.00"), policy) == [D("45.00")]


def test_plan_installments_dates(policy):
    schedule = payments_book.plan_installments(
        D("212.40"), policy, start=date(2026, 10, 1))
    assert [i.due_date for i in schedule] == \
        ["2026-10-15", "2026-11-14", "2026-12-14"]
    assert [i.amount for i in schedule] == [D("70.80")] * 3


def _arrangement(due, paid=D("0"), amount=D("62.50")):
    return PaymentArrangement(
        arrangement_id="ARR-T", account_number="NB-T", total=amount,
        installments=[ArrangementInstallment(
            number=1, amount=amount, due_date=due, paid_amount=paid)],
        state="active", created_date="2026-09-05")


def test_arrangement_state_is_derived():
    on = date(2026, 10, 1)
    assert payments_book.arrangement_state(
        _arrangement("2026-09-19"), on) == "broken"
    # Due today is not yet missed.
    assert payments_book.arrangement_state(
        _arrangement("2026-10-01"), on) == "active"
    assert payments_book.arrangement_state(
        _arrangement("2026-10-19"), on) == "active"
    assert payments_book.arrangement_state(
        _arrangement("2026-09-19", paid=D("62.50")), on) == "completed"


def test_arrangement_execution_writes_the_record(seeded_store):
    conv = _conv(seeded_store, "NB-1010", "CUST-010")
    # The schedule is laid from the shared clock: a schedule
    # written "today" must not already read as broken.
    schedule = payments_book.plan_installments(D("212.40"), POLICY)
    proposal = ProposedAction(
        kind=ActionKind.PAYMENT_ARRANGEMENT, summary="split",
        payload={"installments": [
            {"number": i.number, "amount": str(i.amount),
             "due_date": i.due_date} for i in schedule]},
        amount=D("212.40"))
    record = actions.execute(seeded_store, conv, proposal, POLICY,
                             confirmed=True)
    assert record.status == "executed"
    arrangements = seeded_store.list_payment_arrangements("NB-1010")
    assert len(arrangements) == 1
    arr = arrangements[0]
    assert arr.total == D("212.40")
    assert arr.state == "active"
    assert [i.due_date for i in arr.installments] == \
        [i.due_date for i in schedule]
    assert record.payload["arrangement_id"] == arr.arrangement_id

    # A second arrangement while one is active is refused.
    again = actions.execute(seeded_store, conv, proposal, POLICY,
                            confirmed=True)
    assert again.status == "blocked"
    assert again.block_reason == "arrangement_already_active"
    assert len(seeded_store.list_payment_arrangements("NB-1010")) == 1


def test_payment_pours_into_the_arrangement(seeded_store):
    conv = _conv(seeded_store, "NB-1010", "CUST-010")
    schedule = payments_book.plan_installments(D("212.40"), POLICY)
    proposal = ProposedAction(
        kind=ActionKind.PAYMENT_ARRANGEMENT, summary="split",
        payload={"installments": [
            {"number": i.number, "amount": str(i.amount),
             "due_date": i.due_date} for i in schedule]},
        amount=D("212.40"))
    actions.execute(seeded_store, conv, proposal, POLICY, confirmed=True)

    paid = actions.execute(seeded_store, conv, _payment("70.80"),
                           POLICY, confirmed=True)
    assert paid.status == "executed"
    arr = seeded_store.list_payment_arrangements("NB-1010")[0]
    assert arr.installments[0].paid_amount == D("70.80")
    assert arr.installments[0].paid_date != ""
    assert arr.installments[1].paid_amount == D("0")
    assert arr.state == "active"
    assert seeded_store.get_account("NB-1010").balance_due == D("141.60")


# ---------------------------------------------------------------------------
# Refunds: to the method on file, from the record, nowhere else
# ---------------------------------------------------------------------------


def _refund(amount, last4):
    return ProposedAction(
        kind=ActionKind.REFUND_PAYMENT, summary=f"refund {amount}",
        payload={"amount": amount, "method_last4": last4},
        amount=D(amount))


def _with_credit(store, number="NB-1004", credit="30.00"):
    account = store.get_account(number)
    account.credit_balance = D(credit)
    store.save_account(account)
    return account


def test_refund_to_a_foreign_destination_never_executes(seeded_store):
    conv = _conv(seeded_store, "NB-1004", "CUST-004")
    _with_credit(seeded_store)
    record = actions.execute(seeded_store, conv, _refund("30.00", "9999"),
                             POLICY, confirmed=True)
    assert record.status == "blocked"
    assert record.block_reason == "refund_destination_not_on_file"
    assert seeded_store.get_account("NB-1004").credit_balance == D("30.00")


def test_refund_amount_comes_from_the_record(seeded_store):
    conv = _conv(seeded_store, "NB-1004", "CUST-004")
    _with_credit(seeded_store)
    # A proposal naming more than the record holds is refused:
    # the conversation does not set the amount.
    greedy = actions.execute(seeded_store, conv, _refund("45.00", "7781"),
                             POLICY, confirmed=True)
    assert greedy.status == "blocked"
    assert greedy.block_reason == "no_credit_balance"
    ok = actions.execute(seeded_store, conv, _refund("30.00", "7781"),
                         POLICY, confirmed=True)
    assert ok.status == "executed"
    assert seeded_store.get_account("NB-1004").credit_balance == D("0")


def test_refund_destination_refused_in_conversation(service):
    _with_credit(service.store)
    conv, result = _drive(
        service, "NB-1004", "60614",
        "I overpaid — I paid my bill twice by mistake. Please "
        "refund the extra to my bank account.")
    # The resumed turn is the refusal: the destination is refused
    # with the reason, and nothing is proposed or executed.
    assert "fraud-protection" in result.reply
    assert "card ending 7781" in result.reply
    assert "$30.00" in result.reply
    assert service.store.get_account("NB-1004").credit_balance == D("30.00")
    assert not [a for a in service.store.list_actions(conv.conversation_id)
                if a.status == "executed"]

    result = _say(service, conv.conversation_id,
                  "fine — send it back to the card on file then")
    assert result.action is not None
    assert result.action.status == "proposed"
    result = _say(service, conv.conversation_id, "yes")
    assert result.action.status == "executed"
    assert result.action.kind == ActionKind.REFUND_PAYMENT
    assert result.action.payload["method_last4"] == "7781"
    assert service.store.get_account("NB-1004").credit_balance == D("0")


# ---------------------------------------------------------------------------
# Autopay: the run, the notice, the next contact
# ---------------------------------------------------------------------------


def _autopay_account(store, number="NB-1007", code=""):
    account = store.get_account(number)
    account.autopay_enabled = True
    account.processor_code = code
    store.save_account(account)
    return account


def test_autopay_needs_autopay_on_and_a_balance(seeded_store):
    attempt, notice = payments_book.run_autopay(
        seeded_store, "NB-1007", POLICY)
    assert attempt is None and notice is None  # autopay is off


def test_autopay_decline_raises_the_notice(seeded_store):
    _autopay_account(seeded_store, code="insufficient_funds")
    attempt, notice = payments_book.run_autopay(
        seeded_store, "NB-1007", POLICY)
    assert attempt.outcome == "declined"
    assert attempt.source == "autopay"
    assert notice is not None
    assert notice.amount == D("45.00")
    assert notice.method_last4 == "3098"
    assert "$45.00" in notice.summary
    assert "3098" in notice.summary
    assert "insufficient funds" in notice.summary
    assert seeded_store.get_account("NB-1007").balance_due == D("45.00")

    # The customer tops up; the next run settles the balance and
    # raises no second notice.
    account = seeded_store.get_account("NB-1007")
    account.processor_code = ""
    seeded_store.save_account(account)
    attempt, notice = payments_book.run_autopay(
        seeded_store, "NB-1007", POLICY)
    assert attempt.outcome == "approved"
    assert notice is None
    assert seeded_store.get_account("NB-1007").balance_due == D("0")
    assert len(seeded_store.list_payment_notices("NB-1007")) == 1


def test_autopay_failure_opens_the_diagnosis_on_next_contact(service):
    _autopay_account(service.store, code="insufficient_funds")
    payments_book.run_autopay(service.store, "NB-1007", service.policy)
    conv, result = _drive(service, "NB-1007", "02139",
                          "has my autopay gone through this month?")
    # The resumed turn leads with the notice — figures verbatim.
    assert "$45.00" in result.reply
    assert "3098" in result.reply
    assert "insufficient funds" in result.reply
    failure = payments_book.latest_autopay_failure(
        service.store, "NB-1007")
    assert failure is not None and failure[1] is not None


def test_autopay_enrol_and_cancel_are_confirmed_actions(service):
    conv, result = _drive(service, "NB-1007", "02139",
                          "set up autopay for my bills")
    assert result.action is not None
    assert result.action.status == "proposed"
    assert result.action.kind == ActionKind.SET_AUTOPAY
    result = _say(service, conv.conversation_id, "yes")
    assert result.action.status == "executed"
    assert service.store.get_account("NB-1007").autopay_enabled is True
    result = _say(service, conv.conversation_id, "cancel autopay")
    assert result.action.status == "proposed"
    result = _say(service, conv.conversation_id, "yes")
    assert result.action.status == "executed"
    assert service.store.get_account("NB-1007").autopay_enabled is False


# ---------------------------------------------------------------------------
# The broken arrangement: a human's case, with the record attached
# ---------------------------------------------------------------------------


def test_broken_arrangement_escalates_with_its_record(service):
    conv, result = _drive(
        service, "NB-1015", "11215",
        "I missed a payment on my arrangement — what happens now?")
    assert result.escalated is True
    escalations = [e for e in service.store.list_escalations()
                   if e.conversation_id == conv.conversation_id]
    assert len(escalations) == 1
    assert escalations[0].queue == "general"
    assert escalations[0].reason == "broken_payment_arrangement"
    # The record itself derives broken, and the audit trail names it.
    arr = service.store.get_payment_arrangement("ARR-HIST-1015")
    assert payments_book.arrangement_state(arr) == "broken"
    audit = service.store.list_audit(conv.conversation_id)
    routed = [e for e in audit if e.kind == "payments"]
    assert routed and routed[-1].detail["arrangement_id"] == "ARR-HIST-1015"
    # And nothing was renegotiated: no new arrangement exists.
    assert len(service.store.list_payment_arrangements("NB-1015")) == 1
