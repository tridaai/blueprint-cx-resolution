"""Structured actions: the only way anything changes.

A specialist proposes an action as a typed record. Execution is a
separate act that re-checks, in code, at the moment it happens:

- consequential actions (money or account changes) require a
  VERIFIED identity on the conversation;
- they require the customer's explicit confirmation, recorded as
  the confirmation turn — the supervisor passes `confirmed=True`
  only for that;
- they re-run their kind's policy checks (caps, eligibility,
  balance floors) against the live account, because the world may
  have moved since the proposal.

Every execution and every block writes an audit event and an
ActionRecord. A blocked action is a normal outcome with a reason,
never an exception leaking into the conversation.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from . import payments_book, plan_changes
from .clock import today
from .config import Policy
from .identity import is_verified
from .models import (
    CONSEQUENTIAL_ACTIONS,
    ActionKind,
    ActionRecord,
    ArrangementInstallment,
    AuditEvent,
    Conversation,
    PaymentArrangement,
    ProposedAction,
)


def new_action_record(conv: Conversation, proposal: ProposedAction,
                      status: str = "proposed") -> ActionRecord:
    return ActionRecord(
        action_id=f"ACT-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        kind=proposal.kind,
        summary=proposal.summary,
        payload=proposal.payload,
        amount=proposal.amount,
        basis=proposal.basis,
        status=status,
        turn=conv.turns,
        account_number=conv.account_number or "",
        created_date=today().isoformat(),
    )


def _audit(store, conv: Conversation, record: ActionRecord) -> None:
    store.save_audit(AuditEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        kind="action",
        detail={"action_id": record.action_id, "action_kind": record.kind.value,
                "status": record.status, "block_reason": record.block_reason,
                "amount": str(record.amount) if record.amount is not None else None},
        turn=conv.turns,
    ))


def execute(store, conv: Conversation, proposal: ProposedAction,
            policy: Policy, *, confirmed: bool) -> ActionRecord:
    """Run a proposed action through the gates and, if it clears,
    apply it. Always returns the record (executed or blocked)."""
    record = new_action_record(conv, proposal)

    def blocked(reason: str) -> ActionRecord:
        record.status = "blocked"
        record.block_reason = reason
        store.save_action(record)
        _audit(store, conv, record)
        return record

    if proposal.kind in CONSEQUENTIAL_ACTIONS:
        if not is_verified(conv):
            return blocked("identity_not_verified")
        if not confirmed:
            return blocked("confirmation_missing")

    account = store.get_account(conv.account_number) if conv.account_number else None

    if proposal.kind == ActionKind.APPLY_CREDIT:
        if account is None:
            return blocked("no_account")
        cap = (policy.dispute_credit_cap
               if proposal.payload.get("credit_class") == "dispute"
               else policy.goodwill_credit_cap)
        if proposal.amount is None or proposal.amount > cap:
            return blocked("credit_above_cap")
        if proposal.amount <= 0:
            return blocked("credit_not_positive")
        if proposal.payload.get("credit_class") == "goodwill":
            # The rolling cap is re-read at execution, not trusted
            # from proposal time: goodwill already applied inside
            # the window plus this credit may not pass it.
            from .billing_book import credit_history

            history = credit_history(store, account.account_number, policy)
            if history.executed_total + proposal.amount \
                    > policy.goodwill_rolling_cap:
                return blocked("goodwill_rolling_cap_reached")
        account.balance_due = max(Decimal("0"), account.balance_due - proposal.amount)
        store.save_account(account)

    elif proposal.kind == ActionKind.ACTIVATE_OFFER:
        offer = store.get_offer(proposal.payload.get("offer_id", ""))
        if offer is None or account is None:
            return blocked("offer_or_account_missing")
        if account.plan_id not in offer.eligible_plan_ids:
            return blocked("plan_not_eligible")
        if offer.one_off_price:
            account.balance_due += offer.one_off_price
        account.open_case = f"offer_active:{offer.offer_id}"
        store.save_account(account)

    elif proposal.kind == ActionKind.CHANGE_PLAN:
        if account is None:
            return blocked("no_account")
        # Stale-proposal protection: a plan proposal is priced
        # against a fingerprint of the account at issue time. If
        # the account has moved since — another action executed, a
        # payment landed, a reroute retired the figures — the
        # execution refuses and the supervisor re-quotes; a
        # customer never confirms numbers that no longer match
        # their account.
        stamped = proposal.payload.get("fingerprint")
        if stamped and plan_changes.account_fingerprint(account) != stamped:
            return blocked("stale_proposal")
        if proposal.payload.get("reversal_of"):
            return _execute_reversal(store, conv, proposal, policy,
                                     record, blocked)
        new_plan = store.get_plan(proposal.payload.get("new_plan_id", ""))
        if new_plan is None:
            return blocked("plan_missing")
        if proposal.payload.get("effective") == "next_cycle":
            # A scheduled downgrade is what executes: the plan
            # does not move today and no money moves — the
            # schedule lands on the account, effective at the
            # cycle boundary the customer was quoted.
            account.pending_plan_id = new_plan.plan_id
            account.pending_plan_effective = proposal.payload.get(
                "effective_date", "")
            store.save_account(account)
        else:
            net = Decimal(str(proposal.payload.get("net", "0")))
            account.plan_id = new_plan.plan_id
            account.balance_due += net
            # An immediate change supersedes any scheduled one.
            account.pending_plan_id = None
            account.pending_plan_effective = ""
            store.save_account(account)

    elif proposal.kind == ActionKind.UPDATE_PAYMENT_METHOD:
        last4 = str(proposal.payload.get("last4", ""))
        if account is None or len(last4) != 4 or not last4.isdigit():
            return blocked("bad_card_last4")
        account.payment_method_last4 = last4
        # A new method has no history: the decline standing
        # against the old card was a fact about that card.
        account.processor_code = ""
        store.save_account(account)

    elif proposal.kind == ActionKind.TAKE_PAYMENT:
        if account is None:
            return blocked("no_account")
        amount = (proposal.amount if proposal.amount is not None
                  else Decimal(str(proposal.payload.get("amount", "0"))))
        if amount <= 0 or account.balance_due <= 0:
            return blocked("nothing_to_pay")
        # The hard-decline rule is re-checked here, at the moment
        # of the attempt — a method the processor has declared
        # dead is never run again, whatever was proposed.
        if payments_book.hard_decline_block(store, account) is not None:
            return blocked("hard_decline_retry_refused")
        attempt = payments_book.process_payment(
            store, account, amount, source="manual",
            conversation_id=conv.conversation_id,
            action_id=record.action_id)
        record.payload = {**proposal.payload,
                          "outcome": attempt.outcome,
                          "reason_code": attempt.reason_code,
                          "decline_class": attempt.decline_class,
                          "attempt_id": attempt.attempt_id}
        if attempt.outcome != "approved":
            return blocked(f"payment_declined_{attempt.reason_code}")
        # Floored at a cent-exact zero: max() against Decimal("0")
        # would hand back the unquantised zero, and the balance
        # would read "0" where every other figure reads "0.00".
        account.balance_due = max(Decimal("0.00"),
                                  account.balance_due - amount)
        store.save_account(account)
        # An approved payment pours into the active arrangement,
        # like any money in: earliest open slice first.
        payments_book.apply_payment(store, account.account_number, amount)

    elif proposal.kind == ActionKind.REFUND_PAYMENT:
        if account is None:
            return blocked("no_account")
        # Refunds go to the method on the record, and nowhere
        # else — re-checked here against the live account, so a
        # proposal naming any other destination can never run.
        destination = str(proposal.payload.get("method_last4", ""))
        if not payments_book.refund_destination_allowed(account, destination):
            return blocked("refund_destination_not_on_file")
        amount = (proposal.amount if proposal.amount is not None
                  else Decimal(str(proposal.payload.get("amount", "0"))))
        if amount <= 0 or account.credit_balance < amount:
            return blocked("no_credit_balance")
        account.credit_balance -= amount
        store.save_account(account)

    elif proposal.kind == ActionKind.SET_AUTOPAY:
        if account is None:
            return blocked("no_account")
        enabled = bool(proposal.payload.get("enabled"))
        if enabled and not account.payment_method_last4:
            return blocked("no_payment_method_on_file")
        account.autopay_enabled = enabled
        store.save_account(account)

    elif proposal.kind == ActionKind.PAYMENT_ARRANGEMENT:
        if account is None:
            return blocked("no_account")
        if account.balance_due < policy.arrangement_min_balance:
            return blocked("balance_below_arrangement_minimum")
        # One arrangement at a time, and a broken one is a
        # human's case: neither may be written over here.
        if payments_book.broken_arrangement(store,
                                            account.account_number):
            return blocked("arrangement_broken_human_only")
        if payments_book.active_arrangement(store,
                                            account.account_number):
            return blocked("arrangement_already_active")
        spec = proposal.payload.get("installments")
        if spec:
            installments = [
                ArrangementInstallment(
                    number=int(s["number"]),
                    amount=Decimal(str(s["amount"])),
                    due_date=str(s["due_date"]))
                for s in spec]
        else:
            installments = payments_book.plan_installments(
                account.balance_due, policy)
        arrangement = PaymentArrangement(
            arrangement_id=f"ARR-{uuid.uuid4().hex[:10]}",
            conversation_id=conv.conversation_id,
            account_number=account.account_number,
            total=sum((i.amount for i in installments), Decimal("0")),
            installments=installments, state="active",
            created_date=today().isoformat())
        store.save_payment_arrangement(arrangement)
        record.payload = {**proposal.payload,
                          "arrangement_id": arrangement.arrangement_id}
        account.open_case = f"arrangement:{arrangement.arrangement_id}"
        store.save_account(account)

    elif proposal.kind == ActionKind.BOOK_CALLBACK:
        if account is None and not conv.account_number:
            return blocked("no_account")
        if account is not None:
            account.open_case = f"callback:{proposal.payload.get('reason', 'support')}"
            store.save_account(account)

    record.status = "executed"
    store.save_action(record)
    _audit(store, conv, record)
    return record


def _execute_reversal(store, conv: Conversation, proposal: ProposedAction,
                      policy: Policy, record: ActionRecord,
                      blocked) -> ActionRecord:
    """Execute a plan-change reversal as a compensating pair.

    The original change record is never edited. Two new records
    land, both stamped with its id: the record this execution
    returns restores the plan (or cancels the schedule, when the
    change was a downgrade that never took effect), and a second
    record unwinds the money — the exact inverse of the original
    net, applied to the balance as its own audited act. The
    window and the not-already-reversed rule are re-checked here,
    at execution, not trusted from proposal time."""
    account = store.get_account(conv.account_number) if conv.account_number else None
    if account is None:
        return blocked("no_account")
    orig = next((r for r in store.list_actions()
                 if r.action_id == proposal.payload.get("reversal_of")), None)
    if orig is None or orig.status != "executed" \
            or orig.kind != ActionKind.CHANGE_PLAN:
        return blocked("reversal_target_missing")
    if orig.account_number != account.account_number:
        return blocked("reversal_target_other_account")
    if plan_changes.reversal_of(store, orig):
        return blocked("change_already_reversed")
    if not plan_changes.within_reversal_window(orig, policy):
        return blocked("reversal_window_closed")

    unwind = Decimal(str(proposal.payload.get("unwind_net", "0")))
    # Leg one: the plan. A change that already landed is put back;
    # a scheduled downgrade is cancelled before it ever lands.
    if orig.payload.get("effective") == "next_cycle":
        account.pending_plan_id = None
        account.pending_plan_effective = ""
    else:
        restore = proposal.payload.get("restore_plan_id", "")
        if restore:
            account.plan_id = restore
    store.save_account(account)
    record.payload = {**proposal.payload, "leg": "restore_plan"}
    record.status = "executed"
    store.save_action(record)
    _audit(store, conv, record)

    # Leg two: the money, as its own record — the exact inverse
    # of the original net (zero for a scheduled change, which
    # never moved money; the record still lands, so the pair is
    # always a pair).
    money_leg = new_action_record(conv, ProposedAction(
        kind=ActionKind.CHANGE_PLAN,
        summary=(f"unwind the money from the reversed change "
                 f"({orig.action_id})"),
        payload={"leg": "money_unwind", "reversal_of": orig.action_id,
                 "net": str(unwind)},
        amount=unwind,
        basis=f"Money leg of the reversal of {orig.action_id}."))
    account.balance_due += unwind
    store.save_account(account)
    money_leg.status = "executed"
    store.save_action(money_leg)
    _audit(store, conv, money_leg)
    return record
