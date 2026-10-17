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

from .clock import today
from .config import Policy
from .identity import is_verified
from .models import (
    CONSEQUENTIAL_ACTIONS,
    ActionKind,
    ActionRecord,
    AuditEvent,
    Conversation,
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
        new_plan = store.get_plan(proposal.payload.get("new_plan_id", ""))
        if new_plan is None:
            return blocked("plan_missing")
        net = Decimal(str(proposal.payload.get("net", "0")))
        account.plan_id = new_plan.plan_id
        account.balance_due += net
        store.save_account(account)

    elif proposal.kind == ActionKind.UPDATE_PAYMENT_METHOD:
        last4 = str(proposal.payload.get("last4", ""))
        if account is None or len(last4) != 4 or not last4.isdigit():
            return blocked("bad_card_last4")
        account.payment_method_last4 = last4
        store.save_account(account)

    elif proposal.kind == ActionKind.PAYMENT_ARRANGEMENT:
        if account is None:
            return blocked("no_account")
        if account.balance_due < Decimal("100"):
            return blocked("balance_below_arrangement_minimum")
        account.open_case = (
            f"arrangement:{proposal.payload.get('instalments')}x"
            f"{proposal.payload.get('each')}")
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
