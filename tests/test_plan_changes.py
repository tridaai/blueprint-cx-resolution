"""Round 5: plan-change transactions.

The quote is priced off the account's real cycle calendar; the
effective date is policy (upgrades now, downgrades at the next
cycle); a proposal carries the fingerprint of the state it was
priced against and execution refuses a stale one; a completed
change reverses inside its window as a compensating pair, and
outside it the request belongs to a human queue.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from cx_agent import actions as action_mod
from cx_agent import plan_changes
from cx_agent.identity import expected_code
from cx_agent.models import (
    ActionKind,
    Conversation,
    IdentityState,
    Plan,
    ProposedAction,
)
from cx_agent.money import plan_change_for_cycle
from cx_agent.money import plan_change_net


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


def _verified_conversation(service, account_number, postcode):
    """A verified conversation with nothing pending: the opener
    is a balance question the account desk answers on resume."""
    conv, _ = _drive(service, account_number, postcode, "pay my bill")
    return conv


# ---------------------------------------------------------------------------
# The quote: calendar arithmetic and effective dates
# ---------------------------------------------------------------------------


def test_cycle_proration_reads_the_calendar():
    story = plan_change_for_cycle(
        Decimal("45.00"), Decimal("60.00"),
        date(2026, 9, 13), date(2026, 10, 12), date(2026, 10, 1))
    assert story["days_in_cycle"] == 30
    assert story["days_remaining"] == 11
    assert story["days_used"] == 19
    assert story["credit_for_old_plan"] == Decimal("16.50")
    assert story["charge_for_new_plan"] == Decimal("22.00")
    assert story["net"] == Decimal("5.50")
    # The calendar form and the day-count form are the same rule
    # in two notations — they must never disagree.
    counted = plan_change_net(Decimal("45.00"), Decimal("60.00"), 11, 30)
    assert story["credit_for_old_plan"] == counted["credit_for_old_plan"]
    assert story["charge_for_new_plan"] == counted["charge_for_new_plan"]
    assert story["net"] == counted["net"]


def test_cycle_proration_edges_clamp_to_the_cycle():
    start, end = date(2026, 9, 13), date(2026, 10, 12)
    first = plan_change_for_cycle(Decimal("45"), Decimal("60"),
                                  start, end, start)
    assert first["days_remaining"] == 29  # today counts as used
    last = plan_change_for_cycle(Decimal("45"), Decimal("60"),
                                  start, end, end)
    assert last["days_remaining"] == 0
    assert last["net"] == Decimal("0")


def _account(**overrides):
    from cx_agent.models import Account

    base = dict(account_number="NB-9001", customer_id="CUST-900",
                plan_id="PLAN-ESS", balance_due=Decimal("50.00"),
                cycle_start="2026-09-13", cycle_end="2026-10-12",
                cycle_day=19, cycle_days=30)
    base.update(overrides)
    return Account(**base)


def test_quote_upgrade_is_immediate_and_projects_the_next_bill():
    ess = Plan(plan_id="PLAN-ESS", name="Essential 20GB",
               monthly_price=Decimal("45.00"))
    unl = Plan(plan_id="PLAN-UNL", name="Unlimited Max",
               monthly_price=Decimal("60.00"))
    quote = plan_changes.quote_change(_account(), ess, unl,
                                      Decimal("60.00"),
                                      on=date(2026, 10, 1))
    assert quote.direction == "upgrade"
    assert quote.effective == "immediate"
    assert quote.effective_date == "2026-10-01"
    assert quote.credit == Decimal("16.50")
    assert quote.charge == Decimal("22.00")
    assert quote.net == Decimal("5.50")
    # Net adjustment plus the first full month at the new price.
    assert quote.next_bill_impact == Decimal("65.50")


def test_quote_downgrade_waits_for_the_cycle_and_moves_nothing():
    fam = Plan(plan_id="PLAN-FAM", name="Family Share 100GB",
               monthly_price=Decimal("110.00"))
    unl = Plan(plan_id="PLAN-UNL", name="Unlimited Max",
               monthly_price=Decimal("75.00"))
    quote = plan_changes.quote_change(_account(plan_id="PLAN-FAM"),
                                      fam, unl, Decimal("75.00"),
                                      on=date(2026, 10, 1))
    assert quote.direction == "downgrade"
    assert quote.effective == "next_cycle"
    assert quote.effective_date == "2026-10-13"  # cycle_end + 1
    assert quote.credit == quote.charge == quote.net == Decimal("0")
    assert quote.next_bill_impact == Decimal("75.00")


def test_quote_lateral_move_is_immediate_and_free():
    a = Plan(plan_id="PLAN-A", name="A", monthly_price=Decimal("45.00"))
    b = Plan(plan_id="PLAN-B", name="B", monthly_price=Decimal("45.00"))
    quote = plan_changes.quote_change(_account(), a, b, Decimal("45.00"),
                                      on=date(2026, 10, 1))
    assert quote.direction == "lateral"
    assert quote.effective == "immediate"
    assert quote.net == Decimal("0")


def test_fingerprint_moves_with_the_account():
    account = _account()
    stamp = plan_changes.account_fingerprint(account)
    assert plan_changes.account_fingerprint(account) == stamp
    account.balance_due += Decimal("1.00")
    assert plan_changes.account_fingerprint(account) != stamp
    moved = plan_changes.fingerprint_diff(
        account, plan_changes.fingerprint_parts(_account()))
    assert moved == ["your balance changed"]
    account = _account(pending_plan_id="PLAN-BAS")
    assert plan_changes.account_fingerprint(account) != stamp


# ---------------------------------------------------------------------------
# Execution gates, driven directly
# ---------------------------------------------------------------------------


def test_execute_refuses_a_stale_proposal(seeded_store):
    account = seeded_store.get_account("NB-1001")
    proposal = ProposedAction(
        kind=ActionKind.CHANGE_PLAN, summary="change plan",
        payload={"new_plan_id": "PLAN-UNL", "net": "5.50",
                 "fingerprint": plan_changes.account_fingerprint(account),
                 "fingerprint_parts": plan_changes.fingerprint_parts(account)},
        amount=Decimal("5.50"), basis="test")
    conv = Conversation(conversation_id="CONV-STALE",
                        account_number="NB-1001",
                        identity_state=IdentityState.VERIFIED)
    # The world moves between proposal and confirmation: a payment
    # lands and the balance the figures rested on is gone.
    account.balance_due += Decimal("15.00")
    seeded_store.save_account(account)
    from cx_agent.config import Policy

    record = action_mod.execute(seeded_store, conv, proposal, Policy(),
                                confirmed=True)
    assert record.status == "blocked"
    assert record.block_reason == "stale_proposal"
    after = seeded_store.get_account("NB-1001")
    assert after.plan_id == "PLAN-ESS"  # nothing executed
    assert after.balance_due == Decimal("109.35")


# ---------------------------------------------------------------------------
# Conversation flows
# ---------------------------------------------------------------------------


def test_proposal_is_versioned_and_stamped(service):
    conv, result = _drive(service, "NB-1001", "02134",
                          "change my plan to Unlimited Max")
    assert result.action is not None and result.action.status == "proposed"
    payload = result.action.payload
    assert payload["proposal_version"] == 1
    account = service.store.get_account("NB-1001")
    assert payload["fingerprint"] == \
        plan_changes.account_fingerprint(account)
    assert payload["old_plan_id"] == "PLAN-ESS"
    assert payload["direction"] == "upgrade"
    assert payload["effective"] == "immediate"
    assert payload["next_bill_impact"] == "65.50"


def test_stale_confirmation_requotes_and_only_then_executes(service):
    conv, _ = _drive(service, "NB-1001", "02134",
                     "change my plan to Unlimited Max")
    # A charge lands through another channel before the customer
    # answers: the quoted figures no longer match the account.
    account = service.store.get_account("NB-1001")
    account.balance_due += Decimal("15.00")
    service.store.save_account(account)

    refused = _say(service, conv.conversation_id, "yes")
    assert refused.action is not None
    assert refused.action.status == "blocked"
    assert refused.action.block_reason == "stale_proposal"
    assert "went stale" in refused.reply
    assert "Nothing was applied" in refused.reply
    # Nothing executed: the plan is untouched, the balance is the
    # moved one, and the stale proposal is marked on the books.
    account = service.store.get_account("NB-1001")
    assert account.plan_id == "PLAN-ESS"
    assert account.balance_due == Decimal("109.35")
    records = service.store.list_actions(conv.conversation_id)
    stale = [r for r in records if r.status == "superseded"]
    assert len(stale) == 1
    assert "stale_proposal" in stale[0].block_reason
    # A fresh proposal is pending, re-versioned and re-stamped.
    fresh = service.store.get_conversation(conv.conversation_id).pending_action
    assert fresh is not None
    assert fresh.payload["proposal_version"] == 2
    assert fresh.payload["fingerprint"] == \
        plan_changes.account_fingerprint(account)

    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1001")
    assert account.plan_id == "PLAN-UNL"
    assert account.balance_due == Decimal("114.85")  # 109.35 + 5.50


def test_downgrade_schedules_at_the_boundary(service):
    conv, result = _drive(service, "NB-1003", "10001",
                          "change my plan to Unlimited Max")
    account = service.store.get_account("NB-1003")
    expected_date = (date.fromisoformat(account.cycle_end)
                     + timedelta(days=1)).isoformat()
    assert result.action.status == "proposed"
    assert result.action.payload["effective"] == "next_cycle"
    assert result.action.payload["effective_date"] == expected_date
    assert result.action.payload["net"] == "0"
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1003")
    assert account.plan_id == "PLAN-FAM"  # the live plan has not moved
    assert account.pending_plan_id == "PLAN-UNL"
    assert account.pending_plan_effective == expected_date
    assert account.balance_due == Decimal("118.60")  # nothing charged


def test_immediate_change_supersedes_a_scheduled_one(service):
    conv = _verified_conversation(service, "NB-1007", "02139")
    _say(service, conv.conversation_id, "change my plan to Basic Talk & Text")
    _say(service, conv.conversation_id, "yes")
    account = service.store.get_account("NB-1007")
    assert account.pending_plan_id == "PLAN-BAS"
    _say(service, conv.conversation_id, "change my plan to Unlimited Max")
    _say(service, conv.conversation_id, "yes")
    account = service.store.get_account("NB-1007")
    assert account.plan_id == "PLAN-UNL"
    assert account.pending_plan_id is None
    assert account.pending_plan_effective == ""


def test_reversal_inside_the_window_restores_via_the_pair(service):
    conv, _ = _drive(service, "NB-1005", "94107",
                     "change my plan to Unlimited Max")
    _say(service, conv.conversation_id, "yes")
    account = service.store.get_account("NB-1005")
    assert account.plan_id == "PLAN-UNL"
    assert account.balance_due == Decimal("84.77")  # 76.77 + 8.00 net
    orig = plan_changes.latest_executed_change(service.store, "NB-1005")
    assert orig is not None

    _say(service, conv.conversation_id, "undo that change, put me back")
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1005")
    assert account.plan_id == "PLAN-ESS"
    assert account.balance_due == Decimal("76.77")
    # The pair: two executed records pointing at the original,
    # one per leg — and the original itself is untouched history.
    pair = plan_changes.reversal_of(service.store, orig)
    assert len(pair) == 2
    assert {r.payload["leg"] for r in pair} == {"restore_plan", "money_unwind"}
    money = next(r for r in pair if r.payload["leg"] == "money_unwind")
    assert money.amount == Decimal("-8.00")
    original = next(r for r in service.store.list_actions()
                    if r.action_id == orig.action_id)
    assert original.status == "executed"
    assert original.payload["net"] == "8.00"

    again = _say(service, conv.conversation_id, "undo that change")
    assert "already reversed" in again.reply
    assert again.action is None or again.action.status != "proposed"


def test_reversal_outside_the_window_goes_to_the_offers_queue(service):
    conv, _ = _drive(service, "NB-1007", "02139",
                     "change my plan to Unlimited Max")
    _say(service, conv.conversation_id, "yes")
    orig = plan_changes.latest_executed_change(service.store, "NB-1007")
    made = date.fromisoformat(orig.created_date)
    orig.created_date = (made - timedelta(days=2)).isoformat()
    service.store.save_action(orig)

    result = _say(service, conv.conversation_id, "undo that change")
    assert result.escalated
    assert result.escalation_reason == "plan_reversal_outside_window"
    escalations = service.list_escalations()
    assert escalations[-1].queue == "offers"
    account = service.store.get_account("NB-1007")
    assert account.plan_id == "PLAN-UNL"  # not reversed by the agent
    assert plan_changes.reversal_of(service.store, orig) == []


def test_reversal_window_is_rechecked_at_execution(service):
    conv, _ = _drive(service, "NB-1005", "94107",
                     "change my plan to Unlimited Max")
    _say(service, conv.conversation_id, "yes")
    _say(service, conv.conversation_id, "undo that change")
    # The original ages past the window while the reversal waits
    # for its answer: execution must refuse, not trust the quote.
    orig = plan_changes.latest_executed_change(service.store, "NB-1005")
    made = date.fromisoformat(orig.created_date)
    orig.created_date = (made - timedelta(days=2)).isoformat()
    service.store.save_action(orig)
    refused = _say(service, conv.conversation_id, "yes")
    assert refused.action.status == "blocked"
    assert refused.action.block_reason == "reversal_window_closed"
    assert service.store.get_account("NB-1005").plan_id == "PLAN-UNL"


def test_reversal_of_a_scheduled_downgrade_cancels_the_schedule(service):
    conv, _ = _drive(service, "NB-1003", "10001",
                     "change my plan to Unlimited Max")
    _say(service, conv.conversation_id, "yes")
    account = service.store.get_account("NB-1003")
    assert account.pending_plan_id == "PLAN-UNL"
    _say(service, conv.conversation_id, "undo that change")
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    account = service.store.get_account("NB-1003")
    assert account.plan_id == "PLAN-FAM"
    assert account.pending_plan_id is None
    assert account.balance_due == Decimal("118.60")


def test_undo_with_no_change_is_an_honest_no(service):
    conv = _verified_conversation(service, "NB-1002", "90210")
    result = _say(service, conv.conversation_id, "undo my plan change")
    assert not result.escalated
    assert result.action is None
    assert "no plan change" in result.reply
    assert service.store.get_account("NB-1002").plan_id == "PLAN-UNL"


def test_fault_cited_downgrade_belongs_to_billing(service):
    conv = _verified_conversation(service, "NB-1001", "02134")
    result = _say(
        service, conv.conversation_id,
        "Please downgrade my plan to Basic Talk & Text — the service "
        "is broken and has been for weeks")
    assert result.escalated
    assert result.escalation_reason == "fault_cancellation_dispute_path"
    assert service.list_escalations()[-1].queue == "billing"
    account = service.store.get_account("NB-1001")
    assert account.plan_id == "PLAN-ESS"
    assert account.pending_plan_id is None
