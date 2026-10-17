"""Round 6: roaming depth.

Quotes are priced from the rate table and the customer's own
usage profile, in Decimal, and the cheaper way of paying is the
one recommended — including when that's pay-as-you-go. A
projection past the bill-shock threshold leads and defers any
chargeable proposal a turn, and the same guard fires from actual
usage past the ceiling. Fair-use crossings raise stored notices
with the computed figures; the 100% crossing with no earlier
warning is a policy event on the record. And a roaming dispute
is decided from the records — pass window, warning given, or an
unwarned crossing — through the dispute machinery.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from cx_agent import roaming_book
from cx_agent.billing_book import assess_dispute
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import Account
from cx_agent.money import fmt

D = Decimal


@pytest.fixture()
def policy():
    return Policy()


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _drive(service, account_number, postcode, opener):
    """Open with a request that needs the account, answer the
    identity flow, and let the code turn resume the request."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, opener)
    _say(service, conv.conversation_id,
         f"{account_number} postcode {postcode}")
    result = _say(service, conv.conversation_id,
                  expected_code(account_number))
    return conv, result


# ---------------------------------------------------------------------------
# The rate table and the quotes
# ---------------------------------------------------------------------------


def test_rates_are_data_and_aliases_resolve(seeded_store):
    assert len(seeded_store.list_roaming_rates()) == 16
    assert roaming_book.rate_for(seeded_store, "Spain").per_gb == D("8.50")
    assert roaming_book.rate_for(seeded_store, "UAE").destination == \
        "United Arab Emirates"
    assert roaming_book.rate_for(seeded_store, "Atlantis") is None
    found = roaming_book.find_destination(
        seeded_store, "we're off to Greece in June")
    assert found.destination == "Greece" and found.per_gb == D("10.00")


def test_quote_pass_wins_for_a_heavy_trip(seeded_store, policy):
    account = seeded_store.get_account("NB-1007")  # 0.9 GB/day typical
    quote = roaming_book.quote_trip(seeded_store, account, "Spain", 8, policy)
    assert quote.expected_gb == D("7.2")
    assert quote.payg_cost == D("61.20")          # 7.2 × $8.50
    assert quote.pass_cost == D("30.00")          # fits the 1 GB/day allowance
    assert quote.cheaper == "pass"
    assert quote.saving == D("31.20")
    assert quote.shock is True                    # $61.20 past the $50 guard


def test_quote_payg_wins_for_a_light_trip(seeded_store, policy):
    account = seeded_store.get_account("NB-1004")  # 0.2 GB/day typical
    quote = roaming_book.quote_trip(seeded_store, account, "Spain", 5, policy)
    assert quote.expected_gb == D("1.0")
    assert quote.payg_cost == D("8.50")
    assert quote.pass_cost == D("30.00")
    assert quote.cheaper == "payg"
    assert quote.saving == D("21.50")
    assert quote.shock is False


def test_quote_prices_the_pass_overage(seeded_store, policy):
    account = seeded_store.get_account("NB-1003")  # 1.8 GB/day typical
    quote = roaming_book.quote_trip(seeded_store, account, "Japan", 6, policy)
    assert quote.expected_gb == D("10.8")
    assert quote.payg_cost == D("151.20")         # 10.8 × $14.00
    assert quote.pass_allowance_gb == D("6")     # 1 GB/day × 6 days
    assert quote.pass_overage_gb == D("4.8")
    assert quote.pass_overage_cost == D("67.20")
    assert quote.pass_cost == D("97.20")
    assert quote.cheaper == "pass"
    assert quote.saving == D("54.00")


def test_quote_refuses_a_destination_with_no_rate(seeded_store, policy):
    account = seeded_store.get_account("NB-1007")
    assert roaming_book.quote_trip(
        seeded_store, account, "Atlantis", 5, policy) is None


# ---------------------------------------------------------------------------
# The desk: recommendations, the guard, and its answer
# ---------------------------------------------------------------------------


def test_planning_recommends_the_pass_and_the_guard_defers_the_sale(service):
    conv, result = _drive(
        service, "NB-1007", "02139",
        "I'm travelling to Spain for 8 days — what will roaming cost "
        "me, is a pass cheaper?")
    for figure in ("$61.20", "$30.00", "$31.20"):
        assert figure in result.reply
    # The projection passed the threshold: it led, and nothing
    # chargeable was proposed this turn.
    assert result.action is None
    # The guard held the floor: a bare assent completes the
    # purchase it deferred — proposed first, executed on confirm.
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "proposed"
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "executed"
    account = service.store.get_account("NB-1007")
    assert account.balance_due == D("75.00")


def test_planning_recommends_payg_honestly_for_a_light_trip(service):
    conv, result = _drive(
        service, "NB-1004", "60614",
        "I'm going to Spain for 5 days — what would roaming cost me "
        "without a pass?")
    assert "$8.50" in result.reply and "$30.00" in result.reply
    assert "$21.50" in result.reply
    assert "Pay-as-you-go is cheaper" in result.reply
    assert result.action is None  # no pass pushed at a light user


def test_planning_guard_orders_projection_before_comparison(service):
    conv, result = _drive(
        service, "NB-1003", "10001",
        "I'm travelling to Japan for 6 days — what will roaming cost "
        "me, is a pass cheaper?")
    assert result.action is None
    assert result.reply.index("$151.20") < result.reply.index("$97.20")
    assert "$54.00" in result.reply


# ---------------------------------------------------------------------------
# Fair-use progress and the notices
# ---------------------------------------------------------------------------


def _traveller(store, number="NB-T", used="0", since="2026-09-28",
               country="Spain"):
    account = Account(account_number=number, customer_id="CUST-T",
                      plan_id="PLAN-ESS", roaming_country=country,
                      roaming_gb_used=D(used), roaming_since=since,
                      typical_daily_data_gb=D("1.0"))
    store.save_account(account)
    return account


def test_fair_use_progress_is_computed(seeded_store, policy):
    progress = roaming_book.fair_use_progress(
        seeded_store.get_account("NB-1002"), policy)
    assert progress["used"] == D("31")
    assert progress["cap"] == D("25")
    assert progress["pct"] == D("124.0")


def test_notices_raise_at_each_mark_and_never_twice(seeded_store, policy):
    account = _traveller(seeded_store, used="21")  # 84% of the ceiling
    created = roaming_book.sync_fair_use_notices(seeded_store, account, policy)
    assert [n.kind for n in created] == ["fair_use_80"]
    assert created[0].pct == D("84.0")
    assert roaming_book.sync_fair_use_notices(
        seeded_store, account, policy) == []

    account, created = roaming_book.record_roaming_usage(
        seeded_store, "NB-T", D("5.5"), policy)  # 26.5 GB — past the ceiling
    assert account.roaming_gb_used == D("26.5")
    assert [n.kind for n in created] == ["fair_use_100"]
    # An 80% warning was already on record: this crossing was warned.
    assert created[0].unwarned_crossing is False
    assert len(seeded_store.list_roaming_notices("NB-T")) == 2


def test_unwarned_crossing_is_a_policy_event_on_the_record(
        seeded_store, policy, monkeypatch):
    monkeypatch.setattr(roaming_book, "today", lambda: date(2026, 10, 1))
    account = _traveller(seeded_store, used="26", since="2026-09-28")
    created = roaming_book.sync_fair_use_notices(seeded_store, account, policy)
    assert [n.kind for n in created] == ["fair_use_80", "fair_use_100"]
    crossing = created[-1]
    assert crossing.unwarned_crossing is True
    # 26 GB over 4 days = 6.5 GB/day; carried to a 10-day trip,
    # 65 GB — 40 GB past the ceiling at $8.50 = $340.00.
    assert crossing.projected_overage_cost == D("340.00")
    assert "$340.00" in crossing.summary
    assert "26 GB" in crossing.summary and "25 GB" in crossing.summary


def test_crossing_reaches_the_customer_in_conversation(service):
    account = service.store.get_account("NB-1002")
    account.roaming_gb_used = D("26")
    service.store.save_account(account)
    conv, result = _drive(
        service, "NB-1002", "90210", "how much roaming data have I used?")
    assert "Fair-use warning" in result.reply
    assert "26 GB" in result.reply
    notices = service.store.list_roaming_notices("NB-1002")
    assert sorted(n.kind for n in notices) == ["fair_use_100", "fair_use_80"]
    crossing = next(n for n in notices if n.kind == "fair_use_100")
    assert crossing.unwarned_crossing is True


# ---------------------------------------------------------------------------
# Roaming disputes, decided from the records
# ---------------------------------------------------------------------------


def test_verdict_credits_an_unwarned_crossing(seeded_store, policy):
    account = seeded_store.get_account("NB-1013")
    bill = seeded_store.get_bill_for_account("NB-1013")
    verdicts = roaming_book.roaming_dispute_verdicts(
        seeded_store, account, bill, policy, "CONV-X")
    verdict = verdicts["CH-1013-2"]
    assert verdict.creditable == D("42.00")
    assert "no fair-use warning on record" in verdict.evidence


def test_verdict_stands_a_charge_under_an_active_pass(seeded_store, policy):
    account = seeded_store.get_account("NB-1014")
    bill = seeded_store.get_bill_for_account("NB-1014")
    verdicts = roaming_book.roaming_dispute_verdicts(
        seeded_store, account, bill, policy, "CONV-X")
    verdict = verdicts["CH-1014-2"]
    assert verdict.creditable == 0
    assert "Roam Pass was active" in verdict.evidence
    assert "2026-09-20" in verdict.evidence


def test_verdict_leaves_bill_marker_evidence_to_the_bill(
        seeded_store, policy):
    account = seeded_store.get_account("NB-1012")
    bill = seeded_store.get_bill_for_account("NB-1012")
    verdicts = roaming_book.roaming_dispute_verdicts(
        seeded_store, account, bill, policy, "CONV-X")
    # Felix's roaming line is convicted by its own wrong rate —
    # the records have nothing to add.
    assert "CH-1012-2" not in verdicts


def test_a_warning_on_record_stands_the_charge(
        seeded_store, policy, monkeypatch):
    monkeypatch.setattr(roaming_book, "today", lambda: date(2026, 10, 5))
    account = seeded_store.get_account("NB-1013")
    roaming_book.sync_fair_use_notices(
        seeded_store, account, policy, conversation_id="CONV-EARLIER")
    bill = seeded_store.get_bill_for_account("NB-1013")
    verdicts = roaming_book.roaming_dispute_verdicts(
        seeded_store, account, bill, policy, "CONV-NEW")
    verdict = verdicts["CH-1013-2"]
    assert verdict.creditable == 0
    assert "fair-use warning was given" in verdict.evidence
    # But a warning raised inside the disputing conversation
    # itself was produced by this contact, after the usage was
    # billed — it does not count as a prior warning.
    verdicts = roaming_book.roaming_dispute_verdicts(
        seeded_store, account, bill, policy, "CONV-EARLIER")
    assert verdicts["CH-1013-2"].creditable == D("42.00")


def test_roaming_dispute_credits_end_to_end(service):
    conv, result = _drive(
        service, "NB-1013", "10025",
        "I dispute a wrong charge on my bill — the roaming data")
    assert result.action is not None and result.action.status == "proposed"
    assert result.action.amount == D("42.00")
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "executed"
    case = service.store.list_dispute_cases(conversation_id=conv.conversation_id)[-1]
    assert case.state == "decided_credit"
    assert case.credit_applied == D("42.00")
    assert "no fair-use warning on record" in case.lines[0].evidence
    assert service.store.get_account("NB-1013").balance_due == D("50.40")


def test_roaming_dispute_declines_under_an_active_pass(service):
    conv, result = _drive(
        service, "NB-1014", "60607",
        "I dispute a wrong charge on my bill — the roaming data, and I had a pass")
    assert result.action is None
    case = service.store.list_dispute_cases(conversation_id=conv.conversation_id)[-1]
    assert case.state == "declined_with_reason"
    assert [line.decision for line in case.lines] == ["declined"]
    assert "Roam Pass was active" in case.lines[0].evidence
    assert service.store.get_account("NB-1014").balance_due == D("85.86")


def test_roaming_dispute_above_the_cap_escalates_with_its_evidence(service):
    conv, result = _drive(
        service, "NB-1002", "90210",
        "I dispute a wrong charge on my bill — the roaming data")
    assert result.escalated is True
    case = service.store.list_dispute_cases(conversation_id=conv.conversation_id)[-1]
    assert case.state == "escalated_above_cap"
    assert case.lines[0].decision == "escalated"
    assert "no fair-use warning on record" in case.lines[0].evidence
    # The evidenced total rode the assessment, cap and all.
    assessment = assess_dispute(
        service.store.get_bill_for_account("NB-1002"),
        "dispute the roaming charge", Policy(),
        roaming_book.roaming_dispute_verdicts(
            service.store, service.store.get_account("NB-1002"),
            service.store.get_bill_for_account("NB-1002"), Policy(),
            conv.conversation_id))
    assert assessment.evidenced_total == D("68.40")
    assert fmt(assessment.evidenced_total) == "$68.40"
