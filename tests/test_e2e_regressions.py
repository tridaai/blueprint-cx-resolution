"""Regressions found walking the agent end to end.

A trip longer than the Roam Pass is priced with the pass's own
window, not the trip's; a bill whose duplicate line repeats an
earlier one word for word still itemises every line; a desk that
gates on full verification still tells an unverified customer
what to send; and asking to replace a card reaches the account
desk.
"""

from __future__ import annotations

from decimal import Decimal

from cx_agent import roaming_book
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import Intent
from cx_agent.routing import route

D = Decimal


def test_pass_covers_only_its_own_window(seeded_store):
    account = seeded_store.get_account("NB-1009")  # 1.2 GB/day typical
    quote = roaming_book.quote_trip(
        seeded_store, account, "Japan", 14, Policy())
    assert quote.pass_days == 10                   # a 10-day pass
    assert quote.pass_allowance_gb == D("10")      # 1 GB/day × 10 days
    # 2 GB over inside the window, plus 4 uncovered days × 1.2 GB.
    assert quote.pass_overage_gb == D("6.8")
    assert quote.pass_cost == D("125.20")          # $30 + 6.8 × $14
    assert quote.payg_cost == D("235.20")
    assert quote.saving == D("110.00")


def test_unused_allowance_does_not_carry_past_the_window(seeded_store):
    account = seeded_store.get_account("NB-1007")  # 0.9 GB/day typical
    quote = roaming_book.quote_trip(
        seeded_store, account, "Spain", 14, Policy())
    assert quote.pass_overage_gb == D("3.6")       # 4 days × 0.9 GB
    assert quote.pass_cost == D("60.60")           # $30 + 3.6 × $8.50


def test_long_trip_quote_names_the_pass_window(service):
    service.seed()
    conv = service.start_conversation()
    cid = conv.conversation_id
    service.post_message(cid, "I'm going to Japan for 2 weeks, how much "
                              "will roaming cost?")
    service.post_message(cid, "NB-1009 postcode 98101")
    reply = service.post_message(cid, expected_code("NB-1009")).reply
    assert "over its 10 days" in reply
    assert "$125.20 all-in" in reply


def test_bill_itemisation_keeps_the_duplicate_line(service):
    service.seed()
    conv = service.start_conversation()
    cid = conv.conversation_id
    service.post_message(cid, "Why is my bill so high this month?")
    service.post_message(cid, "NB-1001 postcode 02134")
    reply = service.post_message(cid, expected_code("NB-1001")).reply
    assert "Device protection — monthly: $9.99" in reply
    assert "Device protection — monthly (charged again): $9.99" in reply


def test_verification_gate_asks_for_the_account_details(service):
    service.seed()
    for opener in ("update my card please", "turn off autopay"):
        conv = service.start_conversation()
        reply = service.post_message(conv.conversation_id, opener).reply
        assert "account number" in reply and "postcode" in reply


def test_replacing_a_card_routes_to_the_account_desk():
    assert route("I want to replace my card") == Intent.ACCOUNT
    assert route("I need a replacement card on my account") == \
        Intent.ACCOUNT
