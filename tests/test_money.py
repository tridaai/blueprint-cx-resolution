from decimal import Decimal

import pytest

from cx_agent.money import fmt, money, plan_change_net, prorate


def test_money_parses_without_float_error():
    assert money(19.99) == Decimal("19.99")
    assert money("7.5") == Decimal("7.50")
    assert money(3) == Decimal("3.00")


def test_fmt_renders_customer_money():
    assert fmt(Decimal("1234.5")) == "$1,234.50"
    assert fmt(Decimal("0.5")) == "$0.50"


def test_prorate_exact_days():
    assert prorate(Decimal("45.00"), 11, 30) == Decimal("16.50")
    assert prorate(Decimal("60.00"), 11, 30) == Decimal("22.00")
    assert prorate(Decimal("110.00"), 19, 30) == Decimal("69.67")
    assert prorate(Decimal("75.00"), 19, 30) == Decimal("47.50")
    assert prorate(Decimal("45.00"), 0, 30) == Decimal("0.00")
    assert prorate(Decimal("45.00"), 30, 30) == Decimal("45.00")


def test_prorate_rejects_nonsense():
    with pytest.raises(ValueError):
        prorate(Decimal("45"), 31, 30)
    with pytest.raises(ValueError):
        prorate(Decimal("45"), 10, 0)


def test_plan_change_net_story():
    story = plan_change_net(Decimal("110.00"), Decimal("75.00"), 19, 30)
    assert story["credit_for_old_plan"] == Decimal("69.67")
    assert story["charge_for_new_plan"] == Decimal("47.50")
    assert story["net"] == Decimal("-22.17")
