"""Money math. Decimal end to end, strings at the boundaries, and
the two calculations the agent's promises rest on: proration for a
mid-cycle plan change and the credit a dispute is worth. Both live
here, in code, so a model never invents a figure — it can only
explain one this module produced.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(value: str | int | float | Decimal) -> Decimal:
    """Parse to a cent-quantised Decimal. Floats go through str() so
    19.99 means 19.99, not 19.989999…"""
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(value: Decimal) -> str:
    """$1,234.56 — the only rendering of money a customer ever sees."""
    return f"${value:,.2f}"


def prorate(monthly_price: Decimal, days_remaining: int, days_in_cycle: int) -> Decimal:
    """Charge for the unused/used fraction of a billing cycle when a
    plan changes mid-cycle. days are whole days; the result is exact
    to the cent, half-up — the same rule the billing system uses, so
    the figure the agent quotes is the figure that lands."""
    if days_in_cycle <= 0:
        raise ValueError("days_in_cycle must be positive")
    if not 0 <= days_remaining <= days_in_cycle:
        raise ValueError("days_remaining must fall inside the cycle")
    raw = monthly_price * Decimal(days_remaining) / Decimal(days_in_cycle)
    return raw.quantize(CENT, rounding=ROUND_HALF_UP)


def plan_change_net(old_price: Decimal, new_price: Decimal,
                    days_remaining: int, days_in_cycle: int) -> dict:
    """The full proration story for a mid-cycle change: credit for
    the unused part of the old plan, charge for the remaining part of
    the new one, and the net the customer will see. Positive net =
    amount due on the next bill; negative = credit balance."""
    credit = prorate(old_price, days_remaining, days_in_cycle)
    charge = prorate(new_price, days_remaining, days_in_cycle)
    return {
        "credit_for_old_plan": credit,
        "charge_for_new_plan": charge,
        "net": (charge - credit).quantize(CENT, rounding=ROUND_HALF_UP),
        "days_remaining": days_remaining,
        "days_in_cycle": days_in_cycle,
    }
