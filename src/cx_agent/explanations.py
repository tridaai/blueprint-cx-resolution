"""Second explanation forms.

Every explanation in the book has a first form — the policy
document's text, or the billing desk's family-by-family read of
the bill. When the signals say the customer did not follow it
(signals.detect_confusion: said outright, or the same question
asked again), the desk owes them the same facts in a DIFFERENT
shape, not the same words louder. That is what lives here:

- Summary-first restatements: the answer in one short version up
  front, the detail after.
- Worked examples: the rule walked through on sample figures,
  labelled as an example, so the shape of the thing is visible
  before the customer's own numbers are.

The forms are data, keyed by the topic the supervisor tracked
(the policy document a desk cited, or "bill" for the billing
desk's own-bill explanation). A topic with no entry here has no
second form — the confusion is still recorded, and the
frustration and no-progress rules still protect the customer;
nobody fakes a rephrase by re-serving the first form.

The bill's second form is composed from the customer's own
computed figures by bill_second_form below — summary-first: the
total and due date lead, the driver follows, and only then the
lines, read slowly.
"""

from __future__ import annotations

SECOND_FORMS: dict[str, str] = {
    "POL-BILL-01": (
        "The short version: if a charge looks wrong, name it and we check "
        "it line by line — anything the bill itself proves wrong comes "
        "back to you as a credit, anything that checks out stays, and "
        "anything over the agent's limit goes to a person with the case "
        "already written up. Say your bill shows Device protection at "
        "$9.99 twice in one month: that second line is proven wrong by "
        "the bill itself, so it is credited in full — you do not have to "
        "argue it, and you do not pay it while it is checked."
    ),
    "POL-BILL-03": (
        "Take a sample bill. It totals $58.54 for the month: a $45.00 "
        "plan line, a $9.99 one-off device protection charge, and $3.55 "
        "of tax — the total is just those lines added together. Your "
        "bill reads the same way, so the three things worth checking "
        "each month are the total, the due date, and any line that was "
        "not on last month's bill: a new line is usually the one that "
        "explains a jump."
    ),
    "POL-ROAM-02": (
        "Say you use 8 GB in France on a plan with a 10 GB fair-use "
        "amount: you are at 80%, so you get a heads-up message. Use the "
        "full 10 GB and you get a second message, and from there "
        "pay-as-you-go rates apply until the cycle resets. The clock "
        "that matters is your billing cycle, not the length of the "
        "trip — and both messages land before extra charges do."
    ),
    "POL-PLAN-01": (
        "The short version: upgrading starts today, and you pay only "
        "for the extra days — the part-month charge is worked out from "
        "your real cycle dates, not a flat fee. Downgrading waits "
        "until the cycle turns over, so you keep everything you have "
        "already paid for until then. Either way, the exact figures "
        "are quoted and confirmed with you before anything changes."
    ),
    "POL-PAY-01": (
        "Say you owe $120 and agree an arrangement over three "
        "instalments: $40 lands on the first date, $40 a month later, "
        "$40 a month after that — each amount and date is on the "
        "record before the first payment is due. Keep to the dates "
        "and nothing else is owed; miss one and the arrangement "
        "breaks, the full balance falls due, and a person picks the "
        "case up."
    ),
    "POL-ID-01": (
        "The short version: two steps. First your account number and "
        "the postcode on the account, so we open the right record; "
        "then the 6-digit code we send to the contact on file, so we "
        "know it is you holding it. Three wrong codes lock the chat "
        "and a person finishes verifying with you instead — the lock "
        "is there so nobody can keep guessing at your account."
    ),
    "POL-CARE-01": (
        "The short version: if you need a person, the support line is "
        "the round-the-clock option — any hour, any day, including "
        "account changes. Stores are the daytime option, 9am to 6pm "
        "Monday to Saturday, and they are the only place for SIM swaps "
        "and hands-on device checks. So: urgent or after hours, call "
        "the line; hands-on with a device, visit a store in the day."
    ),
}


def has_second_form(topic: str) -> bool:
    return topic in SECOND_FORMS


def second_form(topic: str) -> str:
    return SECOND_FORMS[topic]


def bill_second_form(issued_line: str, driver_line: str | None) -> str:
    """The customer's own bill, summary-first. `issued_line` is
    the billing desk's computed summary sentence (total, cycle,
    due date); `driver_line` the computed bill-shock read when
    there is one. The decomposition lines themselves still ride
    as must-include lines after this lead — the same figures,
    met in a different order: the answer first, the workings
    after."""
    text = f"In one line: {issued_line}"
    if driver_line:
        text += f" {driver_line}"
    text += (
        " Here is that same bill read slowly: each line below is "
        "one thing you pay for — the plan, the extras, the one-offs, "
        "the tax — and the total at the top is those lines added "
        "together, nothing more hidden underneath.")
    return text
