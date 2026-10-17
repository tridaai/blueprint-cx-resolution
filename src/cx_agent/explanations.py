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

The forms are prompt-registry data (round 13), keyed by the
topic the supervisor tracked (the policy document a desk cited,
or "bill" for the billing desk's own-bill explanation): each is
the `explain.<topic>` entry, versioned and linted like every
governed text, and resolved from the store at serve time — there
is no second copy in this module. A topic with no entry has no
second form — the confusion is still recorded, and the
frustration and no-progress rules still protect the customer;
nobody fakes a rephrase by re-serving the first form.

The bill's second form is the exception that proves the
classification: it is composed from the customer's own computed
figures by bill_second_form below — summary-first: the total and
due date lead, the driver follows, and only then the lines, read
slowly. It is a deterministic template over live figures, not a
stored text, so it is not a registry entry.
"""

from __future__ import annotations

from .prompts import PromptRegistry, explanation_prompt_id


def has_second_form(store, topic: str) -> bool:
    return PromptRegistry(store).version_or_none(
        explanation_prompt_id(topic)) is not None


def second_form(store, topic: str) -> str:
    return PromptRegistry(store).active_text(
        explanation_prompt_id(topic))


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
