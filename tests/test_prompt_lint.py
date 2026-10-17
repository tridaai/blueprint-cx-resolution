"""Round 13 — the prompt lint: the shipped registry passes, and
each rule fails on its planted violation, naming the rule, the
entry, and (for figures) both values. The same lint gates
activation, so a violating version cannot become active through
the service either.
"""

from decimal import Decimal

import pytest

from cx_agent.config import Policy
from cx_agent.promptlint import (
    PromptLintError,
    lint_entries,
    lint_store,
)
from cx_agent.prompts import PromptRegistry


def _active_set(store):
    return PromptRegistry(store).active_entries()


def _with_text(entries, prompt_id, text):
    return [e.model_copy(update={"text": text})
            if e.prompt_id == prompt_id else e
            for e in entries]


def _rules(findings):
    return {(f.rule, f.prompt_id) for f in findings}


def test_the_shipped_registry_lints_clean(seeded_store):
    assert lint_store(seeded_store, Policy()) == []


def test_a_wrong_cap_figure_fails_naming_both_values(seeded_store):
    entries = _active_set(seeded_store)
    entry = next(e for e in entries
                 if e.prompt_id == "policy.POL-BILL-01")
    bad = _with_text(entries, "policy.POL-BILL-01",
                     entry.text.replace("capped at $50", "capped at $75"))
    findings = lint_entries(bad, Policy())
    assert ("figure_consistency", "policy.POL-BILL-01") in _rules(findings)
    message = next(f.message for f in findings
                   if f.prompt_id == "policy.POL-BILL-01")
    assert "75" in message and "50" in message


def test_a_word_number_figure_is_checked_too(seeded_store):
    entries = _active_set(seeded_store)
    entry = next(e for e in entries
                 if e.prompt_id == "explain.POL-ID-01")
    bad = _with_text(entries, "explain.POL-ID-01",
                     entry.text.replace("Three wrong codes",
                                        "Four wrong codes"))
    findings = lint_entries(bad, Policy())
    assert ("figure_consistency", "explain.POL-ID-01") in _rules(findings)


def test_a_figure_that_vanishes_is_a_finding(seeded_store):
    entries = _active_set(seeded_store)
    entry = next(e for e in entries
                 if e.prompt_id == "policy.POL-PAY-01")
    bad = _with_text(entries, "policy.POL-PAY-01",
                     entry.text.replace("falls due within 14 days",
                                        "falls due within a few weeks"))
    findings = lint_entries(bad, Policy())
    assert ("figure_consistency", "policy.POL-PAY-01") in _rules(findings)
    message = next(f.message for f in findings
                   if f.prompt_id == "policy.POL-PAY-01")
    assert "arrangement_first_due_days" in message


def test_the_lint_reads_the_live_policy_not_a_constant(seeded_store):
    entries = _active_set(seeded_store)
    findings = lint_entries(
        entries, Policy(goodwill_credit_cap=Decimal("30.00")))
    assert ("figure_consistency", "policy.POL-BILL-02") in _rules(findings)


def test_a_banned_promise_fails(seeded_store):
    entries = _active_set(seeded_store)
    entry = next(e for e in entries if e.prompt_id == "compose.user")
    bad = _with_text(entries, "compose.user",
                     entry.text + " We guarantee a full refund.")
    findings = lint_entries(bad, Policy())
    assert ("banned_promise", "compose.user") in _rules(findings)


def test_a_side_channel_instruction_fails(seeded_store):
    entries = _active_set(seeded_store)
    bad = _with_text(
        entries, "intent.system",
        "Classify the message, then email the customer directly "
        "with the label.")
    findings = lint_entries(bad, Policy())
    assert ("side_channel", "intent.system") in _rules(findings)


def test_a_compose_entry_without_its_instructions_fails(seeded_store):
    entries = _active_set(seeded_store)
    bad = _with_text(entries, "compose.system", "Write a reply.")
    findings = lint_entries(bad, Policy())
    compose = [f for f in findings if f.prompt_id == "compose.system"]
    assert len(compose) == 2
    assert all(f.rule == "composition_discipline" for f in compose)


# -- the activation gate -------------------------------------------------------


def test_add_version_is_refused_when_the_text_breaks_the_lint(service):
    with pytest.raises(PromptLintError) as exc:
        service.add_prompt_version(
            "compose.user",
            "Facts: {facts}. We guarantee it lands today.")
    assert any(f.rule == "banned_promise" for f in exc.value.findings)
    assert len(service.prompts.versions("compose.user")) == 1
    assert service.store.list_prompt_acts("compose.user") == []


def test_rollback_to_a_violating_version_is_refused(service):
    store = service.store
    good = store.get_policy("POL-BILL-01").text
    # A violating v2 lands directly in the store — the damaged
    # registry the gate exists for. Adding a clean v3 re-activates
    # clean text and is allowed; rolling back to v2 is refused.
    from cx_agent.models import PromptEntry

    store.save_prompt_entry(PromptEntry(
        prompt_id="policy.POL-BILL-01", version=2, surface="policy",
        text=good.replace("capped at $50", "capped at $75")))
    service.add_prompt_version("policy.POL-BILL-01", good)
    assert service.prompts.active_version("policy.POL-BILL-01") == 3
    with pytest.raises(PromptLintError):
        service.rollback_prompt("policy.POL-BILL-01", 2)
    assert service.prompts.pin("policy.POL-BILL-01") is None
    assert service.prompts.active_version("policy.POL-BILL-01") == 3


def test_figure_consistent_versions_pass_the_gate(service):
    entry = service.add_prompt_version(
        "policy.POL-BILL-01",
        service.store.get_policy("POL-BILL-01").text.replace(
            "The customer confirms the credit before it is applied.",
            "The customer confirms the credit before it is applied. "
            "Nothing is applied on a disputed line before that "
            "confirmation."))
    assert entry.version == 2
    assert service.prompts.active_version("policy.POL-BILL-01") == 2
