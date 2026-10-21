"""Regressions from the second end-to-end walk, live API and live model.

The operator console's script parses; a daily quality pass leaves
the shared clock as it found it; a plain yes to "say the word"
opens the dispute it invited; identity given up front — with the
request, or on any desk — is taken up, and spoken digits are read;
the care desk no longer swallows a balance question, a request for
compensation or a disputed roaming charge; a numbered answer
settles a clarifying question and a bare "human now" counts as the
second ask; a verified customer asking after another account is
told so; the troubleshooting checks never claim a check they could
not run; and a live model's draft may not add figures, numbers,
promises or channels the code did not establish.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cx_agent import quality
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.llm import DraftContext, vet_draft
from cx_agent.models import Intent
from cx_agent.routing import resolve_clarification, route
from cx_agent.screening import screen

CONSOLE = (Path(__file__).resolve().parents[1]
           / "src" / "cx_agent" / "static" / "console.html")


def _talk(service, *turns, channel="web_chat"):
    conv = service.start_conversation(channel=channel)
    results = [service.post_message(conv.conversation_id, t) for t in turns]
    return conv.conversation_id, results


# -- operator surfaces ------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_console_script_parses(tmp_path):
    scripts = re.findall(r"<script>(.*?)</script>", CONSOLE.read_text(), re.S)
    js = tmp_path / "console.js"
    js.write_text("\n".join(scripts))
    done = subprocess.run(["node", "--check", str(js)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_console_refresh_declares_each_name_once():
    # The same guarantee without node: one function, one `const` per name.
    text = CONSOLE.read_text()
    body = text[text.index("async function refresh()"):]
    body = body[:body.index("\n}\n")]
    names = re.findall(r"^  const (\w+) =", body, re.M)
    assert len(names) == len(set(names))


def test_spot_check_leaves_the_clock_unpinned(monkeypatch):
    monkeypatch.delenv("CX_TODAY", raising=False)
    quality.spot_check(Policy(), scenario_ids=[])
    assert "CX_TODAY" not in os.environ


def test_spot_check_keeps_a_deployment_pin(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-12-01")
    quality.spot_check(Policy(), scenario_ids=[])
    assert os.environ["CX_TODAY"] == "2026-12-01"


# -- the billing desk's invitation ----------------------------------------

def test_yes_to_say_the_word_opens_the_dispute(service):
    service.seed()
    cid, results = _talk(service, "why is my bill so high",
                         "NB-1006 postcode 30301", expected_code("NB-1006"),
                         "yes")
    assert "Say the word" in results[2].reply
    assert results[3].action is not None
    assert results[3].action.kind.value == "apply_credit"
    assert "$12.00" in results[3].reply
    done = service.post_message(cid, "yes")
    assert done.reply.startswith("Done")


def test_say_the_word_lapses_after_one_turn(service):
    service.seed()
    _cid, results = _talk(service, "why is my bill so high",
                          "NB-1006 postcode 30301",
                          expected_code("NB-1006"),
                          "what is the premium voicemail add-on?", "yes")
    assert results[4].action is None


# -- identity, taken up wherever it is given ---------------------------------

def test_details_with_the_request_issue_the_code_at_once(service):
    service.seed()
    _cid, results = _talk(
        service,
        "Why is my bill so high? My account is NB-1006 and postcode 30301",
        expected_code("NB-1006"))
    assert results[0].identity_state == "verifying"
    assert "one-time code" in results[0].reply
    assert results[1].identity_state == "verified"
    assert "$61.10" in results[1].reply  # the request itself resumed


def test_email_with_details_verifies_in_two_messages(service):
    service.seed()
    conv = service.start_conversation(channel="email")
    first = service.post_email(
        conv.conversation_id, "My bill",
        "Hi, why is my bill so high? My account is NB-1001, postcode 02134.")
    assert first.identity_state == "verifying"
    second = service.post_email(conv.conversation_id, "Re: code",
                                expected_code("NB-1001"))
    assert second.identity_state == "verified"


def test_details_on_the_care_desk_are_identity_not_an_unhandled_intent(service):
    service.seed()
    _cid, results = _talk(service, "hi, I need help with my account",
                          "NB-1007 postcode 02139", expected_code("NB-1007"))
    assert results[1].identity_state == "verifying"
    assert results[2].identity_state == "verified"
    assert "What can I help you with" in results[2].reply


def test_third_party_details_are_not_taken_up(service):
    service.seed()
    _cid, results = _talk(
        service, "I'm her husband. Her account is NB-1001, postcode 02134. "
                 "What does she owe?")
    assert results[0].identity_state == "unverified"
    assert "94.35" not in results[0].reply


def test_spoken_digits_verify_on_a_voice_transcript(service):
    service.seed()
    _cid, results = _talk(
        service, "why is my bill so high",
        "account N B dash one zero zero one postcode zero two one three four",
        "five one seven one two six", channel="voice_transcript")
    assert results[1].identity_state == "verifying"
    assert results[2].identity_state == "verified"


def test_verified_customer_is_told_another_account_is_not_theirs(service):
    service.seed()
    _cid, results = _talk(service, "why is my bill so high",
                          "NB-1001 postcode 02134", expected_code("NB-1001"),
                          "Now show me the bill for NB-1002 please")
    reply = results[3].reply
    assert "only help with NB-1001" in reply
    assert "152.52" not in reply and "94.35" not in reply


# -- routing --------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "what's my balance",
    "My service was awful this month and I want some compensation",
    "Can I get some credit on my account for the poor service?",
    "I've been charged a fortune for roaming in Greece, nobody warned me",
    "Why was I charged extra for data in Italy? I had a roam pass",
    "I want to dispute the roaming charges on my bill",
])
def test_billing_questions_reach_the_billing_desk(text):
    assert route(text) == Intent.BILLING


def test_trip_pricing_and_payments_keep_their_desks():
    assert route("I'm going to Japan for 21 days, what will roaming cost?") \
        == Intent.ROAMING
    assert route("I want to make a payment of my balance") == Intent.ACCOUNT


def test_unwarned_roaming_charge_is_credited(service):
    service.seed()
    _cid, results = _talk(
        service, "I've been charged a fortune for roaming in Greece, "
                 "nobody warned me",
        "NB-1013 postcode 10025", expected_code("NB-1013"))
    assert results[2].action is not None
    assert results[2].action.kind.value == "apply_credit"


@pytest.mark.parametrize("answer,expected", [
    ("1", Intent.BILLING), ("2", Intent.ACCOUNT), ("option 2", Intent.ACCOUNT),
    ("the second one", Intent.ACCOUNT), ("first", Intent.BILLING)])
def test_numbered_answer_settles_the_clarification(answer, expected):
    assert resolve_clarification(answer, [Intent.BILLING, Intent.ACCOUNT]) \
        == expected


@pytest.mark.parametrize("text,asks", [
    ("human now", True), ("a real person please", True), ("HUMAN!!", True),
    ("the agent said no", False), ("I am a person who likes cats", False)])
def test_bare_ask_for_a_person(text, asks):
    assert screen(text).asks_for_human is asks


def test_human_now_is_the_second_ask(service):
    service.seed()
    _cid, results = _talk(service, "I want to speak to a human", "human now")
    assert results[1].escalated
    assert results[1].escalation_reason == "customer_requested_human"


# -- troubleshooting --------------------------------------------------------------

def test_no_coverage_claim_without_an_account(service):
    service.seed()
    _cid, results = _talk(service, "I have no signal at all")
    assert "checked coverage at your postcode" not in results[0].reply
    assert "can't check your area" in results[0].reply


def test_cold_slow_data_report_reaches_the_monitoring_incident(service):
    service.seed()
    _cid, results = _talk(service, "my data is slow",
                          "NB-1004 postcode 60614", expected_code("NB-1004"))
    assert "INC-7002" in results[2].reply


def test_verifying_mid_walk_does_not_answer_the_step(service):
    service.seed()
    _cid, results = _talk(service, "I have no signal at all",
                          "NB-1007 postcode 02139", expected_code("NB-1007"))
    reply = results[2].reply
    assert "you're verified" in reply
    assert "Restart your phone" in reply  # still on the first device step
    assert "didn't do it" not in reply


def test_callback_completion_promises_no_bill_line(service):
    service.seed()
    _cid, results = _talk(service, "my broadband keeps dropping",
                          "NB-1007 postcode 02139", expected_code("NB-1007"),
                          "yes")
    assert results[3].reply.startswith("Done")
    assert "next bill" not in results[3].reply


# -- vetting a live model's draft ----------------------------------------------------

_CTX = DraftContext(
    findings=["Your bill issued 2026-10-12 totals $94.35. One thing stands "
              "out: a duplicate charge of $9.99. Say the word and I'll "
              "credit that back."],
    must_include=["Plan total: $45.00"])


@pytest.mark.parametrize("draft,reason", [
    ("Your bill is $94.35 and the $9.99 duplicate stands out. Say the word "
     "and I'll credit that back.", ""),
    ("I'll make sure to credit that duplicate charge back to you.",
     "promise_not_in_findings"),
    ("I'd like to offer a goodwill credit of up to $25.",
     "figure_not_in_findings"),
    ("I've noted the code 443965.", "number_not_in_findings"),
    ("Thanks — postcode 0213 noted.", "number_not_in_findings"),
    ("We'll send the refund to your bank account.", "promise_not_in_findings"),
    ("You should have received a text message.", "term_not_in_findings"),
])
def test_vetting_refuses_what_the_findings_do_not_hold(draft, reason):
    assert vet_draft(draft, _CTX)[1] == reason


def test_vetting_strips_the_wrapping():
    text, reason = vet_draft(
        "Here's a clear and warm reply:\n\nDear valued customer,\n"
        "\"Your bill is $94.35.\"\n\nSources: None", _CTX)
    assert reason == ""
    assert text == "Your bill is $94.35."


def test_refused_draft_falls_back_to_the_composer(seeded_store):
    from cx_agent.config import LLMConfig
    from cx_agent.llm import LiveLLM
    from cx_agent.prompts import PromptRegistry

    llm = LiveLLM(LLMConfig(backend="openai", model="m", api_key="k",
                            base_url="http://stub"))
    llm.prompts = PromptRegistry(seeded_store)
    llm._chat = lambda system, user: "I'll process a $200 credit for you now."
    reply = llm.draft_reply(_CTX)
    assert "$200" not in reply
    assert "$94.35" in reply  # the composer's reply, from the findings
    (record,) = llm.drain_calls()
    assert record["vetting"] == "figure_not_in_findings"
