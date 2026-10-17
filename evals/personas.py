"""The persona matrix: difficult customers, generated.

The hand-written scenarios in scenarios.jsonl carry one persona
each, chosen by hand. This module does the systematic version:
five core intent flows (a billing dispute, a plan change, a
roaming quote, a payment, a troubleshooting procedure) written
once as role-tagged turns, and seven persona archetypes written
as transformers over those turns — the confused customer who
repeats themselves and answers hesitantly, the terse customer,
the rambler who buries the ask, the non-native writer, the
customer who changes their mind mid-flow, the wall-of-text
paster, the careful elderly typer.

Every persona × flow pair becomes one eval scenario grading
exactly the invariants the plain flow grades: the same final
state, the same money to the cent, the same executed actions.
How a customer talks must not change what the agent does.

Two honest constraints, both deliberate:

- Payload turns (the one-time code above all) are never
  rewritten by any persona. A persona changes how a person
  talks, not the token the system issued them; wrapping the
  code in prose would test the code regex, not the agent.
- Confirmations stay inside the recognised confirmation
  vocabulary's neighbourhood (see routing.classify_confirmation
  for why the vocabulary is closed on purpose) — except the
  confused persona, whose hesitant first answer deliberately
  falls OUTSIDE it. That is the case the Round 9 fixes cover:
  an unreadable answer must be re-asked, never treated as a
  new topic that silently kills the pending decision.
"""

from __future__ import annotations

import copy
import re

# ---------------------------------------------------------------------------
# The five core flows. Turns carry a role so personas know how to
# touch them: opener / payload / code / answer / confirm /
# confirm_guard (an assent owed to the roaming bill-shock guard,
# which holds no pending action — a different mechanism from the
# supervisor's confirmation protocol, graded differently).
# ---------------------------------------------------------------------------

FLOWS: list[dict] = [
    {
        "id": "billing_dispute",
        "rubric": "transaction",
        "account": "NB-1001",
        "zt": True,
        "turns": [
            {"role": "opener", "say": "why is my bill so high",
             "expect": {"intent": "billing"}},
            {"role": "payload", "say": "NB-1001 postcode 02134",
             "expect": {"identity_state": "verifying"}},
            {"role": "code", "say": "{code:NB-1001}",
             "expect": {"identity_state": "verified"}},
            {"role": "answer",
             "say": "I dispute two charges: you charged me twice for "
                    "device protection, and the handset instalment — "
                    "I don't recognise it",
             "expect": {"action_status": "proposed",
                        "action_kind": "apply_credit",
                        "reply_contains": "instalment 14 of 24"}},
            {"role": "confirm", "say": "yes",
             "expect": {"action_status": "executed"}},
        ],
        "final": {"balance_due": "84.36", "executed_kinds": ["apply_credit"],
                  "dispute_state": "decided_partial",
                  "dispute_credit": "9.99",
                  "dispute_decisions": ["credited", "declined"]},
    },
    {
        "id": "plan_change",
        "rubric": "transaction",
        "account": "NB-1001",
        "zt": True,
        "turns": [
            {"role": "opener", "say": "I want to change my plan to Unlimited Max",
             "expect": {"intent": "offers"}},
            {"role": "payload", "say": "NB-1001 postcode 02134",
             "expect": {"identity_state": "verifying"}},
            {"role": "code", "say": "{code:NB-1001}",
             "expect": {"action_status": "proposed",
                        "reply_contains_all": [
                            "Credit for the unused part of Essential 20GB: $16.50",
                            "Charge for Unlimited Max for the remaining 11 days: $22.00",
                            "Net adjustment on your next bill: $5.50 due.",
                            "Projected next bill: $65.50"]}},
            {"role": "confirm", "say": "yes",
             "expect": {"action_status": "executed"}},
        ],
        "final": {"plan_id": "PLAN-UNL", "balance_due": "99.85",
                  "pending_plan_id": None, "executed_kinds": ["change_plan"]},
    },
    {
        "id": "roaming_quote",
        "rubric": "roaming",
        "account": "NB-1007",
        "zt": False,
        "turns": [
            {"role": "opener",
             "say": "I'm travelling to Spain for 8 days — what will "
                    "roaming cost me, is a pass cheaper?",
             "expect": {"intent": "roaming", "identity_state": "unverified"}},
            {"role": "payload", "say": "NB-1007 postcode 02139",
             "expect": {"identity_state": "verifying"}},
            {"role": "code", "say": "{code:NB-1007}",
             "expect": {"identity_state": "verified",
                        "reply_contains_all": ["$61.20", "$30.00", "$31.20"],
                        "reply_ordered": ["$61.20", "$30.00"],
                        "action_status": "none"}},
            {"role": "confirm_guard", "say": "yes",
             "expect": {"action_status": "proposed"}},
            {"role": "confirm", "say": "yes",
             "expect": {"action_status": "executed"}},
        ],
        "final": {"balance_due": "75.00", "executed_kinds": ["activate_offer"]},
    },
    {
        "id": "payment",
        "rubric": "transaction",
        "account": "NB-1003",
        "zt": False,
        "turns": [
            {"role": "opener", "say": "I'd like to pay my bill now, please",
             "poke": {"account": "NB-1003", "processor_code": "insufficient_funds"},
             "expect": {"intent": "account"}},
            {"role": "payload", "say": "NB-1003 postcode 10001",
             "expect": {"identity_state": "verifying"}},
            {"role": "code", "say": "{code:NB-1003}",
             "expect": {"action_status": "proposed",
                        "reply_contains": "$118.60"}},
            {"role": "confirm", "say": "yes",
             "expect": {"action_status": "blocked",
                        "reply_contains_all": ["insufficient funds",
                                               "Nothing was charged",
                                               "$118.60"]}},
            {"role": "answer",
             "say": "I've topped up — go ahead and make a payment again",
             "poke": {"account": "NB-1003", "processor_code": ""},
             "expect": {"action_status": "proposed"}},
            {"role": "confirm", "say": "yes",
             "expect": {"action_status": "executed",
                        "reply_contains": "Paid"}},
        ],
        "final": {"balance_due": "0.00",
                  "attempt_outcomes": ["declined", "approved"],
                  "attempt_codes": ["insufficient_funds", ""],
                  "executed_kinds": ["take_payment"]},
    },
    {
        "id": "technical",
        "rubric": "technical",
        "account": "NB-1006",
        "zt": False,
        "turns": [
            {"role": "opener", "say": "my broadband keeps dropping",
             "expect": {"intent": "technical",
                        "reply_contains": "account number"}},
            {"role": "payload", "say": "NB-1006 postcode 30301",
             "expect": {"identity_state": "verifying"}},
            {"role": "code", "say": "{code:NB-1006}",
             "expect": {"reply_contains": "unplug"}},
            {"role": "answer", "say": "done, it's unplugged",
             "expect": {"reply_contains": "speed test"}},
            {"role": "answer", "say": "it says 42",
             "expect": {"status": "resolved", "reply_contains": "healthy"}},
        ],
        "final": {"run_state": "resolved",
                  "run_history_steps": ["BB-1", "BB-2", "BB-3", "BB-7"]},
    },
]


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

_ACCT = re.compile(r"NB-\d+")
_POSTCODE = re.compile(r"\b\d{5}\b")


def _cap_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _sentence(text: str) -> str:
    """A canonical turn as a finished sentence, for personas who
    wrap prose around it."""
    return text if text.endswith((".", "!", "?")) else text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _payload_parts(say: str) -> tuple[str, str]:
    return _ACCT.search(say).group(0), _POSTCODE.search(say).group(0)


def _keep(turn: dict) -> list[dict]:
    return [turn]


# ---------------------------------------------------------------------------
# The personas. Each is a set of per-role transforms; a transform
# returns the list of turns that replace the canonical one
# (usually one; the confused confirmation returns two).
# ---------------------------------------------------------------------------

def _confused_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = f"Sorry, I'm a bit confused. {_cap_first(turn['say'])}"
    return [new]


def _confused_after_opener(turn, flow):
    # The repetition: the same ask again, word for word, while the
    # agent is waiting for identity details. The expectations are
    # the opener's own state assertions — the conversation must be
    # exactly where the first ask left it.
    dup = copy.deepcopy(turn)
    dup["say"] = f"Sorry, I'm a bit confused. {_cap_first(turn['say'])}"
    dup["expect"] = {k: v for k, v in turn["expect"].items()
                    if k in ("intent", "escalated", "identity_state",
                             "reply_contains")}
    return [dup]


def _confused_payload(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = f"Sorry, it's {turn['say']}"
    return [new]


def _confused_answer(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = f"Sorry, let me think. {_cap_first(turn['say'])}"
    return [new]


def _hesitant_then(turn, reask_fragment):
    """The confused confirmation: a first answer outside the
    confirmation vocabulary — the agent must re-ask, keeping the
    decision open — then the plain answer the flow expects."""
    hesitant = {"role": turn["role"],
                "say": "um, yes I think so",
                "expect": {"escalated": False,
                           "reply_contains": reask_fragment}}
    if turn.get("poke"):
        hesitant["poke"] = turn["poke"]
    plain = copy.deepcopy(turn)
    plain.pop("poke", None)
    return [hesitant, plain]


def _confused_confirm(turn, flow):
    return _hesitant_then(turn, "shall I go ahead")


def _confused_confirm_guard(turn, flow):
    return _hesitant_then(turn, "shall I add the Roam Pass")


_TERSE_OPENERS = {
    "billing_dispute": "why bill so high?",
    "plan_change": "change plan to Unlimited Max",
    "roaming_quote": "Spain, 8 days — roaming cost? pass cheaper?",
    "payment": "pay my bill",
    "technical": "broadband keeps dropping",
}

_TERSE_ANSWERS = {
    ("billing_dispute", 0):
        "dispute: charged twice for device protection. "
        "handset instalment — don't recognise it.",
    ("payment", 0): "topped up — pay my bill again",
    ("technical", 0): "done, unplugged",
    ("technical", 1): "says 42",
}


def _terse_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = _TERSE_OPENERS[flow["id"]]
    return [new]


def _terse_answer(turn, flow):
    new = copy.deepcopy(turn)
    ordinal = flow["_answer_ordinal"]
    key = (flow["id"], ordinal)
    if key in _TERSE_ANSWERS:
        new["say"] = _TERSE_ANSWERS[key]
    return [new]


def _rambling_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = (
        "Hi there! So my neighbour came round yesterday and we got "
        "talking about phones for ages, and it reminded me — "
        f"{_lower_first(turn['say'])} "
        "Honestly, I've been meaning to ask all week.")
    return [new]


def _rambling_payload(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = (f"Oh — my details, I have them here: {turn['say']}, "
                  "written down from my last letter.")
    return [new]


def _rambling_answer(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = f"OK so, {_sentence(turn['say'])} Sorry, I do talk a lot."
    return [new]


def _rambling_confirm(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = "Yes, go ahead — thank you, you've been so patient."
    return [new]


def _non_native_text(text: str) -> str:
    out = text
    out = out.replace("I'd like to", "I want to")
    out = out.replace("I would like to", "I want to")
    out = out.replace("I'm travelling", "I travel")
    out = out.replace("it's", "it is")
    out = out.replace("I've", "I have")
    out = re.sub(r"\bwhy is\b", "why", out)
    out = re.sub(r"\bwhat will\b", "what", out)
    out = re.sub(r"\bkeeps dropping\b", "is keep dropping", out)
    out = re.sub(r"(?<!make )\b(the|a|an)\b ?", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _non_native_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = _cap_first(_non_native_text(turn["say"]))
    return [new]


def _non_native_payload(turn, flow):
    acct, pc = _payload_parts(turn["say"])
    new = copy.deepcopy(turn)
    new["say"] = f"account {acct}, postcode {pc}"
    return [new]


def _non_native_answer(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = _cap_first(_non_native_text(turn["say"]))
    return [new]


def _non_native_confirm(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = "yes, please do it"
    return [new]


def _changes_mind_after_opener(turn, flow):
    # The wandering off: a side question while identity is being
    # collected, then back. The turns assert nothing but that the
    # conversation is not escalated — the flow's own later turns
    # and final state grade whether the detour cost anything.
    return [
        {"role": "detour",
         "say": "Sorry — wait, before that: what are your opening "
                "hours on a Saturday?",
         "expect": {"escalated": False}},
        {"role": "detour",
         "say": "OK, thanks. Anyway — back to what I asked about.",
         "expect": {"escalated": False}},
    ]


def _changes_mind_confirm(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = "Hmm, actually — no wait. Yes, go ahead."
    return [new]


_WALL_BEFORE = (
    "I have been meaning to write this down properly all week, "
    "because on the phone I always forget half of what I wanted "
    "to say. The kettle has been on twice already and the dog "
    "keeps asking to go out, so let me get this down in one go.")
_WALL_AFTER = (
    "Anyway, that is the whole story as best I can tell it. The "
    "garden has gone mad with weeds this month and I still have "
    "not sorted the spare room, but this phone thing matters "
    "more, so I am putting it first for once.")


def _wall_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = f"{_WALL_BEFORE} {_cap_first(turn['say'])} {_WALL_AFTER}"
    return [new]


def _wall_payload(turn, flow):
    acct, pc = _payload_parts(turn["say"])
    new = copy.deepcopy(turn)
    new["say"] = (f"Let me find the letter it came on. My account "
                  f"number is {acct} and the postcode is {pc}. I "
                  "keep all of these in a drawer by the door, so I "
                  "do not lose them.")
    return [new]


def _wall_answer(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = (f"Sorry, another long message from me. "
                  f"{_sentence(_cap_first(turn['say']))} That is "
                  "everything I can think of to tell you, I promise.")
    return [new]


def _wall_confirm(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = ("I have read all of that twice and thought about "
                  "it properly, and after all of that my answer is "
                  "— yes, go ahead please.")
    return [new]


def _elderly_opener(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = (f"Hello dear. {_cap_first(turn['say'])} I do hope "
                  "you can help me, I am not very quick with these "
                  "phones.")
    return [new]


def _elderly_payload(turn, flow):
    acct, pc = _payload_parts(turn["say"])
    new = copy.deepcopy(turn)
    new["say"] = (f"Let me read it off my letter: my account is "
                  f"{acct} and the postcode is {pc}. I hope I have "
                  "read that right.")
    return [new]


def _elderly_answer(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = (f"Let me see… {_sentence(_cap_first(turn['say']))} "
                  "I think that is right.")
    return [new]


def _elderly_confirm(turn, flow):
    new = copy.deepcopy(turn)
    new["say"] = "Yes please, dear."
    return [new]


PERSONAS: dict[str, dict] = {
    "confused_repetitive": {
        "opener": _confused_opener,
        "after_opener": _confused_after_opener,
        "payload": _confused_payload,
        "answer": _confused_answer,
        "confirm": _confused_confirm,
        "confirm_guard": _confused_confirm_guard,
    },
    "terse": {
        "opener": _terse_opener,
        "answer": _terse_answer,
    },
    "rambling_multi_ask": {
        "opener": _rambling_opener,
        "payload": _rambling_payload,
        "answer": _rambling_answer,
        "confirm": _rambling_confirm,
        "confirm_guard": _rambling_confirm,
    },
    "non_native": {
        "opener": _non_native_opener,
        "payload": _non_native_payload,
        "answer": _non_native_answer,
        "confirm": _non_native_confirm,
        "confirm_guard": _non_native_confirm,
    },
    "changes_mind": {
        "after_opener": _changes_mind_after_opener,
        "confirm": _changes_mind_confirm,
        "confirm_guard": _changes_mind_confirm,
    },
    "wall_of_text": {
        "opener": _wall_opener,
        "payload": _wall_payload,
        "answer": _wall_answer,
        "confirm": _wall_confirm,
        "confirm_guard": _wall_confirm,
    },
    "elderly_careful": {
        "opener": _elderly_opener,
        "payload": _elderly_payload,
        "answer": _elderly_answer,
        "confirm": _elderly_confirm,
        "confirm_guard": _elderly_confirm,
    },
}


# ---------------------------------------------------------------------------
# The matrix itself
# ---------------------------------------------------------------------------

def build_matrix_scenarios() -> list[dict]:
    """Every persona across every flow, as runner scenarios."""
    scenarios = []
    for flow in FLOWS:
        for persona_id, persona in PERSONAS.items():
            turns: list[dict] = []
            answer_ordinal = -1
            flow_ctx = dict(flow)
            for turn in flow["turns"]:
                role = turn["role"]
                if role == "answer":
                    answer_ordinal += 1
                    flow_ctx["_answer_ordinal"] = answer_ordinal
                transform = persona.get(role)
                produced = transform(turn, flow_ctx) if transform else _keep(turn)
                turns.extend(produced)
                if role == "opener" and persona.get("after_opener"):
                    turns.extend(persona["after_opener"](turn, flow_ctx))
            scenarios.append({
                "id": f"mx-{flow['id']}-{persona_id}",
                "rubric": flow["rubric"],
                "persona": persona_id,
                "matrix": True,
                "account": flow["account"],
                "zt": flow["zt"],
                "turns": turns,
                "final": copy.deepcopy(flow["final"]),
            })
    return scenarios
