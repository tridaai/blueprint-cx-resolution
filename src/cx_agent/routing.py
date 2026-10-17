"""Deterministic routing and reply classification.

The router is a weighted phrase table in code — auditable, testable,
and identical offline and live. A live model may offer a hint when
the table is torn between two intents (the supervisor asks; the
table still breaks the tie it can). Nothing here is a prompt.

The same module classifies the two short replies the conversation
protocol depends on: confirmation answers to a proposed action, and
step outcomes inside a troubleshooting procedure. Both are small
closed vocabularies on purpose — a "yes" must never be a guess.
"""

from __future__ import annotations

import re

from .models import Intent

_PHRASES: dict[Intent, list[tuple[str, int]]] = {
    Intent.BILLING: [
        ("bill", 2), ("charged twice", 4), ("charge", 1), ("invoice", 2),
        ("refund", 3), ("overcharged", 4), ("why is my bill", 4),
        ("bill is high", 3), ("bill went up", 3), ("fee", 1),
        ("statement", 2), ("credit my account", 3), ("dispute", 3),
        ("wrong charge", 4), ("balance", 1), ("how much do i owe", 3),
        ("itemised", 2), ("itemized", 2),
    ],
    Intent.ROAMING: [
        ("roaming", 4), ("roam", 3), ("abroad", 3), ("travel", 2),
        ("travelling", 3), ("traveling", 3), ("holiday", 2),
        ("vacation", 2), ("overseas", 3), ("trip", 1),
        ("data in spain", 4), ("use my phone in", 3),
    ],
    Intent.OFFERS: [
        ("offer", 3), ("upgrade", 3), ("deal", 2), ("discount", 3),
        ("promotion", 3), ("change my plan", 4), ("change plan", 4),
        ("switch plan", 4), ("switch to", 2), ("new plan", 3),
        ("better plan", 3), ("unlimited max", 2),
    ],
    Intent.TECHNICAL: [
        ("no signal", 4), ("no service", 4), ("signal", 2),
        ("slow data", 4), ("internet is slow", 4), ("data is slow", 4),
        ("connection", 2), ("coverage", 2), ("wifi calling", 4),
        ("wi-fi calling", 4), ("not working", 2), ("phone won't", 3),
        ("dropped calls", 3), ("voicemail", 2),
    ],
    Intent.ACCOUNT: [
        ("payment method", 4), ("update my card", 4), ("new card", 3),
        ("change my card", 4), ("direct debit", 3), ("pay my bill", 3),
        ("can't pay", 4), ("cannot pay", 4), ("can not pay", 4),
        ("payment arrangement", 4), ("split my bill", 3),
        ("pay in instalments", 4), ("pay in installments", 4),
        ("behind on", 2), ("missed a payment", 3),
    ],
    Intent.CARE: [
        ("store", 2), ("opening hours", 4), ("open on", 2),
        ("thank you", 1), ("thanks", 1), ("hello", 1), ("hi there", 1),
        ("compliment", 3), ("feedback", 2),
    ],
}


def score_intents(text: str) -> dict[Intent, int]:
    lowered = f" {text.lower()} "
    scores: dict[Intent, int] = {}
    for intent, phrases in _PHRASES.items():
        total = 0
        for phrase, weight in phrases:
            if phrase in lowered:
                total += weight
        if total:
            scores[intent] = total
    return scores


def route(text: str, hint: str | None = None) -> Intent:
    """Best intent for a message. A tie at the top is broken by the
    model hint when one was offered; a clean zero falls to CARE,
    the catch-all, never to a dead end."""
    scores = score_intents(text)
    if not scores:
        return Intent.CARE
    top = max(scores.values())
    leaders = [i for i, s in scores.items() if s == top]
    if len(leaders) == 1:
        return leaders[0]
    if hint:
        for leader in leaders:
            if leader.value == hint:
                return leader
    # Stable, documented tie order: the more specific desks first.
    order = [Intent.BILLING, Intent.ACCOUNT, Intent.TECHNICAL,
             Intent.ROAMING, Intent.OFFERS, Intent.CARE]
    for candidate in order:
        if candidate in leaders:
            return candidate
    return leaders[0]


_CONFIRM = [
    r"^\s*(yes|yeah|yep|yup|ok|okay|sure|confirm|confirmed|agreed)\b",
    r"\bgo ahead\b", r"\bplease do\b", r"\bdo it\b", r"\bsounds good\b",
    r"\bthat works\b",
]
_DECLINE = [
    r"^\s*(no|nope|nah)\b", r"\bdon'?t\b", r"\bdo not\b", r"\bcancel\b",
    r"\bleave it\b", r"\bnever ?mind\b", r"\bstop\b", r"\bnot now\b",
]


def classify_confirmation(text: str) -> str:
    """confirm | decline | other — for a reply to a proposed action."""
    lowered = text.lower().strip()
    if any(re.search(p, lowered) for p in _DECLINE):
        return "decline"
    if any(re.search(p, lowered) for p in _CONFIRM):
        return "confirm"
    return "other"


_STEP_FAIL = [
    r"\bstill\b", r"\bnothing\b", r"\bnot working\b", r"\bno luck\b",
    r"\bsame\b", r"\bdidn'?t\b", r"\bdoesn'?t\b", r"\bwont\b", r"\bwon'?t\b",
    r"^\s*no\b", r"\bno signal\b", r"\bstill slow\b", r"\bnope\b",
]
_STEP_SUCCESS = [
    r"\byes\b", r"\bworked\b", r"\bit'?s back\b", r"\bsignal is back\b",
    r"\bfaster\b", r"\bfixed\b", r"\bconnected\b", r"\bit'?s on\b",
    r"\bdone\b", r"\bworking now\b", r"\bgot signal\b", r"\bimproved\b",
]


def classify_step_outcome(text: str) -> str:
    """success | fail | unclear — for a reply inside a procedure.
    Failure phrases are checked first: 'no, still nothing' is a fail,
    however many hopeful words surround it."""
    lowered = text.lower().strip()
    if any(re.search(p, lowered) for p in _STEP_FAIL):
        return "fail"
    if any(re.search(p, lowered) for p in _STEP_SUCCESS):
        return "success"
    return "unclear"
