"""Deterministic routing and reply classification.

The router is a weighted phrase table in code — auditable, testable,
and identical offline and live. A live model may offer a hint when
the table is torn between two intents (the supervisor asks; the
table still breaks the tie it can). Nothing here is a prompt.

The same module classifies the two short replies the conversation
protocol depends on: confirmation answers to a proposed action, and
step outcomes inside a troubleshooting procedure. Both are small
closed vocabularies on purpose — a "yes" must never be a guess.

`analyze` is the router read as evidence: the matched signals, the
leaders, the runner-up, and two honesty flags — `ambiguous` (two
desks tie on a real signal; the supervisor asks rather than
guesses) and `low_confidence` (no clear signal; the care desk, not
a transactional desk, takes the turn).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
        ("dropped calls", 3), ("voicemail", 2), ("broadband", 4),
        ("keeps dropping", 4), ("internet keeps dropping", 4),
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


# ---------------------------------------------------------------------------
# Route analysis — the router shows its working
# ---------------------------------------------------------------------------

# A tie is only worth a clarifying question when both readings rest
# on a real signal. Two stray weight-1 words pointing different ways
# are separable by the documented tie order; two weight-4 phrases
# are two genuine needs in one message.
CLARIFY_MIN_SCORE = 3

# A turn whose best score is one weak phrase ("fee", "trip") has no
# clear signal: it belongs with the care desk, never with a
# transactional desk on a guess.
LOW_CONFIDENCE_MAX_SCORE = 1

# How each desk reads in plain words — the supervisor's clarifying
# question offers these, not desk names.
READINGS: dict[Intent, str] = {
    Intent.BILLING: "your bill",
    Intent.ROAMING: "using your phone abroad",
    Intent.OFFERS: "changing your plan",
    Intent.TECHNICAL: "a fault with your service",
    Intent.ACCOUNT: "a payment, or the card on your account",
    Intent.CARE: "something else",
}

# The words people actually use to name a desk when they answer a
# clarifying question in their own way ("the signal one").
_ANSWER_WORDS: dict[Intent, list[str]] = {
    Intent.BILLING: ["bill", "charge", "invoice", "refund"],
    Intent.ROAMING: ["roam", "abroad", "travel", "trip", "holiday"],
    Intent.OFFERS: ["plan", "offer", "upgrade", "deal"],
    Intent.TECHNICAL: ["signal", "fault", "service", "internet",
                        "connection", "phone"],
    Intent.ACCOUNT: ["pay", "card", "payment", "debit"],
    Intent.CARE: [],
}


def match_signals(text: str) -> dict[Intent, list[str]]:
    """The phrases that matched, per intent — the evidence behind
    the scores, kept so a routing decision can be inspected."""
    lowered = f" {text.lower()} "
    matched: dict[Intent, list[str]] = {}
    for intent, phrases in _PHRASES.items():
        hits = [phrase for phrase, _weight in phrases if phrase in lowered]
        if hits:
            matched[intent] = hits
    return matched


@dataclass
class RouteAnalysis:
    scores: dict[Intent, int] = field(default_factory=dict)
    signals: dict[Intent, list[str]] = field(default_factory=dict)
    leaders: list[Intent] = field(default_factory=list)  # tied at the top
    runner_up: Intent | None = None  # best of the rest, if any scored
    top_score: int = 0
    # Exactly two desks tie on a real signal and neither is the
    # care catch-all: the router cannot separate them.
    ambiguous: bool = False
    # No clear signal at all: at most one weak phrase matched.
    low_confidence: bool = False


def analyze(text: str) -> RouteAnalysis:
    """Read a message the way the supervisor needs it read: what
    matched, who leads, who is runner-up, and whether the table can
    honestly separate the top two readings."""
    scores = score_intents(text)
    out = RouteAnalysis(scores=scores, signals=match_signals(text))
    if not scores:
        out.low_confidence = True
        return out
    out.top_score = max(scores.values())
    out.leaders = [i for i, s in scores.items() if s == out.top_score]
    rest = [i for i, s in scores.items() if s < out.top_score]
    if rest:
        out.runner_up = max(rest, key=lambda i: scores[i])
    out.low_confidence = out.top_score <= LOW_CONFIDENCE_MAX_SCORE
    out.ambiguous = (
        len(out.leaders) == 2
        and out.top_score >= CLARIFY_MIN_SCORE
        and Intent.CARE not in out.leaders
    )
    return out


def resolve_clarification(text: str, options: list[Intent]) -> Intent | None:
    """Did this reply pick one of the two offered readings? Scores
    first (counted only across the offered pair), then the words
    people name desks by. None means the reply settled nothing and
    the router must commit without asking again."""
    scores = score_intents(text)
    offered = {intent: scores.get(intent, 0) for intent in options}
    best = max(offered.values())
    if best > 0:
        winners = [i for i, s in offered.items() if s == best]
        if len(winners) == 1:
            return winners[0]
    lowered = text.lower()
    hits = [intent for intent in options
            if any(re.search(rf"\b{w}\b", lowered)
                   for w in _ANSWER_WORDS.get(intent, []))]
    if len(hits) == 1:
        return hits[0]
    return None


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


_SKIP_ATTEMPT = [
    r"\bskip\b", r"\bjust fix it\b", r"\bdon'?t want to\b",
    r"\bwon'?t do\b", r"\bnot doing that\b", r"\brefuse to\b",
    r"\bno way\b", r"\bforget that step\b",
]


def classify_skip_attempt(text: str) -> bool:
    """Is the customer trying to get out of the current step —
    'skip that', 'just fix it', 'I won't do that'? Only meaningful
    at a mandatory safety step, where the walker refuses in code;
    anywhere else the words are just conversation."""
    lowered = text.lower().strip()
    return any(re.search(p, lowered) for p in _SKIP_ATTEMPT)
