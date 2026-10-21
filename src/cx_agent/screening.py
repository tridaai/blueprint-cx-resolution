"""Intake screening of customer text, in code, before any model
sees it. Two jobs:

- **Prompt injection**: instructions addressed to the agent ("ignore
  your instructions", "reveal the system prompt", fake authority
  claims) are flagged, the turn is answered as ordinary customer
  service, and the attempt is audit-logged. The flag never reaches
  the drafting prompt as an instruction.
- **Escalation signals**: threats, legal action, fraud reporting and
  vulnerability markers route the conversation to a human
  immediately (see escalation.py). These are read here so every
  node downstream sees the same verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    r"ignore (all )?(your |the )?(previous |prior )?instructions",
    r"disregard (your |the )?(instructions|rules|policy)",
    r"system prompt",
    r"you are now",
    r"pretend (to be|you are)",
    r"jailbreak",
    r"developer mode",
    r"reveal (your|the) (prompt|instructions)",
    r"as (an? )?(ai|language model|administrator|admin|manager|supervisor),? (i (order|instruct|demand)|you (must|will))",
    r"override (the )?(policy|rules|verification)",
    r"skip (the )?(verification|identity check|security)",
]

_THREAT_PATTERNS = [
    r"\b(kill|hurt|harm|bomb|shoot)\b.*\b(you|myself|himself|herself|someone)\b",
    r"\bi('ll| will) (kill|hurt|harm)\b",
    r"\bthreaten(ing)? (to )?(sue|report|call the police)\b",
]

_LEGAL_PATTERNS = [
    r"\b(my )?(lawyer|attorney|solicitor)\b",
    r"\b(sue|lawsuit|small claims|court|legal action|ombudsman|regulator|attorney general)\b",
]

_FRAUD_PATTERNS = [
    r"\b(fraud|fraudulent|unauthori[sz]ed (charge|transaction|payment)|identity theft|not my (charge|bill|account)|someone else('s)? (card|account))\b",
]

_VULNERABLE_PATTERNS = [
    r"\b(bereave|funeral|hospital|terminal|disabled|disability|can't afford (food|rent|medicine)|homeless)\b",
]

# The governed explicit-statement classes — the ONLY words that
# set the persistent vulnerability flag on the account record.
# Each pattern is a circumstance the customer states plainly, in
# the first person: a bereavement, a serious illness, money
# hardship. Anything weaker — tone, age ("I'm old"), apologies
# for slow typing, a sad turn of phrase, a circumstance mentioned
# about someone else — matches nothing here, however the broad
# screen above reads the conversation itself. The distinction is
# deliberate: escalation protects this conversation on a mention;
# the flag is a lasting record about a person, and it is set only
# from their own plain statement.
_VULNERABLE_STATEMENTS: dict[str, list[str]] = {
    "bereavement": [
        r"\bmy (husband|wife|partner|mother|father|mum|dad|son|daughter|brother|sister) (has )?(died|passed away)\b",
        r"\bi lost my (husband|wife|partner|mother|father|mum|dad|son|daughter|brother|sister)\b",
        r"\bi('ve| have) (just )?been bereaved\b",
        r"\bi('m| am) (recently )?bereaved\b",
        r"\bthere('s| has) been a death in (the|my) family\b",
    ],
    "serious_illness": [
        r"\bi('ve| have) been diagnosed with\b",
        r"\bi have (cancer|a terminal illness|a serious illness)\b",
        r"\bi('ve| have) got cancer\b",
        r"\bi('m| am) terminally ill\b",
        r"\bi('m| am) seriously ill\b",
        r"\bi('m| am) in (hospital|hospice)\b",
        r"\bi have a (serious|terminal) (illness|condition)\b",
    ],
    "financial_hardship": [
        r"\bi can'?t afford (food|rent|medicine|medication|to eat|my bills|the bills|this bill|the bill)\b",
        r"\bi can'?t afford to (eat|heat my home|heat the house)\b",
        r"\bi('m| am) homeless\b",
        r"\bwe('ve| have) lost our home\b",
        r"\bi('m| am) in (serious )?debt\b",
        r"\b(i|we)('m| am|'re| are) facing financial hardship\b",
    ],
}

_HUMAN_PATTERNS = [
    r"\b(human|person|agent|representative|rep|someone real|real person|manager|supervisor|operator)\b",
]
_HUMAN_WORDS = {"human", "person", "agent", "representative", "rep",
                "someone", "manager", "supervisor", "operator"}
_BARE_FILLER = {"a", "an", "the", "real", "live", "actual", "proper",
                "please", "now", "just", "i", "want", "need", "need a",
                "right", "away", "asap", "ok", "okay"}
_HUMAN_FRAMING = [
    r"\b(speak|talk|chat) (to|with)\b",
    r"\bconnect me\b",
    r"\btransfer me\b",
    r"\bget me\b",
    r"\blet me (speak|talk)\b",
    r"\bgive me a\b",
]


@dataclass
class Screening:
    injection: bool = False
    threat: bool = False
    legal: bool = False
    fraud: bool = False
    vulnerable: bool = False
    # The governed explicit-statement class this message carries
    # (bereavement | serious_illness | financial_hardship), or "".
    # This — and only this — sets the persistent account flag.
    vulnerable_statement: str = ""
    asks_for_human: bool = False
    matched: list[str] = field(default_factory=list)


def _hits(patterns: list[str], text: str, label: str, matched: list[str]) -> bool:
    found = False
    for pat in patterns:
        if re.search(pat, text):
            found = True
            matched.append(f"{label}:{pat[:24]}")
    return found


def screen(text: str) -> Screening:
    lowered = f" {text.lower()} "
    out = Screening()
    out.injection = _hits(_INJECTION_PATTERNS, lowered, "injection", out.matched)
    out.threat = _hits(_THREAT_PATTERNS, lowered, "threat", out.matched)
    out.legal = _hits(_LEGAL_PATTERNS, lowered, "legal", out.matched)
    out.fraud = _hits(_FRAUD_PATTERNS, lowered, "fraud", out.matched)
    out.vulnerable = _hits(_VULNERABLE_PATTERNS, lowered, "vulnerable", out.matched)
    for statement_class, patterns in _VULNERABLE_STATEMENTS.items():
        if any(re.search(p, lowered) for p in patterns):
            out.vulnerable_statement = statement_class
            out.matched.append(f"vulnerable_statement:{statement_class}")
            break
    asks = _hits(_HUMAN_PATTERNS, lowered, "human", out.matched)
    framed = any(re.search(p, lowered) for p in _HUMAN_FRAMING)
    # A bare ask ("human now", "agent please", "a real person")
    # carries no framing verb: the message is the ask when, filler
    # aside, a person is all it names.
    words = set(re.findall(r"[a-z']+", lowered))
    bare = bool(words) and words <= (_HUMAN_WORDS | _BARE_FILLER)
    out.asks_for_human = bool(asks and (framed or bare))
    return out
