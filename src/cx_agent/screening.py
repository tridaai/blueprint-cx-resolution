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

_HUMAN_PATTERNS = [
    r"\b(human|person|agent|representative|rep|someone real|real person|manager|supervisor|operator)\b",
]
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
    asks = _hits(_HUMAN_PATTERNS, lowered, "human", out.matched)
    framed = any(re.search(p, lowered) for p in _HUMAN_FRAMING)
    out.asks_for_human = bool(asks and framed)
    return out
