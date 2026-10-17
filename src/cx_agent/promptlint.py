"""The prompt lint (round 13): deterministic checks over the
registry's ACTIVE entries, run as a test gate, from the CLI
(`cx-agent prompt-lint`), and inside every activation act — a
version whose text breaks the lint cannot become active
(prompts._lint_gate).

Four rules, all in code, none of them a model's opinion:

- **figure_consistency** — a governed figure stated in a text
  must equal the Policy value it names. The mapping is declared
  below (entry, pattern, policy field): the policy corpus states
  the caps, thresholds and windows the desks enforce, and the
  second forms restate two of them; if the words and the code
  ever disagree, one of them is lying to the customer. A check
  whose pattern no longer matches at all is a finding too — the
  figure the entry is declared to state has gone missing.
  Figures in worked examples are NOT mapped: the second forms'
  sample bills and sample trips use round invented numbers,
  labelled as examples in the text itself, and they state no
  policy. The mapping covers exactly the figures that ARE
  policy.
- **banned_promise** — promise phrases the agent may never make
  (guarantees, unbounded or instant refunds) fail the lint
  wherever they appear, in any surface. The compose prompt's own
  instruction — "do not promise anything the facts do not
  state" — is not a promise and is not on the list.
- **side_channel** — no entry outside the compose surface may
  instruct contacting the customer directly or sending them
  anything. Specialists return findings; only the supervisor
  composes customer text, exactly once. (The compose entries are
  exempt by definition: they ARE the supervisor's instructions.)
- **composition_discipline** — a compose entry must carry the
  verbatim-figures instruction and the Sources instruction, the
  two guarantees composition exists to enforce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .prompts import PromptRegistry

# (prompt_id, pattern with one capture group, Policy field).
# The patterns are written against the shipped texts; if a text
# is reworded, its pattern is re-declared here in the same change.
FIGURE_CHECKS: list[tuple[str, str, str]] = [
    ("policy.POL-BILL-01", r"capped at \$(\d+(?:\.\d+)?)", "dispute_credit_cap"),
    ("policy.POL-BILL-02", r"goodwill credit of up to \$(\d+(?:\.\d+)?)",
     "goodwill_credit_cap"),
    ("policy.POL-BILL-02", r"no more than \$(\d+(?:\.\d+)?) of goodwill",
     "goodwill_rolling_cap"),
    ("policy.POL-BILL-02", r"in any (\d+)-day period", "credit_history_days"),
    ("policy.POL-ROAM-02", r"fair use is (\d+(?:\.\d+)?) GB",
     "roaming_fair_use_gb"),
    ("policy.POL-ROAM-03", r"pay-as-you-go cost passes \$(\d+(?:\.\d+)?)",
     "roaming_bill_shock_amount"),
    ("policy.POL-ID-01", r"(\w+) failed code attempts", "otp_max_attempts"),
    ("policy.POL-PAY-01", r"balance of \$(\d+(?:\.\d+)?) or more",
     "arrangement_min_balance"),
    ("policy.POL-PAY-01", r"into up to (\w+) monthly instalments",
     "arrangement_max_installments"),
    ("policy.POL-PAY-01", r"each of at least \$(\d+(?:\.\d+)?)",
     "arrangement_min_installment"),
    ("policy.POL-PAY-01", r"falls due within (\d+) days",
     "arrangement_first_due_days"),
    ("policy.POL-CARE-02", r"responds in substance within (\w+) days",
     "complaint_response_days"),
    ("explain.POL-ID-01", r"(\w+) wrong codes", "otp_max_attempts"),
    ("explain.POL-PAY-01", r"over (\w+) instalments",
     "arrangement_max_installments"),
]

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

BANNED_PROMISES = [
    "guarantee",
    "we promise",
    "i promise",
    "no questions asked",
    "instant refund",
    "always refund",
    "full refund",
    "definitely receive",
]

SIDE_CHANNEL_PATTERNS = [
    r"\b(?:email|text|call|phone|message)\s+the customer\b",
    r"\bsend\s+(?:the customer|them|him|her)\s+(?:a|an|the)\s+"
    r"(?:email|text|message|sms|letter)\b",
    r"\bcontact the customer\s+(?:directly|yourself|personally)\b",
    r"\breach out to the customer\b",
    r"\bnotify the customer directly\b",
]

# What a compose-surface entry must carry (case-insensitive):
# the verbatim-figures instruction and the Sources instruction.
COMPOSE_MARKERS = ["verbatim", "sources"]


@dataclass
class LintFinding:
    rule: str
    prompt_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.prompt_id}: {self.message}"


class PromptLintError(RuntimeError):
    """An activation was refused by the lint gate. Carries the
    findings; nothing was written and nothing was recorded."""

    def __init__(self, findings: list[LintFinding]) -> None:
        self.findings = findings
        super().__init__(
            "prompt lint refused the activation: "
            + "; ".join(str(f) for f in findings))


def _number(token: str) -> Decimal | None:
    token = token.strip().lower()
    if token in _WORD_NUMBERS:
        return Decimal(_WORD_NUMBERS[token])
    try:
        return Decimal(token)
    except Exception:
        return None


def lint_entries(entries, policy) -> list[LintFinding]:
    """Lint one active set (at most one entry per prompt id, as
    the registry resolves it) against the policy. Pure: the same
    entries and policy always give the same findings."""
    findings: list[LintFinding] = []
    by_id: dict[str, object] = {}
    for entry in entries:
        if entry.prompt_id in by_id:
            findings.append(LintFinding(
                "duplicate_entry", entry.prompt_id,
                "the active set carries this prompt twice"))
        by_id[entry.prompt_id] = entry

    for prompt_id, pattern, field in FIGURE_CHECKS:
        entry = by_id.get(prompt_id)
        if entry is None:
            continue
        expected = Decimal(str(getattr(policy, field)))
        match = re.search(pattern, entry.text)
        if match is None:
            findings.append(LintFinding(
                "figure_consistency", prompt_id,
                f"no longer states the governed figure for {field} "
                f"(pattern {pattern!r} does not match) — the text "
                f"drifted from the declared check"))
            continue
        stated = _number(match.group(1))
        if stated is None:
            findings.append(LintFinding(
                "figure_consistency", prompt_id,
                f"states an unreadable figure ({match.group(1)!r}) "
                f"for {field}"))
        elif stated != expected:
            findings.append(LintFinding(
                "figure_consistency", prompt_id,
                f"states {match.group(1)} for {field}, but the "
                f"policy value is {expected} — the words and the "
                f"code disagree"))

    for entry in entries:
        lowered = entry.text.lower()
        for phrase in BANNED_PROMISES:
            if phrase in lowered:
                findings.append(LintFinding(
                    "banned_promise", entry.prompt_id,
                    f"contains the banned promise phrase {phrase!r} "
                    f"— the agent may never promise this"))
        if entry.surface != "compose":
            for pattern in SIDE_CHANNEL_PATTERNS:
                if re.search(pattern, lowered):
                    findings.append(LintFinding(
                        "side_channel", entry.prompt_id,
                        f"instructs contacting the customer outside "
                        f"the supervisor's composition (pattern "
                        f"{pattern!r}) — specialists return findings; "
                        f"only the supervisor writes to the customer"))
                    break
        if entry.surface == "compose":
            for marker in COMPOSE_MARKERS:
                if marker not in lowered:
                    findings.append(LintFinding(
                        "composition_discipline", entry.prompt_id,
                        f"a compose entry must carry the {marker} "
                        f"instruction — composition exists to enforce "
                        f"it"))
    return findings


def lint_store(store, policy) -> list[LintFinding]:
    """Lint the active set of the registry in a store."""
    return lint_entries(PromptRegistry(store).active_entries(), policy)
