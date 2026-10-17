"""Environment-driven configuration.

One decision lives here and nowhere else: which store profile is
running (`store_profile`). `DATABASE_URL` set means PostgreSQL +
pgvector (production); otherwise a single SQLite file is the local
default. The LLM backend resolves the same way — the first configured
provider wins under `auto`, and with no provider configured the
labelled offline fallback drafts from structured findings so the
whole system still runs (and the eval gate still means something)
with no key at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def store_profile() -> tuple[str, str]:
    """(kind, target): ("postgres", dsn) or ("sqlite", path)."""
    dsn = _env("DATABASE_URL")
    if dsn:
        return "postgres", dsn
    return "sqlite", _env("SQLITE_PATH", "data/cx.db")


@dataclass(frozen=True)
class Policy:
    """The governed thresholds. Every number a rule depends on is
    named here (and overridable by env), never buried in a prompt.

    The credit rules form two named tiers. The EVIDENCED tier pays
    back what the bill itself proves wrong, in full, up to its cap;
    above the cap the dispute case escalates with its record intact.
    The GOODWILL tier pays for a poor experience when the bill is
    right: a smaller per-offer cap, and a rolling cap across the
    account's trailing credit history that the policy reads in code
    before any offer is made."""

    # GOODWILL tier, per offer: a credit the agent may propose on
    # its own authority when the customer is unhappy but the bill
    # is correct.
    goodwill_credit_cap: Decimal = Decimal("25.00")
    # GOODWILL tier, rolling: the most goodwill an account may
    # receive inside the trailing credit-history window. Goodwill
    # beyond it is refused in code, with the reason recorded — the
    # evidenced tier is never touched by goodwill history.
    goodwill_rolling_cap: Decimal = Decimal("50.00")
    # The trailing window the credit history is read over, in days.
    credit_history_days: int = 90
    # EVIDENCED tier: a credit backed by evidence on the bill itself
    # (duplicate charge, charge after cancellation, wrong rate).
    # Above this the case escalates to the billing team instead of
    # being credited.
    dispute_credit_cap: Decimal = Decimal("50.00")
    # A bill whose total exceeds the account's trailing average by
    # at least this percentage is a bill-shock bill: its explanation
    # leads with the computed driver, not the line-by-line.
    bill_shock_threshold_pct: Decimal = Decimal("25")
    # Roaming fair-use ceiling per trip, in GB, from the roaming
    # policy corpus. Past it the agent must warn before selling more.
    roaming_fair_use_gb: int = 25
    # Roaming bill-shock threshold: when the projected
    # pay-as-you-go cost of a trip passes this, the projection and
    # the pass comparison lead the conversation — no chargeable
    # roaming product is proposed until the customer has seen
    # them (POL-ROAM-03).
    roaming_bill_shock_amount: Decimal = Decimal("50.00")
    # Unproductive turns in a row before the conversation escalates.
    max_no_progress_turns: int = 3
    # One-time-code attempts before verification locks.
    otp_max_attempts: int = 3
    # Asks-for-a-human tolerated before escalation: the first ask gets
    # one genuine solve attempt, the second escalates.
    human_requests_before_escalation: int = 2
    # The plan-change reversal window, in days past the day the
    # change was made. Zero — a change can be reversed the same day
    # it was made, as a compensating pair of records; after that
    # the request goes to the offers team with the change record.
    plan_change_reversal_days: int = 0
    # Payment arrangements (POL-PAY-01): the smallest balance an
    # arrangement may be written over, the most installments it
    # may run to, the smallest installment it may contain, and the
    # calendar the schedule is laid on — the first date lands
    # within this many days, the rest follow monthly.
    arrangement_min_balance: Decimal = Decimal("100.00")
    arrangement_max_installments: int = 3
    arrangement_min_installment: Decimal = Decimal("25.00")
    arrangement_first_due_days: int = 14
    arrangement_interval_days: int = 30
    # Complaints (POL-CARE-02): the substantive response is due
    # this many days after the complaint is recorded; the SLA
    # watch calls a still-open complaint breached once that date
    # passes, and near-breach within this many days of it.
    complaint_response_days: int = 5
    complaint_near_breach_days: int = 1
    # Unhandled-intent mining: a cluster becomes a candidate for
    # the new-intent report at this many conversations, and two
    # utterances cluster together at or above this cosine over the
    # repo's own embedder (measured: same-need phrasings score
    # ~0.39+, unrelated unhandled asks ~0.22 or less).
    mining_min_cluster: int = 3
    mining_similarity: float = 0.30
    # Annotation queue sampling: every escalated or unresolved
    # conversation is queued for human review; resolved ones are
    # sampled deterministically — one in this many, by a stable
    # hash of the conversation id — so the queue also audits the
    # successes without drowning the reviewers. 1 queues them all.
    annotation_sample_modulus: int = 5

    @classmethod
    def from_env(cls) -> "Policy":
        def dec(name: str, default: str) -> Decimal:
            return Decimal(_env(name, default))

        def num(name: str, default: int) -> int:
            raw = _env(name)
            return int(raw) if raw else default

        def flt(name: str, default: float) -> float:
            raw = _env(name)
            return float(raw) if raw else default

        base = cls()
        return cls(
            goodwill_credit_cap=dec("CX_GOODWILL_CREDIT_CAP", str(base.goodwill_credit_cap)),
            goodwill_rolling_cap=dec("CX_GOODWILL_ROLLING_CAP", str(base.goodwill_rolling_cap)),
            credit_history_days=num("CX_CREDIT_HISTORY_DAYS", base.credit_history_days),
            dispute_credit_cap=dec("CX_DISPUTE_CREDIT_CAP", str(base.dispute_credit_cap)),
            bill_shock_threshold_pct=dec(
                "CX_BILL_SHOCK_THRESHOLD_PCT", str(base.bill_shock_threshold_pct)),
            roaming_fair_use_gb=num("CX_ROAMING_FAIR_USE_GB", base.roaming_fair_use_gb),
            roaming_bill_shock_amount=dec(
                "CX_ROAMING_BILL_SHOCK_AMOUNT", str(base.roaming_bill_shock_amount)),
            max_no_progress_turns=num("CX_MAX_NO_PROGRESS_TURNS", base.max_no_progress_turns),
            otp_max_attempts=num("CX_OTP_MAX_ATTEMPTS", base.otp_max_attempts),
            human_requests_before_escalation=num(
                "CX_HUMAN_REQUESTS_BEFORE_ESCALATION", base.human_requests_before_escalation
            ),
            plan_change_reversal_days=num(
                "CX_PLAN_CHANGE_REVERSAL_DAYS", base.plan_change_reversal_days
            ),
            arrangement_min_balance=dec(
                "CX_ARRANGEMENT_MIN_BALANCE", str(base.arrangement_min_balance)),
            arrangement_max_installments=num(
                "CX_ARRANGEMENT_MAX_INSTALLMENTS", base.arrangement_max_installments),
            arrangement_min_installment=dec(
                "CX_ARRANGEMENT_MIN_INSTALLMENT", str(base.arrangement_min_installment)),
            arrangement_first_due_days=num(
                "CX_ARRANGEMENT_FIRST_DUE_DAYS", base.arrangement_first_due_days),
            arrangement_interval_days=num(
                "CX_ARRANGEMENT_INTERVAL_DAYS", base.arrangement_interval_days),
            complaint_response_days=num(
                "CX_COMPLAINT_RESPONSE_DAYS", base.complaint_response_days),
            complaint_near_breach_days=num(
                "CX_COMPLAINT_NEAR_BREACH_DAYS", base.complaint_near_breach_days),
            mining_min_cluster=num(
                "CX_MINING_MIN_CLUSTER", base.mining_min_cluster),
            mining_similarity=flt(
                "CX_MINING_SIMILARITY", base.mining_similarity),
            annotation_sample_modulus=num(
                "CX_ANNOTATION_SAMPLE_MODULUS", base.annotation_sample_modulus),
        )


@dataclass(frozen=True)
class LLMConfig:
    backend: str  # openai | anthropic | ollama | offline
    model: str = ""
    api_key: str = ""
    base_url: str = ""

    @property
    def is_live(self) -> bool:
        return self.backend != "offline"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        want = _env("LLM_BACKEND", "auto").lower()
        openai_key, openai_url = _env("OPENAI_API_KEY"), _env("OPENAI_BASE_URL")
        anthropic_key = _env("ANTHROPIC_API_KEY")
        ollama_url = _env("OLLAMA_BASE_URL")

        def openai() -> "LLMConfig":
            return cls("openai", _env("OPENAI_MODEL", "gpt-4o-mini"), openai_key,
                       openai_url or "https://api.openai.com/v1")

        def anthropic() -> "LLMConfig":
            return cls("anthropic", _env("ANTHROPIC_MODEL", "claude-sonnet-4-5"), anthropic_key,
                       "https://api.anthropic.com")

        def ollama() -> "LLMConfig":
            return cls("ollama", _env("OLLAMA_MODEL", "llama3.1"), "", ollama_url)

        if want == "openai" and (openai_key or openai_url):
            return openai()
        if want == "anthropic" and anthropic_key:
            return anthropic()
        if want == "ollama" and ollama_url:
            return ollama()
        if want == "auto":
            if openai_key or openai_url:
                return openai()
            if anthropic_key:
                return anthropic()
            if ollama_url:
                return ollama()
        return cls("offline")


@dataclass(frozen=True)
class S3Config:
    bucket: str = ""
    endpoint_url: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"

    @property
    def configured(self) -> bool:
        return bool(self.bucket and self.endpoint_url)

    @classmethod
    def from_env(cls) -> "S3Config":
        return cls(
            bucket=_env("S3_BUCKET"),
            endpoint_url=_env("S3_ENDPOINT_URL"),
            access_key=_env("S3_ACCESS_KEY_ID"),
            secret_key=_env("S3_SECRET_ACCESS_KEY"),
            region=_env("S3_REGION", "us-east-1"),
        )
