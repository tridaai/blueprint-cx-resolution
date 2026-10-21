"""The model boundary.

One job: language. Given the supervisor's structured findings —
facts established by tools, figures computed in code, an action
awaiting confirmation — produce the customer-facing reply. In live
mode a real model does that drafting (OpenAI-compatible endpoints
including NVIDIA NIM, Anthropic, or Ollama, over plain HTTPS), and
may also offer an intent hint when the code router is torn. In
offline mode a deterministic composer assembles the same findings
into the same shape of reply; it is labelled wherever it surfaces
and exists so the system, the tests and the eval gate run with no
key — it is not a substitute for a model on real customer language.

The model never decides: routing, identity, escalation, money and
action gates are code (see supervisor.py, identity.py,
escalation.py, actions.py). If a live call fails, drafting degrades
to the offline composer for that turn and the failure is logged —
the rules never degrade with it.

Every call is accounted (round 18). Each attempt — the compose
call (kind `decision`), the router's tie-break hint (kind
`advisory`), answered, errored or timed out — appends a record:
its tokens with their basis (`measured` where the provider
reported usage, `estimated` where the deterministic estimate
stood in — the offline composer's calls always estimate), its
outcome, and its wall-clock latency. The supervisor drains the
records each turn onto the conversation's audit trail; budgets
and the cost report fold from there (costs.py), so this boundary
keeps no second set of books.

Latency is disciplined the same way. A provider call carries
the governed timeout (`Policy.model_call_timeout_seconds`,
wired onto the live boundary by the service). A timed-out call
is retried ONCE; a second timeout raises ModelTimeoutError out
of the boundary — it does not degrade to the offline composer,
because an answer assembled after a timeout would be a
fabrication — and the supervisor closes the turn to a person
with the reason `model_timeout`.

The prompts themselves are not in this file: the compose and
intent instructions are prompt-registry entries (prompts.py),
resolved at call time at their active versions. A live drafter
with no registry wired, or a registry that cannot serve an
entry, refuses with PromptRegistryError — the boundary never
falls back to a prompt from memory.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .config import LLMConfig
from .prompts import (
    COMPOSE_SYSTEM_ID,
    COMPOSE_USER_ID,
    INTENT_SYSTEM_ID,
    INTENT_USER_ID,
    PromptRegistryError,
    render_template,
)

log = logging.getLogger("cx_agent.llm")


class ModelTimeoutError(Exception):
    """A model call — and its one governed retry — both timed
    out. Raised out of the boundary, never degraded around: the
    supervisor closes the turn to a person (`model_timeout`),
    because a reply composed after a timeout would be made of
    nothing."""


def estimate_tokens(text: str) -> int:
    """The deterministic token estimate: four characters to a
    token, rounded up. Crude on purpose — stable across runs,
    and labelled `estimated` wherever it surfaces, never dressed
    as a measurement."""
    return (len(text) + 3) // 4 if text else 0


def _elapsed_ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover — httpx is a dependency
        return False
    return isinstance(exc, httpx.TimeoutException)


class _CallLedger:
    """The boundary's pending call records: one per attempt,
    drained by the supervisor onto the conversation's audit
    trail each turn. A hand-off, not a store — nothing reads
    the ledger but the drain, and the audit trail is the only
    accounting that survives."""

    def _init_ledger(self) -> None:
        self._pending_calls: list[dict] = []

    def _record_call(self, call_kind: str, *, attempt: int, outcome: str,
                     tokens_in: int, tokens_out: int, basis: str,
                     latency_ms: int, model: str = "") -> None:
        self._pending_calls.append({
            "call_kind": call_kind, "backend": self.backend,
            "model": model, "attempt": attempt, "outcome": outcome,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "tokens": tokens_in + tokens_out,
            "tokens_basis": basis, "latency_ms": latency_ms})

    def drain_calls(self) -> list[dict]:
        pending, self._pending_calls = self._pending_calls, []
        return pending


@dataclass
class DraftContext:
    customer_name: str = ""
    intent: str = "care"
    findings: list[str] = field(default_factory=list)  # customer-ready sentences
    must_include: list[str] = field(default_factory=list)  # verbatim lines
    action_prompt: str = ""  # the confirmation ask, when an action pends
    citations: list[tuple[str, str]] = field(default_factory=list)  # (doc_id, title)
    customer_message: str = ""


def _offline_compose(ctx: DraftContext) -> str:
    parts: list[str] = []
    for text in ctx.findings:
        if text and text not in parts:
            parts.append(text)
    body = " ".join(parts)
    for line in ctx.must_include:
        if line and line not in body:
            parts.append(line)
            body = " ".join(parts)
    if ctx.action_prompt:
        parts.append(ctx.action_prompt)
    if ctx.citations:
        refs = "; ".join(f"{doc_id} ({title})" for doc_id, title in ctx.citations)
        parts.append(f"Sources: {refs}.")
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Vetting a live draft
# ---------------------------------------------------------------------------
#
# The supervisor guarantees what a reply must CONTAIN (the verbatim
# lines, the confirmation ask, the Sources line). Vetting is the
# other half: what a model draft may not ADD. A draft is checked
# against the material it was given — the findings, the verbatim
# lines and the confirmation ask, which is everything the code
# established this turn — and refused when it states a figure that
# material doesn't hold, repeats a number the material doesn't
# carry (a one-time code, a postcode read back wrong), promises an
# act the material doesn't offer, or names a channel or promise of
# its own. A refused draft is replaced by the offline composer's
# reply, made from the same material, so the turn still lands —
# with nothing in it the code did not establish.

_MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{4,}(?!\d)")
_PROMISE_RE = re.compile(
    r"\b(?:i'?ll|i will|we'?ll|we will|i'?m going to|i am going to|"
    r"let me|i'?d like to|i would like to|i can)\s+(?:\w+\s+){0,3}?"
    r"(credit|refund|apply|process|activate|cancel|book|waive|"
    r"reimburse|compensate|offer|remove|reverse|send)\b", re.IGNORECASE)
# Words a draft may only use when the material uses them too.
_FOREIGN_TERMS = ("bank account", "text message", "sms", "within 24 hours",
                  "guarantee", "compensation")
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here'?s|here is)\b[^\n]{0,80}?\b(?:reply|response|draft|"
    r"message)\b[^\n]*?:\s*", re.IGNORECASE)
_SALUTATION_RE = re.compile(
    r"^\s*(?:dear [^\n,]{1,40}|hello there|hi there),?\s*\n", re.IGNORECASE)
_SOURCES_RE = re.compile(r"\s*\bSources?:[^\n]*", re.IGNORECASE)


def _amounts(text: str) -> set[Decimal]:
    out = set()
    for raw in _MONEY_RE.findall(text):
        try:
            out.add(Decimal(raw.replace(",", "")))
        except InvalidOperation:
            continue
    return out


def _tidy_draft(text: str) -> str:
    """Strip what a model wraps around a reply: a "Here's a warm
    reply:" preamble, a letter salutation, enclosing quotes, and any
    Sources line of its own — the supervisor appends the true one
    from the turn's citations."""
    text = _PREAMBLE_RE.sub("", text.strip(), count=1)
    text = _SALUTATION_RE.sub("", text, count=1).strip()
    if len(text) > 1 and text[0] in "\"\u201c" and text[-1] in "\"\u201d":
        text = text[1:-1].strip()
    text = _SOURCES_RE.sub("", text).strip()
    if len(text) > 1 and text[0] in "\"\u201c" and text[-1] in "\"\u201d":
        text = text[1:-1].strip()
    return text


def vet_draft(draft: str, ctx: "DraftContext") -> tuple[str, str]:
    """(tidied draft, refusal reason) — the reason is "" when the
    draft may stand."""
    text = _tidy_draft(draft)
    if not text:
        return text, "empty"
    material = "\n".join([*ctx.findings, *ctx.must_include,
                           ctx.action_prompt])
    lowered_material = material.lower()
    invented = _amounts(text) - _amounts(material)
    if invented:
        return text, "figure_not_in_findings"
    for number in _LONG_NUMBER_RE.findall(text):
        if number not in material:
            return text, "number_not_in_findings"
    for match in _PROMISE_RE.finditer(text):
        if match.group(0).lower() not in lowered_material:
            return text, "promise_not_in_findings"
    lowered = text.lower()
    for term in _FOREIGN_TERMS:
        if term in lowered and term not in lowered_material:
            return text, "term_not_in_findings"
    return text, ""


class OfflineLLM(_CallLedger):
    backend = "offline"

    def __init__(self) -> None:
        self._init_ledger()

    def draft_reply(self, ctx: DraftContext) -> str:
        # The offline composer stands in for the decision call,
        # and is accounted like one: its tokens are the
        # deterministic estimate over the material it composed
        # from and the reply it produced, labelled estimated.
        started = time.perf_counter()
        reply = _offline_compose(ctx)
        prompt_text = "\n".join(
            [ctx.customer_message, *ctx.findings, *ctx.must_include,
             ctx.action_prompt])
        self._record_call(
            "decision", attempt=1, outcome="ok",
            tokens_in=estimate_tokens(prompt_text),
            tokens_out=estimate_tokens(reply), basis="estimated",
            latency_ms=_elapsed_ms(started))
        return reply

    def intent_hint(self, text: str) -> str | None:
        # No call is made offline — the router commits on its own
        # evidence — so no record is written. A record for a call
        # that never happened would be the one fabrication this
        # ledger exists to prevent.
        return None


class LiveLLM(_CallLedger):
    # The prompt registry the prompts resolve from — wired by the
    # service at construction. Deliberately not defaulted: a live
    # drafter without a registry refuses (see _require_prompts).
    prompts = None

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.backend = config.backend
        self._init_ledger()
        # The governed per-call timeout (Policy carries the
        # value; the service wires it on). Sixty seconds is the
        # boundary's own default for a directly built boundary.
        self.timeout_seconds = 60.0
        # The last provider response's reported usage, as
        # (tokens_in, tokens_out) — set by _chat, read by the
        # attempt that made the call. None when the provider
        # reported none: the attempt then estimates, and says so.
        self._last_usage: tuple[int, int] | None = None

    def _require_prompts(self):
        if self.prompts is None:
            raise PromptRegistryError(
                "prompt registry: the live drafter has no registry "
                "wired — it will not draft from a remembered prompt; "
                "the service wires the store's registry at "
                "construction")
        return self.prompts

    # -- provider calls ------------------------------------------------
    def _chat(self, system: str, user: str) -> str:
        import httpx

        cfg = self.config
        if cfg.backend == "openai":
            resp = httpx.post(
                f"{cfg.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json={"model": cfg.model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "temperature": 0.3},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage") or {}
            self._last_usage = (usage.get("prompt_tokens"),
                                usage.get("completion_tokens"))
            return body["choices"][0]["message"]["content"]
        if cfg.backend == "anthropic":
            resp = httpx.post(
                f"{cfg.base_url.rstrip('/')}/v1/messages",
                headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
                json={"model": cfg.model, "max_tokens": 700, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            usage = body.get("usage") or {}
            self._last_usage = (usage.get("input_tokens"),
                                usage.get("output_tokens"))
            blocks = body["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if cfg.backend == "ollama":
            resp = httpx.post(
                f"{cfg.base_url.rstrip('/')}/api/chat",
                json={"model": cfg.model, "stream": False,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=self.timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            self._last_usage = (body.get("prompt_eval_count"),
                                body.get("eval_count"))
            return body["message"]["content"]
        raise ValueError(f"unknown backend {cfg.backend}")

    # -- the guarded attempt --------------------------------------------
    def _attempt(self, call_kind: str, attempt: int, system: str,
                 user: str) -> str | None:
        """One provider call, recorded however it ends: the text
        on success, None on an ordinary failure (the caller
        degrades, as it always has). A timeout is recorded and
        raised as ModelTimeoutError — the retry decision, and
        what a second timeout means, belong to `_guarded` and
        the supervisor, not to a silent fallback here."""
        self._last_usage = None
        started = time.perf_counter()
        prompt_text = system + "\n" + user
        try:
            text = self._chat(system, user)
        except Exception as exc:
            latency = _elapsed_ms(started)
            if _is_timeout(exc):
                self._record_call(
                    call_kind, attempt=attempt, outcome="timeout",
                    tokens_in=estimate_tokens(prompt_text),
                    tokens_out=0, basis="estimated",
                    latency_ms=latency, model=self.config.model)
                raise ModelTimeoutError(
                    f"the {call_kind} call timed out "
                    f"(attempt {attempt}, backend {self.backend})"
                ) from exc
            self._record_call(
                call_kind, attempt=attempt, outcome="error",
                tokens_in=estimate_tokens(prompt_text),
                tokens_out=0, basis="estimated",
                latency_ms=latency, model=self.config.model)
            log.warning("live model call failed",
                        extra={"fields": {"error": str(exc)}})
            return None
        latency = _elapsed_ms(started)
        usage_in, usage_out = self._last_usage or (None, None)
        if usage_in is not None and usage_out is not None:
            tokens_in, tokens_out, basis = usage_in, usage_out, "measured"
        else:
            tokens_in = estimate_tokens(prompt_text)
            tokens_out = estimate_tokens(text)
            basis = "estimated"
        self._record_call(
            call_kind, attempt=attempt, outcome="ok",
            tokens_in=tokens_in, tokens_out=tokens_out, basis=basis,
            latency_ms=latency, model=self.config.model)
        return text

    def _guarded(self, call_kind: str, system: str, user: str) -> str | None:
        """A call with its one governed retry on timeout. A
        second timeout raises ModelTimeoutError out of the
        boundary; any other failure returns None and the caller
        degrades."""
        for attempt in (1, 2):
            try:
                return self._attempt(call_kind, attempt, system, user)
            except ModelTimeoutError:
                if attempt == 2:
                    raise
        return None  # pragma: no cover — the loop always returns or raises

    # -- the two language jobs -----------------------------------------
    def draft_reply(self, ctx: DraftContext) -> str:
        # The prompts resolve BEFORE the guarded call, on purpose:
        # a registry that cannot serve them is a deployment defect
        # and refuses loudly — it is not a drafting failure to
        # degrade around.
        registry = self._require_prompts()
        system = registry.active_text(COMPOSE_SYSTEM_ID)
        facts = "\n".join(f"- {t}" for t in ctx.findings) or "- (no findings)"
        verbatim = "\n".join(f"- {t}" for t in ctx.must_include)
        sources = ""
        if ctx.citations:
            refs = "; ".join(f"{d} ({t})" for d, t in ctx.citations)
            sources = f"\nSources line to end with: Sources: {refs}."
        user = render_template(
            registry.active_text(COMPOSE_USER_ID),
            customer_message=ctx.customer_message, facts=facts,
            verbatim=verbatim or "- (none)",
            action_prompt=ctx.action_prompt or "(none)", sources=sources)
        # A double timeout leaves the boundary as
        # ModelTimeoutError — the supervisor closes the turn to
        # a person rather than composing from nothing. Any other
        # failure degrades to the offline composer, as before:
        # drafting degrades, rules do not.
        text = self._guarded("decision", system, user)
        if text and text.strip():
            vetted, refusal = vet_draft(text, ctx)
            if not refusal:
                return vetted
            # The call happened and is accounted; the draft simply
            # doesn't stand. The refusal rides the call's record onto
            # the audit trail, and the composer answers the turn.
            if self._pending_calls:
                self._pending_calls[-1]["vetting"] = refusal
            log.warning("live draft refused",
                        extra={"fields": {"reason": refusal}})
        return _offline_compose(ctx)

    def intent_hint(self, text: str) -> str | None:
        allowed = "billing, roaming, offers, technical, account, care"
        registry = self._require_prompts()
        system = registry.active_text(INTENT_SYSTEM_ID)
        user = render_template(registry.active_text(INTENT_USER_ID),
                               labels=allowed, text=text)
        out = self._guarded("advisory", system, user)
        if out is None:
            return None
        out = out.strip().lower()
        return out if out in allowed.split(", ") else None


def make_llm(config: LLMConfig | None = None):
    config = config or LLMConfig.from_env()
    if config.is_live:
        return LiveLLM(config)
    return OfflineLLM()
