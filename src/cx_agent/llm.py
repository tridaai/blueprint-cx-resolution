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

The prompts themselves are not in this file: the compose and
intent instructions are prompt-registry entries (prompts.py),
resolved at call time at their active versions. A live drafter
with no registry wired, or a registry that cannot serve an
entry, refuses with PromptRegistryError — the boundary never
falls back to a prompt from memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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


class OfflineLLM:
    backend = "offline"

    def draft_reply(self, ctx: DraftContext) -> str:
        return _offline_compose(ctx)

    def intent_hint(self, text: str) -> str | None:
        return None


class LiveLLM:
    # The prompt registry the prompts resolve from — wired by the
    # service at construction. Deliberately not defaulted: a live
    # drafter without a registry refuses (see _require_prompts).
    prompts = None

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.backend = config.backend

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
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        if cfg.backend == "anthropic":
            resp = httpx.post(
                f"{cfg.base_url.rstrip('/')}/v1/messages",
                headers={"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"},
                json={"model": cfg.model, "max_tokens": 700, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=60,
            )
            resp.raise_for_status()
            blocks = resp.json()["content"]
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if cfg.backend == "ollama":
            resp = httpx.post(
                f"{cfg.base_url.rstrip('/')}/api/chat",
                json={"model": cfg.model, "stream": False,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        raise ValueError(f"unknown backend {cfg.backend}")

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
        try:
            text = self._chat(system, user).strip()
            if text:
                return text
        except Exception as exc:  # drafting degrades, rules do not
            log.warning("live drafting failed; offline composer used",
                        extra={"fields": {"error": str(exc)}})
        return _offline_compose(ctx)

    def intent_hint(self, text: str) -> str | None:
        allowed = "billing, roaming, offers, technical, account, care"
        registry = self._require_prompts()
        system = registry.active_text(INTENT_SYSTEM_ID)
        user = render_template(registry.active_text(INTENT_USER_ID),
                               labels=allowed, text=text)
        try:
            out = self._chat(system, user).strip().lower()
            return out if out in allowed.split(", ") else None
        except Exception as exc:
            log.warning("intent hint failed", extra={"fields": {"error": str(exc)}})
            return None


def make_llm(config: LLMConfig | None = None):
    config = config or LLMConfig.from_env()
    if config.is_live:
        return LiveLLM(config)
    return OfflineLLM()
