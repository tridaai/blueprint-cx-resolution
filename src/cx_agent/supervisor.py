"""The supervisor: a LangGraph graph that owns every conversation.

    screen ─▶ dispatch ─┬─▶ escalate ──────────────┐
                        ├─▶ identity ─┬─▶ post ─┬─▶ compose ─▶ finalize
                        ├─▶ confirm ──┤         └─▶ escalate
                        └─▶ route ─▶ specialist ─▶ post

The supervisor applies screening, identity, escalation discipline
and the confirmation protocol, routes to one of six specialists,
and composes the single customer-facing reply from the specialist's
structured findings. Specialists return data; only this graph
formats prose, exactly once per turn — the production guides'
hard-won lesson about duplicated formatting and cost.

Conversation state lives in the store (the app is stateless); the
graph is invoked per turn over the loaded record.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import actions as action_mod
from . import escalation as esc_mod
from . import identity as id_mod
from .config import Policy
from .llm import DraftContext
from .models import (
    AuditEvent,
    Conversation,
    Escalation,
    Finding,
    IdentityState,
    Intent,
    SpecialistResult,
    TurnResult,
)
from .routing import classify_confirmation, classify_step_outcome, route, score_intents
from .screening import Screening, screen
from .specialists import Ctx, account, billing, care, offers, roaming, technical

log = logging.getLogger("cx_agent.supervisor")

SPECIALISTS = {
    Intent.BILLING: billing,
    Intent.ROAMING: roaming,
    Intent.OFFERS: offers,
    Intent.TECHNICAL: technical,
    Intent.ACCOUNT: account,
    Intent.CARE: care,
}

_QUEUE_LABEL = {
    "general": "a support specialist",
    "billing": "a billing specialist",
    "fraud": "our fraud team",
    "vulnerable_care": "a member of our care team",
}

_ACCOUNT_RE = re.compile(r"\b(NB-\d{4,})\b", re.IGNORECASE)
_POSTCODE_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


class TurnState(TypedDict, total=False):
    conv: Conversation
    text: str
    scr: Screening
    decision: esc_mod.EscalationDecision
    result: SpecialistResult
    reply: str
    action_record: object
    escalated: bool
    first_human_ask: bool
    turn_result: TurnResult
    next: str  # dispatch scratch: escalate | identity | confirm | route | post | compose


class Supervisor:
    def __init__(self, store, policy: Policy, llm, retriever) -> None:
        self.store = store
        self.policy = policy
        self.llm = llm
        self.ctx = Ctx(store=store, policy=policy, retriever=retriever)
        self.graph = self._build()

    # -- graph construction ---------------------------------------------
    def _build(self):
        g = StateGraph(TurnState)
        g.add_node("screen", self._screen)
        g.add_node("identity", self._identity)
        g.add_node("confirm", self._confirm)
        g.add_node("route", self._route)
        g.add_node("specialist", self._specialist)
        g.add_node("post", self._post)
        g.add_node("compose", self._compose)
        g.add_node("escalate", self._escalate)
        g.add_node("finalize", self._finalize)
        g.set_entry_point("screen")
        g.add_conditional_edges("screen", self._dispatch,
                                {"escalate": "escalate", "identity": "identity",
                                 "confirm": "confirm", "route": "route"})
        g.add_conditional_edges("identity", lambda s: s["next"],
                                {"post": "post", "route": "route", "escalate": "escalate"})
        g.add_conditional_edges("confirm", lambda s: s["next"],
                                {"post": "post", "route": "route"})
        g.add_edge("route", "specialist")
        g.add_edge("specialist", "post")
        g.add_conditional_edges("post", lambda s: s["next"],
                                {"compose": "compose", "escalate": "escalate"})
        g.add_edge("compose", "finalize")
        g.add_edge("escalate", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # -- nodes ------------------------------------------------------------
    def _audit(self, conv: Conversation, kind: str, detail: dict) -> None:
        self.store.save_audit(AuditEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:10]}",
            conversation_id=conv.conversation_id, kind=kind,
            detail=detail, turn=conv.turns))

    def _screen(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        conv.turns += 1
        scr = screen(text)
        state["scr"] = scr
        if scr.injection:
            self._audit(conv, "screening",
                        {"injection_attempt": True, "matched": scr.matched})
            log.info("injection screened", extra={"fields": {"matched": scr.matched}})
        customer = (self.store.get_customer(conv.customer_id)
                    if conv.customer_id else None)
        acct = (self.store.get_account(conv.account_number)
                if conv.account_number else None)
        if acct is not None and customer is None:
            customer = self.store.get_customer(acct.customer_id)
        decision = esc_mod.evaluate(
            conv, scr, self.policy,
            account_vulnerable=bool(customer and customer.vulnerable))
        state["decision"] = decision
        state["first_human_ask"] = bool(
            scr.asks_for_human and conv.human_requests == 1 and not decision.escalate)
        return state

    def _dispatch(self, state: TurnState) -> str:
        conv = state["conv"]
        if state["decision"].escalate:
            return "escalate"
        if conv.specialist_state.get("awaiting"):
            return "identity"
        if conv.pending_action is not None:
            return "confirm"
        return "route"

    # -- identity flow ------------------------------------------------------
    def _identity(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        awaiting = conv.specialist_state.get("awaiting")
        findings: list[Finding] = []

        if awaiting == "details":
            acct_match = _ACCOUNT_RE.search(text)
            pc_match = _POSTCODE_RE.search(text)
            if acct_match and pc_match:
                ok, key = id_mod.begin_verification(
                    conv, acct_match.group(1).upper(), pc_match.group(1), self.store)
                if ok:
                    conv.specialist_state["awaiting"] = "otp"
                    self._audit(conv, "identity", {"event": "code_issued"})
                    findings.append(Finding(
                        topic="identity",
                        text=("Thank you — that matches. I've sent a one-time "
                              "code to the contact on file; enter the 6-digit "
                              "code here and we'll carry straight on.")))
                    state["result"] = SpecialistResult(
                        specialist="supervisor", findings=findings, progress=True)
                    state["next"] = "post"
                    return state
                findings.append(Finding(
                    topic="identity",
                    text=("Those details don't match an account I can find. "
                          "Check the account number (NB- followed by digits) "
                          "and the postcode on the account, and try again.")))
                state["result"] = SpecialistResult(
                    specialist="supervisor", findings=findings, progress=False)
                state["next"] = "post"
                return state
            findings.append(Finding(
                topic="identity",
                text=("I need both pieces together: your account number "
                      "(it starts NB-) and the postcode on the account.")))
            state["result"] = SpecialistResult(
                specialist="supervisor", findings=findings, progress=False)
            state["next"] = "post"
            return state

        # awaiting == "otp"
        code_match = _CODE_RE.search(text)
        if not code_match:
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(
                    topic="identity",
                    text="Enter the 6-digit code I sent — it looks like a "
                         "short number on its own.")],
                progress=True)
            state["next"] = "post"
            return state
        ok, key = id_mod.submit_code(conv, code_match.group(1),
                                     self.policy.otp_max_attempts)
        if ok:
            self._audit(conv, "identity", {"event": "verified"})
            conv.specialist_state.pop("awaiting", None)
            if conv.specialist_state.pop("resume_pending", False):
                pending = conv.pending_action
                state["result"] = SpecialistResult(
                    specialist="supervisor",
                    findings=[Finding(
                        topic="identity",
                        text=("Identity verified. Back to where we were: "
                              f"shall I go ahead with this — {pending.summary}?"))],
                    progress=True)
                state["next"] = "post"
                return state
            resume_intent = conv.specialist_state.pop("resume_intent", None)
            if resume_intent:
                conv.specialist_state["force_intent"] = resume_intent
            resume_text = conv.specialist_state.pop("resume_text", "") or text
            state["text"] = resume_text
            state["next"] = "route"
            return state
        if key == "locked":
            self._audit(conv, "identity", {"event": "locked"})
            state["decision"] = esc_mod.EscalationDecision(
                True, "identity_verification_failed", "general")
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(
                    topic="identity",
                    text=("That's the third code that hasn't matched, so "
                          "verification is locked for this chat. A person "
                          "can complete it with you instead."))],
                progress=True)
            state["next"] = "escalate"
            return state
        left = self.policy.otp_max_attempts - conv.identity_attempts
        state["result"] = SpecialistResult(
            specialist="supervisor",
            findings=[Finding(
                topic="identity",
                text=f"That code doesn't match. {left} attempt(s) left — "
                     "check the newest message and try again.")],
            progress=True)
        state["next"] = "post"
        return state

    # -- confirmation protocol ---------------------------------------------
    def _confirm(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        pending = conv.pending_action
        verdict = classify_confirmation(text)
        if verdict == "other":
            state["next"] = "route"
            return state
        if verdict == "decline":
            record = action_mod.new_action_record(conv, pending, status="declined")
            self.store.save_action(record)
            self._audit(conv, "action", {"action_id": record.action_id,
                                         "status": "declined"})
            conv.pending_action = None
            state["action_record"] = record
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(
                    topic="declined",
                    text=("No problem — I've left everything exactly as it "
                          "was. Nothing has been changed or charged."))],
                progress=True)
            state["next"] = "post"
            return state
        # confirm
        from .identity import is_verified
        from .models import CONSEQUENTIAL_ACTIONS
        if pending.kind in CONSEQUENTIAL_ACTIONS and not is_verified(conv):
            conv.specialist_state["resume_pending"] = True
            conv.specialist_state["resume_text"] = text
            if conv.identity_state == IdentityState.VERIFYING:
                conv.specialist_state["awaiting"] = "otp"
                ask = ("Before I apply that, I need the one-time code I sent "
                       "to the contact on file — enter the 6-digit code here.")
            else:
                conv.specialist_state["awaiting"] = "details"
                ask = ("Before I apply that, I need to verify it's really "
                       "you: your account number (it starts NB-) and the "
                       "postcode on the account.")
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(topic="verify_for_action", text=ask)],
                progress=True)
            state["next"] = "post"
            return state
        record = action_mod.execute(self.store, conv, pending, self.policy,
                                    confirmed=True)
        conv.pending_action = None
        state["action_record"] = record
        if record.status == "executed":
            text_out = (f"Done — {pending.summary}. It's recorded on your "
                        "account, and you'll see it on your next bill.")
            resolved = True
        else:
            text_out = ("I couldn't apply that after all — a final check at "
                        "our end stopped it, and nothing has changed. A person "
                        "can pick this up if you'd like.")
            resolved = False
        state["result"] = SpecialistResult(
            specialist="supervisor",
            findings=[Finding(topic="action_done", text=text_out)],
            progress=True, resolved=resolved)
        state["next"] = "post"
        return state

    # -- routing + specialists ------------------------------------------------
    def _route(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        forced = conv.specialist_state.pop("force_intent", None)
        if forced:
            # Resuming after identity: the desk that asked for
            # verification owns the next turn, whatever the resumed
            # text happens to score as.
            conv.intent = Intent(forced)
            return state
        scores = score_intents(text)
        hint = None
        if scores and self.llm.backend != "offline":
            top = max(scores.values())
            if sum(1 for s in scores.values() if s == top) > 1:
                hint = self.llm.intent_hint(text)
        intent = route(text, hint)
        # A live procedure owns short outcome replies ("yes", "still
        # nothing") that carry no routing signal of their own.
        walk = conv.specialist_state.get("technical", {})
        if walk.get("procedure_id") and intent != Intent.TECHNICAL:
            if classify_step_outcome(text) != "unclear" or not scores:
                intent = Intent.TECHNICAL
        conv.intent = intent
        return state

    def _specialist(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        fn = SPECIALISTS.get(conv.intent, care)
        result = fn(conv, text, self.ctx)
        # The escalation rubric's first-ask rule: a customer who asks
        # for a human with no stated problem gets one genuine solve
        # attempt — an invitation to describe the problem — not a
        # policy dump from the catch-all desk.
        if state.get("first_human_ask") and conv.intent == Intent.CARE \
                and not result.proposed_action:
            result = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(
                    topic="human_offer",
                    text=("Of course — a person is always available. Before "
                          "I transfer you, give me one chance to sort it "
                          "here; it's usually quicker. What's happened?"))],
                progress=True)
        state["result"] = result
        return state

    # -- post-specialist bookkeeping -------------------------------------------
    def _post(self, state: TurnState) -> TurnState:
        conv: Conversation = state["conv"]
        result: SpecialistResult = state["result"]

        if result.needs_identity or result.needs_verification:
            if result.needs_identity and conv.account_number is not None:
                pass  # already identified; the specialist can proceed next turn
            elif result.needs_verification and conv.identity_state == IdentityState.VERIFIED:
                pass  # specialist misread the state; let its findings stand
            else:
                conv.specialist_state["resume_text"] = state["text"]
                conv.specialist_state["resume_intent"] = conv.intent.value
                if result.needs_verification and conv.identity_state == IdentityState.VERIFYING:
                    conv.specialist_state["awaiting"] = "otp"
                    result.findings.append(Finding(
                        topic="identity",
                        text="Enter the 6-digit code I sent to the contact on file."))
                else:
                    conv.specialist_state["awaiting"] = "details"
            self._bookkeep_progress(conv, result)
            state["next"] = "compose"
            return state

        if result.escalate_reason:
            state["decision"] = esc_mod.EscalationDecision(
                True, result.escalate_reason, result.escalate_queue or "general")
            self._bookkeep_progress(conv, result)
            state["next"] = "escalate"
            return state

        if result.proposed_action is not None and state.get("action_record") is None:
            record = action_mod.new_action_record(conv, result.proposed_action)
            self.store.save_action(record)
            conv.pending_action = result.proposed_action
            state["action_record"] = record

        self._bookkeep_progress(conv, result)
        if result.resolved:
            conv.status = "resolved"
        if conv.no_progress_turns >= self.policy.max_no_progress_turns \
                and not result.resolved:
            state["decision"] = esc_mod.EscalationDecision(
                True, "no_progress",
                "billing" if conv.intent == Intent.BILLING else "general")
            state["next"] = "escalate"
            return state
        state["next"] = "compose"
        return state

    def _bookkeep_progress(self, conv: Conversation, result: SpecialistResult) -> None:
        if result.progress:
            conv.no_progress_turns = 0
        else:
            conv.no_progress_turns += 1

    # -- composition -------------------------------------------------------------
    def _compose(self, state: TurnState) -> TurnState:
        conv, result = state["conv"], state["result"]
        customer = (self.store.get_customer(conv.customer_id)
                    if conv.customer_id else None)
        must_include = list(result.must_include)
        if state.get("first_human_ask"):
            must_include.append(
                "If you'd still rather speak to a person after this, just ask "
                "again and I'll transfer you straight away.")
        citations = []
        for doc_id in result.citations:
            doc = self.store.get_policy(doc_id)
            if doc is not None:
                citations.append((doc.doc_id, doc.title))
        action_prompt = ""
        if result.proposed_action is not None:
            p = result.proposed_action
            amount = f" ({p.amount})" if p.amount is not None else ""
            action_prompt = (
                f"Shall I go ahead and {p.summary}? Reply yes and I'll do it "
                "now, or no and nothing changes.")
        ctx = DraftContext(
            customer_name=customer.full_name if customer else "",
            intent=conv.intent.value,
            findings=[f.text for f in result.findings],
            must_include=must_include,
            action_prompt=action_prompt,
            citations=citations,
            customer_message=state["text"],
        )
        reply = self.llm.draft_reply(ctx)
        # Code-level guarantees, whatever the drafter did:
        for line in must_include:
            if line not in reply:
                reply = f"{reply} {line}".strip()
        if action_prompt and action_prompt not in reply:
            reply = f"{reply} {action_prompt}".strip()
        if citations and not any(doc_id in reply for doc_id, _ in citations):
            refs = "; ".join(f"{d} ({t})" for d, t in citations)
            reply = f"{reply} Sources: {refs}.".strip()
        state["reply"] = reply
        return state

    # -- escalation ----------------------------------------------------------------
    def _escalate(self, state: TurnState) -> TurnState:
        conv = state["conv"]
        decision = state["decision"]
        packet = esc_mod.build_packet(conv, self.store, decision.reason,
                                       decision.queue)
        esc = Escalation(
            escalation_id=f"ESC-{uuid.uuid4().hex[:10]}",
            conversation_id=conv.conversation_id,
            reason=decision.reason, queue=decision.queue,
            packet=packet, turn=conv.turns)
        self.store.save_escalation(esc)
        self._audit(conv, "escalation",
                    {"escalation_id": esc.escalation_id, "reason": decision.reason,
                     "queue": decision.queue})
        conv.status = "escalated"
        conv.pending_action = None
        state["escalated"] = True
        prefix = ""
        result = state.get("result")
        if result is not None and result.findings:
            prefix = " ".join(f.text for f in result.findings) + " "
        state["reply"] = (
            f"{prefix}I'm transferring you to {_QUEUE_LABEL.get(decision.queue, 'a specialist')} "
            f"now. Your reference is {esc.escalation_id}, and I've passed on "
            "everything we've covered — the summary, what we tried, and what's "
            "still open — so you won't need to repeat yourself.")
        return state

    # -- finalize ----------------------------------------------------------------------
    def _finalize(self, state: TurnState) -> TurnState:
        conv = state["conv"]
        state["turn_result"] = TurnResult(
            conversation_id=conv.conversation_id,
            reply=state.get("reply", ""),
            intent=conv.intent.value,
            identity_state=conv.identity_state.value,
            status=conv.status,
            escalated=bool(state.get("escalated")),
            escalation_reason=(state["decision"].reason
                               if state.get("escalated") and state.get("decision")
                               else ""),
            action=state.get("action_record"),
            citations=(state["result"].citations if state.get("result") else []),
        )
        self._audit(conv, "turn",
                    {"intent": conv.intent.value,
                     "escalated": bool(state.get("escalated")),
                     "identity_state": conv.identity_state.value})
        return state

    # -- entry point ----------------------------------------------------------------------
    def run_turn(self, conv: Conversation, text: str) -> TurnResult:
        from .models import Message

        conv.messages.append(Message(role="customer", text=text,
                                     turn=conv.turns + 1))
        state: TurnState = {"conv": conv, "text": text}
        out = self.graph.invoke(state)
        result: TurnResult = out["turn_result"]
        conv.messages.append(Message(role="agent", text=result.reply,
                                     turn=conv.turns))
        return result
