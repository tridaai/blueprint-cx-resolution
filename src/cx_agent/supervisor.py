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
from . import billing_book
from . import payments_book
from . import escalation as esc_mod
from . import identity as id_mod
from . import procedures
from .config import Policy
from .llm import DraftContext
from .models import (
    ActionKind,
    AuditEvent,
    Conversation,
    Escalation,
    Finding,
    IdentityState,
    Intent,
    SpecialistResult,
    TurnResult,
)
from .routing import (
    READINGS,
    analyze,
    classify_confirmation,
    classify_step_outcome,
    resolve_clarification,
    route,
)
from .screening import Screening, screen
from . import specialists as specialists_mod
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
    "offers": "an offers specialist",
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
    routing: dict  # the router's evidence for this turn (see routing.analyze)
    clarification_resolved: bool
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
        g.add_conditional_edges("route", lambda s: s["next"],
                                {"specialist": "specialist", "post": "post"})
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
        if scr.vulnerable_statement:
            # A plainly stated circumstance sets the persistent
            # flag on the account record — here, in code, with the
            # audit event — and never anything weaker: the broad
            # keyword screen routes this conversation to a person,
            # but it does not write a lasting record about the
            # customer. The account is the one the conversation is
            # linked to, or the one the customer names in this
            # message; with neither, there is no record to set the
            # flag on, and the escalation below still protects the
            # conversation.
            target = customer
            if target is None:
                named = _ACCOUNT_RE.search(text)
                if named:
                    target = self.store.get_customer_by_account(
                        named.group(1).upper())
            if target is not None and not target.vulnerable:
                target.vulnerable = True
                self.store.save_customer(target)
                self._audit(conv, "vulnerability", {
                    "event": "flag_set",
                    "statement_class": scr.vulnerable_statement,
                    "source": "explicit_customer_statement",
                    "customer_id": target.customer_id})
                if conv.customer_id is None:
                    # The record for the human who picks this up:
                    # the conversation hands over linked to the
                    # account it concerns. (The turn escalates
                    # below — no desk acts on this link.)
                    conv.customer_id = target.customer_id
                    conv.account_number = target.account_number
                customer = target
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
            if self._identity_detour_stale(state):
                # A desk asked for identity mid-procedure and the
                # customer has come back to the procedure instead.
                # The detour is stale — drop it and route the turn;
                # the walk-ownership rule below hands the reply to
                # the technical desk, and the run resumes at its
                # exact step. An identity detour never swallows a
                # live procedure.
                conv.specialist_state.pop("awaiting", None)
                conv.specialist_state.pop("resume_intent", None)
                conv.specialist_state.pop("resume_text", None)
                conv.specialist_state.pop("resume_pending", None)
            else:
                return "identity"
        if conv.pending_action is not None:
            return "confirm"
        return "route"

    def _identity_detour_stale(self, state: TurnState) -> bool:
        """The identity detour is stale when a procedure run is
        live and this message carries no identity material at all
        — no account number, no postcode, no code. A message that
        could be identity material always goes to the identity
        node, detour or not."""
        conv, text = state["conv"], state["text"]
        tech = conv.specialist_state.get("technical", {})
        run_id = tech.get("run_id")
        if not run_id:
            return False
        run = self.store.get_procedure_run(run_id)
        if run is None or run.state != "active":
            return False
        return not (_ACCOUNT_RE.search(text) or _POSTCODE_RE.search(text)
                    or _CODE_RE.search(text))

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
            if not analyze(text).scores:
                # No desk signal of any kind: this is not a new
                # request, it is an answer we cannot read — a
                # hesitant "um, yes I think so", a fragment, a
                # mumble. Routing it as a fresh turn would land it
                # on the care desk, and the desk change would
                # retire the proposal the customer was mid-answer
                # to: a credit they had effectively agreed to
                # would vanish without anyone deciding that. The
                # proposal stays open and the question is asked
                # again, plainly. A reply carrying any real signal
                # still routes (and a desk change still retires,
                # per the rerouting rule) — only the signal-less
                # non-answer is held here.
                self._audit(conv, "action", {
                    "event": "confirmation_reasked",
                    "action_kind": pending.kind.value})
                state["result"] = SpecialistResult(
                    specialist="supervisor",
                    findings=[Finding(
                        topic="confirmation",
                        text=("Just so I have it straight — shall I go "
                              f"ahead with this: {pending.summary}? "
                              "A yes or a no is all I need."))],
                    progress=False)
                state["next"] = "post"
                return state
            state["next"] = "route"
            return state
        if verdict == "decline":
            record = action_mod.new_action_record(conv, pending, status="declined")
            self.store.save_action(record)
            self._audit(conv, "action", {"action_id": record.action_id,
                                         "status": "declined"})
            case_id = pending.payload.get("dispute_case_id")
            if case_id:
                # The case stays opened — the offer was refused, the
                # dispute wasn't decided — with the answer on record.
                billing_book.settle_dispute_case(
                    self.store, conv, case_id, "declined", None,
                    note=(f"The customer declined the offered credit of "
                          f"{pending.amount}; nothing was applied."))
            callback_id = pending.payload.get("callback_id")
            if callback_id:
                # The callback record settles with the answer too:
                # the booking was offered and declined, on record.
                procedures.settle_callback(self.store, callback_id,
                                           "declined")
            conv.pending_action = None
            conv.specialist_state.pop("pending_desk", None)
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
        if record.status == "blocked" \
                and record.block_reason == "stale_proposal" \
                and pending.kind == ActionKind.CHANGE_PLAN:
            # The proposal was priced against an account that has
            # since moved. Its record is marked stale on the books
            # (superseded, with the reason), and the offers desk
            # re-quotes the same request from live state for a fresh
            # confirmation — the customer is never asked to confirm
            # figures that no longer match their account, and the
            # refusal is explained rather than buried.
            self._retire_pending(conv, (
                "stale_proposal: the account state changed after this "
                "proposal was priced; refused at execution and "
                "superseded by a fresh proposal"))
            fresh = specialists_mod.refresh_plan_change(
                conv, self.ctx, pending)
            conv.pending_action = None
            conv.specialist_state.pop("pending_desk", None)
            if fresh.proposed_action is not None:
                fresh_record = action_mod.new_action_record(
                    conv, fresh.proposed_action)
                self.store.save_action(fresh_record)
                conv.pending_action = fresh.proposed_action
                conv.specialist_state["pending_desk"] = "offers"
            state["action_record"] = record
            state["result"] = fresh
            state["next"] = "post"
            return state
        callback_id = pending.payload.get("callback_id")
        if callback_id:
            # The callback record settles on the booking's outcome:
            # booked, with the executed action's id, or blocked at
            # the final check with nothing applied.
            if record.status == "executed":
                procedures.settle_callback(self.store, callback_id,
                                           "booked",
                                           action_id=record.action_id)
            else:
                procedures.settle_callback(self.store, callback_id,
                                           "blocked")
        case_id = pending.payload.get("dispute_case_id")
        if case_id:
            # The dispute case settles on the credit's outcome: the
            # applied amount when it lands, or the block on record.
            if record.status == "executed":
                billing_book.settle_dispute_case(
                    self.store, conv, case_id, "executed", record.amount)
            else:
                billing_book.settle_dispute_case(
                    self.store, conv, case_id, "blocked", None,
                    note=(f"The credit was stopped at execution "
                          f"({record.block_reason}); nothing was applied."))
        if pending.kind == ActionKind.TAKE_PAYMENT:
            # A payment's outcome is the attempt's outcome: the
            # reply is composed from the attempt record (paid, or
            # the diagnosis for the decline's class), never from
            # the generic action text — "Done" would be a lie about
            # a decline. Either way the next turn belongs to the
            # account desk: a soft decline may be retried there, a
            # hard one starts the new-card path there.
            text_out, resolved = payments_book.payment_outcome_text(
                self.store, record)
            conv.specialist_state["force_intent"] = "account"
            if record.payload.get("decline_class") == "soft":
                conv.specialist_state.setdefault("account", {})[
                    "await_retry"] = True
            conv.pending_action = None
            conv.specialist_state.pop("pending_desk", None)
            state["action_record"] = record
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(topic="payment_outcome", text=text_out)],
                progress=True, resolved=resolved)
            state["next"] = "post"
            return state
        conv.pending_action = None
        conv.specialist_state.pop("pending_desk", None)
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
    def _hint(self, text: str, analysis) -> str | None:
        """The model's tie-break hint, live mode only. Used when the
        router commits — a three-way tie, or a clarification the
        customer left unanswered — never instead of asking when
        exactly two desks tie on a real signal."""
        if analysis.scores and self.llm.backend != "offline" \
                and len(analysis.leaders) > 1:
            return self.llm.intent_hint(text)
        return None

    def _route(self, state: TurnState) -> TurnState:
        conv, text = state["conv"], state["text"]
        forced = conv.specialist_state.pop("force_intent", None)
        if forced:
            # Resuming after identity: the desk that asked for
            # verification owns the next turn, whatever the resumed
            # text happens to score as.
            conv.intent = Intent(forced)
            conv.specialist_state["desk"] = conv.intent.value
            state["routing"] = {"intent": conv.intent.value,
                                "decision": "resume_after_identity"}
            state["next"] = "specialist"
            return state

        analysis = analyze(text)
        record: dict = {
            "intent": conv.intent.value,
            "decision": "route",
            "signals": {i.value: hits
                        for i, hits in analysis.signals.items()},
            "scores": {i.value: s for i, s in analysis.scores.items()},
            "runner_up": (analysis.runner_up.value
                          if analysis.runner_up else None),
        }
        state["routing"] = record

        # An open clarification is answered before anything else
        # routes. A reply that picks one of the two readings
        # resolves it — and that resolution is real progress. A
        # reply that settles nothing falls through to a commit:
        # the router decides below and never asks a second question.
        clarify = conv.specialist_state.pop("clarify", None)
        if clarify:
            options = [Intent(o) for o in clarify["options"]]
            resolved = resolve_clarification(text, options)
            if resolved is not None:
                record["decision"] = "clarification_resolved"
                state["clarification_resolved"] = True
                conv.intent = resolved
                self._change_desk(state, resolved, record)
                state["next"] = "specialist"
                return state
            record["decision"] = "clarification_committed"

        # A live procedure owns short outcome replies ("yes", "still
        # nothing") that carry no routing signal of their own.
        walk = conv.specialist_state.get("technical", {})
        if walk.get("procedure_id"):
            raw = route(text, self._hint(text, analysis))
            if raw != Intent.TECHNICAL and (
                    classify_step_outcome(text) != "unclear"
                    or not analysis.scores):
                record["decision"] = "procedure_walk"
                conv.intent = Intent.TECHNICAL
                self._change_desk(state, Intent.TECHNICAL, record)
                state["next"] = "specialist"
                return state

        # Two desks tie on a real signal: ask ONE targeted question
        # offering the two readings in plain words instead of
        # guessing a desk. The desk does not change while the
        # question is open — the supervisor holds the turn.
        if clarify is None and analysis.ambiguous:
            options = analysis.leaders
            conv.specialist_state["clarify"] = {
                "options": [o.value for o in options],
                "asked_turn": conv.turns,
            }
            record["decision"] = "clarify"
            first, second = (READINGS[o] for o in options)
            state["result"] = SpecialistResult(
                specialist="supervisor",
                findings=[Finding(
                    topic="clarify",
                    text=("That could be two different things, and I want "
                          "the right desk on it: is it "
                          f"(1) {first}, or (2) {second}?"))],
                progress=True)
            state["next"] = "post"
            return state

        if analysis.low_confidence:
            # No clear signal: the care desk takes the turn, with
            # the reason on the routing record — never a
            # transactional desk on a guess.
            intent = Intent.CARE
            if record["decision"] == "route":
                record["decision"] = "low_confidence_care"
        else:
            intent = route(text, self._hint(text, analysis))
        conv.intent = intent
        self._change_desk(state, intent, record)
        state["next"] = "specialist"
        return state

    def _change_desk(self, state: TurnState, intent: Intent,
                     record: dict) -> None:
        """Set the conversation's desk, recording any move. A
        reroute is an audit event (from-desk, to-desk, trigger
        turn), and it retires any proposal the old desk left
        awaiting an answer: a proposal may only ever execute after
        being re-made and re-confirmed at the desk that now owns
        the conversation. Identity, context and the executed-action
        history are the conversation's, not the desk's — they all
        survive the move untouched."""
        conv = state["conv"]
        record["intent"] = intent.value
        previous = conv.specialist_state.get("desk")
        if previous is None or previous == intent.value:
            conv.specialist_state["desk"] = intent.value
            return
        record["rerouted_from"] = previous
        self._audit(conv, "reroute", {
            "from_desk": previous, "to_desk": intent.value,
            "trigger_turn": conv.turns,
            "signals": record.get("signals", {}),
            "runner_up": record.get("runner_up")})
        if conv.pending_action is not None and \
                conv.specialist_state.get("pending_desk") != intent.value:
            self._retire_pending(conv, (
                f"the conversation moved from the {previous} desk to the "
                f"{intent.value} desk; the proposal must be re-made and "
                "re-confirmed there"))
        conv.specialist_state["desk"] = intent.value

    def _retire_pending(self, conv: Conversation, reason: str) -> None:
        """Retire the awaiting proposal without executing it. The
        record stays on the books as superseded — visible in the
        action log and the audit, never executable — and the
        confirmation slate is wiped clean."""
        pending = conv.pending_action
        if pending is None:
            return
        for action_record in self.store.list_actions(conv.conversation_id):
            if action_record.status == "proposed" \
                    and action_record.kind == pending.kind \
                    and action_record.summary == pending.summary:
                action_record.status = "superseded"
                action_record.block_reason = reason
                self.store.save_action(action_record)
                self._audit(conv, "action", {
                    "action_id": action_record.action_id,
                    "status": "superseded", "reason": reason})
                break
        conv.pending_action = None
        conv.specialist_state.pop("pending_desk", None)

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

        if state.get("clarification_resolved") and not result.progress:
            # The customer just answered the clarifying question.
            # That is progress by definition, whatever the desk then
            # manages with the turn — the bookkeeping stays honest
            # in the other direction too: an unanswered clarification
            # gets no such credit, only the commit below.
            result = result.model_copy(update={"progress": True})
            state["result"] = result

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
            if conv.pending_action is not None:
                # A fresh proposal replaces the unanswered one; the
                # old record is retired, never silently overwritten.
                self._retire_pending(
                    conv, "superseded by a new proposal before it was answered")
            record = action_mod.new_action_record(conv, result.proposed_action)
            self.store.save_action(record)
            conv.pending_action = result.proposed_action
            conv.specialist_state["pending_desk"] = conv.intent.value
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
        conv.specialist_state.pop("pending_desk", None)
        # A live procedure walk is abandoned with the conversation —
        # its run record closes with the reason, history intact for
        # the human who picks the case up.
        procedures.abandon_run(self.store, conv, "conversation_escalated")
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
        detail = {"intent": conv.intent.value,
                  "escalated": bool(state.get("escalated")),
                  "identity_state": conv.identity_state.value}
        routing = state.get("routing") or {}
        if routing:
            # The router's evidence rides the turn audit: matched
            # signals, the runner-up, and how the turn was decided.
            detail["routing"] = {
                key: routing[key]
                for key in ("decision", "signals", "scores", "runner_up")
                if key in routing}
            if routing.get("rerouted_from"):
                detail["rerouted_from"] = routing["rerouted_from"]
        self._audit(conv, "turn", detail)
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
