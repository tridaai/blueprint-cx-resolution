"""The troubleshooting engine: procedures are data, this walks them.

A procedure is a versioned data record (models.Procedure); this
module is the only walker, and it interprets the record — step
kinds, thresholds, branch tables, safety flags — so changing a
procedure's data changes the walk without touching code. A walk
is a ProcedureRun record in the store: the procedure and version
it pinned, the step it is waiting on, and a StepRecord for
everything that has happened. The conversation's specialist state
holds only the run's id.

Step kinds, as the walker meets them:

- **tool_check** runs immediately — the agent tests a condition
  itself through the TOOLS registry and the result selects the
  path (`branches`, else the next step). Tools read the book; a
  tool with nothing to read reports its neutral result, it never
  invents one.
- **branch** evaluates a named rule (RULES) over the run's
  recorded results and follows `branches`.
- **terminal** ends the walk: resolved, book_callback (a
  CallbackRecord is created with the full transcript), or
  escalate_field_visit (the conversation escalates — an engineer
  visit is a human's booking to make).
- **instruction** and **measurement** wait for the customer.
  An instruction step is answered success / fail / unclear in
  code; a measurement step parses the reported value and compares
  it against the step's threshold in code.

Safety: a step flagged mandatory_safety completes only on
success. A skip attempt or an outright refusal is refused in
code, recorded on the run, and the step stays current; a second
insistence ends the walk and escalates with the record attached.
"""

from __future__ import annotations

import re
import uuid
from decimal import Decimal, InvalidOperation

from .clock import today
from .models import (
    ActionKind,
    CallbackRecord,
    Conversation,
    Finding,
    Procedure,
    ProcedureRun,
    ProcedureStep,
    ProposedAction,
    SpecialistResult,
    StepRecord,
)
from .routing import classify_skip_attempt, classify_step_outcome

# The checks that mean "the symptom is gone" when they pass, as
# opposed to set-up steps (switching Wi-Fi calling on) that pass
# and advance.
RESOLUTION_CHECKS = {"signal_restored", "speed_restored", "call_connected"}

_RESOLVED_TEXT = (
    "That's the fix — your service should stay up now. If it drops "
    "again, quote this conversation and we'll pick up right here.")

_CALLBACK_BASIS = "Guided procedure exhausted without resolution (POL-TECH-01)."

# Postcodes with a known network outage on the book. A coverage
# check at one of these reports the outage instead of clearing
# the area; no seeded customer lives at one — it is the static
# register beside the live one: since round 15 the coverage tool
# also reads the incident feed itself (a live no-signal incident
# covering the customer's postcode), which is the register a
# deployment actually fills.
OUTAGE_POSTCODES: set[str] = {"99999"}

_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Tools — the agent-side checks. Each reads the book through the
# context and returns a short result word the step's branch table
# (or the walker's default) routes on. Each also writes the
# finding the customer hears; a check with nothing to say returns
# no finding.
# ---------------------------------------------------------------------------


def _account(ctx, conv: Conversation, run: ProcedureRun):
    number = run.account_number or conv.account_number
    return ctx.store.get_account(number) if number else None


def _tool_coverage_ok(ctx, conv, run) -> tuple[str, Finding | None]:
    customer = None
    account = _account(ctx, conv, run)
    if account is not None:
        customer = ctx.store.get_customer(account.customer_id)
    # The incident feed is the live outage register (round 15): a
    # no-signal incident still active or monitoring whose scope
    # covers this postcode reports the outage here too. Read off
    # the store directly — the feed's records, no matching rules:
    # this tool answers for one postcode, not for a conversation.
    feed_outage = False
    if customer is not None:
        for incident in ctx.store.list_incidents():
            if incident.service == "no_signal" \
                    and incident.status in ("active", "monitoring") \
                    and (not incident.affected_postcodes
                         or customer.postcode
                         in incident.affected_postcodes):
                feed_outage = True
                break
    if customer is not None and (customer.postcode in OUTAGE_POSTCODES
                                 or feed_outage):
        return "outage", Finding(
            topic="check",
            text=("I've checked coverage at your postcode: there is a "
                  "reported outage there and engineers are already on "
                  "it — the device steps may not bite until it's "
                  "cleared, but let's rule out the phone side too."))
    if customer is None:
        # No account on the conversation means no postcode to check:
        # say so, rather than report a check that never ran. The
        # walk goes on exactly as on a clear result.
        return "clear", Finding(
            topic="check",
            text=("I can't check your area for outages until I know "
                  "your account — if you'd like me to, tell me your "
                  "account number (it starts NB-) and postcode. "
                  "Meanwhile, let's work through the device side."))
    return "clear", Finding(
        topic="check",
        text=("I've checked coverage at your postcode: no outage is "
              "reported there, so let's work through the device side."))


def _tool_allowance_ok(ctx, conv, run) -> tuple[str, Finding | None]:
    account = _account(ctx, conv, run)
    if account is not None:
        plan = ctx.store.get_plan(account.plan_id)
        if plan is not None and plan.data_gb is not None \
                and account.cycle_day >= 28:
            return "near_limit", Finding(
                topic="check",
                text=("I've checked your data allowance: you're near "
                      "the end of the cycle on a limited plan, which "
                      "can slow things down — but let's rule out the "
                      "device side before we call it that."))
    if account is None:
        return "ok", Finding(
            topic="check",
            text=("I can't check your data allowance or your area for "
                  "known issues until I know your account — if you'd "
                  "like me to, tell me your account number (it starts "
                  "NB-) and postcode. Meanwhile, let's try the quick "
                  "fixes."))
    return "ok", Finding(
        topic="check",
        text=("I've checked your data allowance: that isn't what's "
              "slowing you down."))


def _tool_line_test(ctx, conv, run) -> tuple[str, Finding | None]:
    account = _account(ctx, conv, run)
    if account is None:
        return "unknown", None
    result = account.diagnostics.get("line_test", "clear")
    if result == "fault_found":
        return "fault_found", Finding(
            topic="check",
            text=("I've run a line test on your line: it's showing a "
                  "fault on the network side of the line."))
    return "clear", Finding(
        topic="check",
        text=("I've run a line test on your line: it comes back "
              "clear, so the fault is on the home side — let's work "
              "through it."))


TOOLS = {
    "coverage_ok": _tool_coverage_ok,
    "allowance_ok": _tool_allowance_ok,
    "line_test": _tool_line_test,
}


# ---------------------------------------------------------------------------
# Branch rules — named reads over the run's recorded results. A
# rule returns the answer word the step's branch table routes on.
# ---------------------------------------------------------------------------


def _rule_speed_critical(run: ProcedureRun, step: ProcedureStep) -> str:
    """The last reported speed against the step's hard threshold:
    critically low points at the line, not the devices."""
    for record in reversed(run.history):
        if record.kind == "measurement" and record.result:
            try:
                value = Decimal(record.result.split()[0])
            except (InvalidOperation, IndexError):
                continue
            if step.threshold is not None and value < step.threshold:
                return "critical"
            return "normal"
    return "normal"


RULES = {
    "speed_critical": _rule_speed_critical,
}


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


def _steps_by_id(proc: Procedure) -> dict[str, ProcedureStep]:
    return {s.step_id: s for s in proc.steps}


def _next_id(proc: Procedure, step: ProcedureStep) -> str | None:
    if step.next_step:
        return step.next_step
    ids = [s.step_id for s in proc.steps]
    index = ids.index(step.step_id)
    return ids[index + 1] if index + 1 < len(ids) else None


def _record(run: ProcedureRun, step: ProcedureStep, kind: str,
            detail: str, result: str, conv: Conversation) -> None:
    run.history.append(StepRecord(
        step_id=step.step_id, kind=kind, detail=detail, result=result,
        turn=conv.turns))
    run.updated_date = today().isoformat()


def _save(ctx, run: ProcedureRun) -> None:
    ctx.store.save_procedure_run(run)


def _process(ctx, conv: Conversation, run: ProcedureRun, proc: Procedure,
             step_id: str | None, findings: list[Finding]) -> str:
    """Walk the automatic steps from `step_id` — tool checks,
    branches, terminals — until the walk waits on the customer or
    ends. Appends findings as it goes and returns the mode:
    present | resolved | callback | field_visit | escalate."""
    steps = _steps_by_id(proc)
    guard = 0
    while step_id is not None:
        guard += 1
        if guard > 50:  # a data cycle, not a walk — stop rather than spin
            return _dead_end(ctx, conv, run, proc, findings)
        step = steps.get(step_id)
        if step is None:
            break
        if step.kind == "tool_check":
            tool = TOOLS.get(step.check)
            result, finding = tool(ctx, conv, run) if tool else ("unknown", None)
            _record(run, step, "tool_check", step.check, result, conv)
            if finding is not None:
                findings.append(finding)
            step_id = step.branches.get(result) or _next_id(proc, step)
            continue
        if step.kind == "branch":
            rule = RULES.get(step.rule)
            answer = rule(run, step) if rule else "normal"
            _record(run, step, "branch", step.rule, answer, conv)
            step_id = step.branches.get(answer) or _next_id(proc, step)
            continue
        if step.kind == "terminal":
            _record(run, step, "terminal", step.outcome, step.outcome, conv)
            if step.outcome == "resolved":
                return _finish_resolved(ctx, run, findings, step)
            if step.outcome == "escalate_field_visit":
                run.state = "dead_ended"
                run.outcome = "field_visit"
                run.current_step_id = ""
                _save(ctx, run)
                if step.instruction:
                    findings.append(Finding(topic="field_visit",
                                            text=step.instruction))
                return "field_visit"
            return _dead_end(ctx, conv, run, proc, findings)
        # instruction | measurement — the walk waits here.
        run.current_step_id = step.step_id
        _save(ctx, run)
        findings.append(Finding(topic="step", text=step.instruction))
        return "present"
    # The steps ran out with no terminal: the walk is done.
    return _finish_resolved(ctx, run, findings, None)


def _finish_resolved(ctx, run: ProcedureRun, findings: list[Finding],
                     step: ProcedureStep | None) -> str:
    run.state = "resolved"
    run.outcome = "resolved"
    run.current_step_id = ""
    _save(ctx, run)
    text = step.instruction if step is not None and step.instruction \
        else _RESOLVED_TEXT
    findings.append(Finding(topic="resolved", text=text))
    return "resolved"


def _clear_state(conv: Conversation) -> None:
    state = conv.specialist_state.get("technical", {})
    state.pop("run_id", None)
    state.pop("procedure_id", None)


def _dead_end(ctx, conv: Conversation, run: ProcedureRun, proc: Procedure,
              findings: list[Finding]) -> str:
    """The procedure is exhausted without resolution: create the
    callback record with the full transcript, then hand the desk
    the mode — the desk words the proposal (or the identity ask
    that precedes it)."""
    run.state = "dead_ended"
    run.outcome = "callback_booked"
    run.current_step_id = ""
    terminal = StepRecord(step_id="book_callback", kind="terminal",
                          detail="callback", result="book_callback",
                          turn=conv.turns)
    run.history.append(terminal)
    _save(ctx, run)
    callback = CallbackRecord(
        callback_id=f"CBK-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=run.account_number or conv.account_number or "",
        procedure_id=proc.procedure_id,
        procedure_version=run.procedure_version,
        run_id=run.run_id,
        reason=(f"The {proc.title} procedure was worked through in "
                "full and service was not restored."),
        status="proposed",
        transcript=list(run.history),
        created_date=today().isoformat(),
    )
    ctx.store.save_callback(callback)
    state = conv.specialist_state.setdefault("technical", {})
    state["callback_id"] = callback.callback_id
    state["run_id"] = run.run_id
    state["procedure_id"] = proc.procedure_id
    return "callback"


def settle_callback(store, callback_id: str, status: str,
                    action_id: str = "",
                    account_number: str = "") -> None:
    """Settle a callback record when its booking proposal is
    answered — booked (with the executed action's id), declined,
    or blocked at execution — the way a dispute case settles.
    The record is created at the dead-end, which can precede
    identification; the account the (by then verified)
    conversation belongs to is stamped here if the record
    doesn't carry one yet, so the settled callback stands on
    the account it was promised to (round 16)."""
    callback = store.get_callback(callback_id)
    if callback is None:
        return
    callback.status = status
    if action_id:
        callback.action_id = action_id
    if account_number and not callback.account_number:
        callback.account_number = account_number
    store.save_callback(callback)


def abandon_run(store, conv: Conversation, outcome: str) -> None:
    """Close out the conversation's active run when the
    conversation itself ends another way (an escalation): the run
    is abandoned, with the reason on the record — its history
    stays readable for whoever picks the case up."""
    state = conv.specialist_state.get("technical", {})
    run_id = state.get("run_id")
    if not run_id:
        return
    run = store.get_procedure_run(run_id)
    if run is not None and run.state == "active":
        run.state = "abandoned"
        run.outcome = outcome
        run.current_step_id = ""
        run.updated_date = today().isoformat()
        store.save_procedure_run(run)


def start_run(ctx, conv: Conversation, proc: Procedure) -> SpecialistResult:
    """Open a run on `proc` (pinned at its current version) and
    walk to the first customer step, or wherever the data leads."""
    run = ProcedureRun(
        run_id=f"RUN-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=conv.account_number or "",
        procedure_id=proc.procedure_id,
        procedure_version=proc.version,
        created_date=today().isoformat(),
        updated_date=today().isoformat(),
    )
    _save(ctx, run)
    state = conv.specialist_state.setdefault("technical", {})
    state["run_id"] = run.run_id
    state["procedure_id"] = proc.procedure_id
    findings = [Finding(
        topic="procedure_start",
        text=(f"Let's work through the {proc.title} steps in order — "
              "they fix most cases like this."))]
    first = proc.steps[0].step_id if proc.steps else None
    mode = _process(ctx, conv, run, proc, first, findings)
    return _result_for_mode(ctx, conv, run, proc, mode, findings)


def _result_for_mode(ctx, conv: Conversation, run: ProcedureRun,
                     proc: Procedure, mode: str,
                     findings: list[Finding]) -> SpecialistResult:
    if mode == "present":
        return SpecialistResult(specialist="technical", findings=findings,
                                progress=True)
    if mode == "resolved":
        _clear_state(conv)
        return SpecialistResult(specialist="technical", findings=findings,
                                progress=True, resolved=True)
    if mode == "field_visit":
        _clear_state(conv)
        return SpecialistResult(
            specialist="technical", findings=findings, progress=True,
            escalate_reason="field_visit_required",
            escalate_queue="general")
    if mode == "callback":
        return _callback_result(ctx, conv, run, proc, findings)
    return SpecialistResult(specialist="technical", findings=findings,
                            progress=True)


def _callback_result(ctx, conv: Conversation, run: ProcedureRun,
                     proc: Procedure,
                     findings: list[Finding]) -> SpecialistResult:
    """The dead-end proposal: a technician callback, with the
    callback record already carrying the transcript. Without an
    identified account the desk asks for identity first and the
    proposal follows on the next turn (the want_callback state)."""
    state = conv.specialist_state.setdefault("technical", {})
    callback_id = state.get("callback_id", "")
    if not conv.account_number:
        state["want_callback"] = proc.procedure_id
        result = SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="identity",
                text=("We've worked through the guided steps without "
                      "restoring service. I'd like to book a technician "
                      "callback — I can help with that. So I can pull up "
                      "the right account, tell me your account number "
                      "(it starts NB-) and the postcode on the account."))],
            needs_identity=True, progress=True)
        return result
    findings.append(Finding(
        topic="steps_exhausted",
        text=(f"We've worked through the {proc.title} steps and the "
              "problem is still there, so this needs a technician.")))
    return SpecialistResult(
        specialist="technical", findings=findings,
        proposed_action=ProposedAction(
            kind=ActionKind.BOOK_CALLBACK,
            summary="book a technician callback within one working day",
            payload={"reason": proc.procedure_id,
                     "callback_id": callback_id,
                     "run_id": run.run_id},
            basis=_CALLBACK_BASIS),
        citations=[d["doc_id"] for d in
                   ctx.retriever.search(
                       "troubleshooting steps callback technician",
                       limit=2, section="technical")],
        progress=True)


def callback_proposal(ctx, conv: Conversation,
                      proc: Procedure) -> SpecialistResult:
    """The proposal that follows identity on the want_callback
    path: the callback record was created at the dead-end; this
    turn just proposes its booking."""
    state = conv.specialist_state.setdefault("technical", {})
    return SpecialistResult(
        specialist="technical",
        findings=[Finding(
            topic="steps_exhausted",
            text=("Thank you — identity confirmed. As promised, this "
                  "one goes to a technician rather than more "
                  "guesswork."))],
        proposed_action=ProposedAction(
            kind=ActionKind.BOOK_CALLBACK,
            summary="book a technician callback within one working day",
            payload={"reason": proc.procedure_id,
                     "callback_id": state.get("callback_id", ""),
                     "run_id": state.get("run_id", "")},
            basis=_CALLBACK_BASIS),
        citations=[d["doc_id"] for d in
                   ctx.retriever.search(
                       "troubleshooting steps callback technician",
                       limit=2, section="technical")],
        progress=True)


def _repeat(step: ProcedureStep) -> SpecialistResult:
    if step.kind == "measurement":
        text = (f"I need the reading itself to decide the next step: "
                f"{step.instruction}")
    else:
        text = (f"Just to be sure we're on the same step: "
                f"{step.instruction} Tell me yes if it worked, or no "
                "if nothing changed.")
    return SpecialistResult(
        specialist="technical",
        findings=[Finding(topic="step_repeat", text=text)],
        progress=False)


def _safety_refusal(ctx, conv: Conversation, run: ProcedureRun,
                    proc: Procedure, step: ProcedureStep,
                    text: str) -> SpecialistResult:
    """A mandatory safety step was refused or skipped-at. The
    refusal is refused in code and recorded; the step stays
    current. A second insistence ends the walk and escalates."""
    run.safety_refusals += 1
    _record(run, step, "safety_refusal", step.check or step.step_id,
            "refused", conv)
    if run.safety_refusals >= 2:
        run.state = "dead_ended"
        run.outcome = "safety_escalation"
        run.current_step_id = ""
        _save(ctx, run)
        _clear_state(conv)
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="safety_escalation",
                text=("That step is a safety requirement, and I can't "
                      "work around it or ask you to — so I'm handing "
                      "this to a person, with the full record of what "
                      "we covered attached."))],
            progress=True,
            escalate_reason="safety_step_refused",
            escalate_queue="general")
    _save(ctx, run)
    return SpecialistResult(
        specialist="technical",
        findings=[Finding(
            topic="safety_refusal",
            text=("I can't skip that one — it's a safety step, and it "
                  "has to happen before anything else we try. "
                  f"{step.instruction}"))],
        progress=False)


def advance(ctx, conv: Conversation, run: ProcedureRun,
            text: str) -> SpecialistResult:
    """The customer's reply at the run's current step."""
    proc = ctx.store.get_procedure_version(run.procedure_id,
                                            run.procedure_version)
    if proc is None:  # the pinned version is gone — cannot happen
        # through the store's own writes; fail safe to a dead end.
        proc = ctx.store.get_procedure(run.procedure_id)
    steps = _steps_by_id(proc)
    step = steps.get(run.current_step_id)
    if step is None:
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="unknown_fault",
                text=("Let's pick the troubleshooting up again — tell "
                      "me what the phone is doing now and I'll restart "
                      "the right steps."))],
            progress=False)

    if step.mandatory_safety and (
            classify_skip_attempt(text)
            or classify_step_outcome(text) == "fail"):
        return _safety_refusal(ctx, conv, run, proc, step, text)

    if step.kind == "measurement":
        match = _NUMBER.search(text)
        if match is None:
            return _repeat(step)
        value = Decimal(match.group(1))
        _record(run, step, "measurement", step.metric,
                f"{value.normalize()} {step.unit}", conv)
        _save(ctx, run)
        findings: list[Finding] = []
        if step.threshold is not None and value < step.threshold:
            findings.append(Finding(
                topic="measurement",
                text=(f"{value.normalize()} {step.unit} is below the "
                      f"{step.threshold.normalize()} {step.unit} a "
                      "healthy line should hold — that points at a "
                      "fault, not at anything you've done.")))
            target = step.below_step or _next_id(proc, step)
        else:
            target = step.at_or_above_step or _next_id(proc, step)
        mode = _process(ctx, conv, run, proc, target, findings)
        return _result_for_mode(ctx, conv, run, proc, mode, findings)

    outcome = classify_step_outcome(text)
    if outcome == "unclear":
        return _repeat(step)
    _record(run, step, "instruction", step.check or step.step_id,
            outcome, conv)
    _save(ctx, run)
    findings = []
    if outcome == "success":
        if step.check in RESOLUTION_CHECKS:
            _finish_resolved(ctx, run, findings, None)
            _clear_state(conv)
            return SpecialistResult(specialist="technical",
                                    findings=findings, progress=True,
                                    resolved=True)
        mode = _process(ctx, conv, run, proc,
                        _next_id(proc, step), findings)
        return _result_for_mode(ctx, conv, run, proc, mode, findings)
    # outcome == fail
    if step.on_fail == "next":
        target = _next_id(proc, step)
        if target is not None:
            findings.append(Finding(
                topic="step_failed",
                text="Understood — that step didn't do it. Next one:"))
            mode = _process(ctx, conv, run, proc, target, findings)
            return _result_for_mode(ctx, conv, run, proc, mode, findings)
    if step.on_fail == "escalate":
        run.state = "dead_ended"
        run.outcome = "escalated"
        run.current_step_id = ""
        _save(ctx, run)
        _clear_state(conv)
        return SpecialistResult(
            specialist="technical", findings=findings, progress=True,
            escalate_reason="procedure_step_failed",
            escalate_queue="general")
    mode = _dead_end(ctx, conv, run, proc, findings)
    return _result_for_mode(ctx, conv, run, proc, mode, findings)


def match_procedure(store, text: str) -> Procedure | None:
    """The procedure whose symptom the message names. Symptom
    phrases are procedure data; the longest matching phrase wins,
    so a specific symptom beats a generic one."""
    lowered = text.lower()
    best: tuple[int, Procedure] | None = None
    for proc in store.list_procedures():
        for phrase in proc.symptom_phrases:
            if phrase and phrase in lowered:
                if best is None or len(phrase) > best[0]:
                    best = (len(phrase), proc)
    return best[1] if best else None
