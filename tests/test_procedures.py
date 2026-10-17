"""The troubleshooting engine (round 4): procedures as versioned
data, the run as a stored record, measurements and branches
decided in code, the dead-end callback with its transcript, and
mandatory safety steps that refuse to be skipped."""

from copy import deepcopy
from decimal import Decimal

from cx_agent.dataset import build_procedures
from cx_agent.identity import expected_code


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _identify(service, conv_id, account, postcode):
    _say(service, conv_id, f"{account} postcode {postcode}")
    return _say(service, conv_id, expected_code(account))


def _run(service, conv_id):
    runs = service.store.list_procedure_runs(conversation_id=conv_id)
    assert len(runs) == 1
    return runs[0]


def _start_broadband(service, account="NB-1006", postcode="30301"):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "my broadband keeps dropping")
    result = _identify(service, conv.conversation_id, account, postcode)
    return conv, result


# ---------------------------------------------------------------------------
# Branch correctness — the tool's result selects the path
# ---------------------------------------------------------------------------


def test_line_test_fault_branches_straight_to_a_field_visit(service):
    conv, result = _start_broadband(service, "NB-1005", "94107")
    assert result.escalated
    assert result.escalation_reason == "field_visit_required"
    run = _run(service, conv.conversation_id)
    assert run.state == "dead_ended" and run.outcome == "field_visit"
    assert [(h.step_id, h.result) for h in run.history] == [
        ("BB-1", "fault_found"), ("BB-6", "escalate_field_visit")]
    escalations = service.list_escalations()
    assert escalations[-1].queue == "general"


def test_line_test_clear_proceeds_to_the_safety_step(service):
    conv, result = _start_broadband(service)
    assert not result.escalated
    assert "unplug" in result.reply
    run = _run(service, conv.conversation_id)
    assert run.state == "active" and run.current_step_id == "BB-2"
    assert [(h.step_id, h.result) for h in run.history] == [("BB-1", "clear")]


# ---------------------------------------------------------------------------
# Measurements — the reported value, compared in code, decides
# ---------------------------------------------------------------------------


def _to_measurement(service, conv_id):
    _say(service, conv_id, "done, it's unplugged")


def test_measurement_below_threshold_takes_the_fault_path(service):
    conv, _ = _start_broadband(service)
    _to_measurement(service, conv.conversation_id)
    result = _say(service, conv.conversation_id, "it says 18")
    assert "Reseat the cable" in result.reply
    run = _run(service, conv.conversation_id)
    assert run.current_step_id == "BB-5"
    assert [h.step_id for h in run.history] == ["BB-1", "BB-2", "BB-3", "BB-4"]
    branch = run.history[-1]
    assert branch.kind == "branch" and branch.result == "normal"
    result = _say(service, conv.conversation_id, "yes, much better now")
    assert result.status == "resolved"
    assert _run(service, conv.conversation_id).state == "resolved"


def test_measurement_critically_low_goes_straight_to_a_field_visit(service):
    conv, _ = _start_broadband(service)
    _to_measurement(service, conv.conversation_id)
    result = _say(service, conv.conversation_id, "it says 6")
    assert result.escalated
    assert result.escalation_reason == "field_visit_required"
    run = _run(service, conv.conversation_id)
    branch = next(h for h in run.history if h.kind == "branch")
    assert branch.step_id == "BB-4" and branch.result == "critical"
    assert run.history[-1].step_id == "BB-6"


def test_measurement_at_or_above_threshold_resolves(service):
    conv, _ = _start_broadband(service)
    _to_measurement(service, conv.conversation_id)
    result = _say(service, conv.conversation_id, "it says 42")
    assert result.status == "resolved"
    assert "healthy" in result.reply
    run = _run(service, conv.conversation_id)
    assert run.state == "resolved"
    assert [h.step_id for h in run.history] == ["BB-1", "BB-2", "BB-3", "BB-7"]
    measurement = run.history[2]
    assert measurement.kind == "measurement"
    assert measurement.result == "42 Mbps"


def test_measurement_without_a_number_repeats_the_step(service):
    conv, _ = _start_broadband(service)
    _to_measurement(service, conv.conversation_id)
    result = _say(service, conv.conversation_id, "dunno")
    assert "reading" in result.reply
    run = _run(service, conv.conversation_id)
    assert run.current_step_id == "BB-3"
    assert [h.step_id for h in run.history] == ["BB-1", "BB-2"]


# ---------------------------------------------------------------------------
# The dead end — a callback record with the full transcript
# ---------------------------------------------------------------------------


def _walk_no_signal_to_the_end(service, conv_id):
    _say(service, conv_id, "I have no signal at all")
    _say(service, conv_id, "no, still nothing")
    _say(service, conv_id, "nope, still no signal")
    return _say(service, conv_id, "no, that didn't work either")


def test_dead_end_callback_carries_the_full_transcript(service):
    conv = service.start_conversation()
    _walk_no_signal_to_the_end(service, conv.conversation_id)
    _say(service, conv.conversation_id, "NB-1004 postcode 60614")
    proposed = _say(service, conv.conversation_id, expected_code("NB-1004"))
    assert proposed.action is not None
    assert proposed.action.status == "proposed"
    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"

    callbacks = service.store.list_callbacks(
        conversation_id=conv.conversation_id)
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.status == "booked"
    assert callback.action_id == done.action.action_id
    assert callback.procedure_id == "PROC-NOSIGNAL"
    assert [s.step_id for s in callback.transcript] == [
        "NS-1", "NS-2", "NS-3", "NS-4", "book_callback"]
    assert [s.result for s in callback.transcript] == [
        "clear", "fail", "fail", "fail", "book_callback"]
    run = _run(service, conv.conversation_id)
    assert run.state == "dead_ended" and run.outcome == "callback_booked"


def test_dead_end_before_identity_parks_a_proposed_callback(service):
    conv = service.start_conversation()
    result = _walk_no_signal_to_the_end(service, conv.conversation_id)
    assert "technician callback" in result.reply
    callbacks = service.store.list_callbacks(
        conversation_id=conv.conversation_id)
    assert len(callbacks) == 1
    assert callbacks[0].status == "proposed"
    assert callbacks[0].account_number == ""
    assert len(callbacks[0].transcript) == 5


def test_declined_callback_settles_the_record(service):
    conv = service.start_conversation()
    _walk_no_signal_to_the_end(service, conv.conversation_id)
    _say(service, conv.conversation_id, "NB-1004 postcode 60614")
    _say(service, conv.conversation_id, expected_code("NB-1004"))
    result = _say(service, conv.conversation_id, "no, leave it")
    assert result.action.status == "declined"
    callback = service.store.list_callbacks(
        conversation_id=conv.conversation_id)[0]
    assert callback.status == "declined"
    account = service.store.get_account("NB-1004")
    assert account.open_case is None  # nothing was booked


# ---------------------------------------------------------------------------
# Interruption and resumption — the run survives a change of desk
# ---------------------------------------------------------------------------


def test_procedure_resumes_at_its_step_after_a_billing_detour(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "I have no signal at all")
    _say(service, conv.conversation_id, "no, still nothing")
    detour = _say(service, conv.conversation_id,
                  "actually, why is my bill so high this month?")
    assert detour.intent == "billing"
    back = _say(service, conv.conversation_id,
                "ok back to the signal — still nothing")
    assert back.intent == "technical"
    assert "Reset network settings" in back.reply
    run = _run(service, conv.conversation_id)
    assert run.state == "active"
    assert run.current_step_id == "NS-4"
    # Completed steps were not re-asked or re-walked: the history
    # is exactly the steps that happened, in order.
    assert [h.step_id for h in run.history] == ["NS-1", "NS-2", "NS-3"]
    assert [h.result for h in run.history] == ["clear", "fail", "fail"]


def test_escalation_abandons_the_live_run_with_its_history(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "I have no signal")
    _say(service, conv.conversation_id, "let me talk to a real person")
    result = _say(service, conv.conversation_id, "get me a human")
    assert result.escalated
    run = _run(service, conv.conversation_id)
    assert run.state == "abandoned"
    assert run.outcome == "conversation_escalated"
    assert [h.step_id for h in run.history] == ["NS-1"]


# ---------------------------------------------------------------------------
# Mandatory safety steps
# ---------------------------------------------------------------------------


def test_safety_step_refuses_a_skip_and_records_it(service):
    conv, _ = _start_broadband(service)
    result = _say(service, conv.conversation_id,
                  "skip that step, just fix it")
    assert not result.escalated
    assert "safety step" in result.reply
    run = _run(service, conv.conversation_id)
    assert run.safety_refusals == 1
    assert run.current_step_id == "BB-2"  # the step did not move
    assert run.state == "active"
    # The walk continues only through the step itself.
    onward = _say(service, conv.conversation_id, "done — unplugged it")
    assert "speed test" in onward.reply
    assert _run(service, conv.conversation_id).current_step_id == "BB-3"


def test_safety_step_refuses_an_outright_no(service):
    conv, _ = _start_broadband(service)
    result = _say(service, conv.conversation_id, "no, I won't do that")
    # "won't do that" is both a fail word and a refusal — at a
    # mandatory step the refusal reading wins, in code.
    assert "safety step" in result.reply
    assert _run(service, conv.conversation_id).safety_refusals == 1


def test_safety_step_second_insistence_escalates_with_the_record(service):
    conv, _ = _start_broadband(service)
    _say(service, conv.conversation_id, "skip that step, just fix it")
    result = _say(service, conv.conversation_id,
                  "I said skip it, just fix it")
    assert result.escalated
    assert result.escalation_reason == "safety_step_refused"
    run = _run(service, conv.conversation_id)
    assert run.state == "dead_ended"
    assert run.outcome == "safety_escalation"
    assert run.safety_refusals == 2
    assert [h.kind for h in run.history] == [
        "tool_check", "safety_refusal", "safety_refusal"]


# ---------------------------------------------------------------------------
# Procedures are data — versions pin, and data alone moves the walk
# ---------------------------------------------------------------------------


def _procedure(procedure_id):
    return next(p for p in build_procedures()
                if p.procedure_id == procedure_id)


def test_a_run_pins_the_procedure_version_it_started_on(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "I have no signal at all")
    revised = deepcopy(_procedure("PROC-NOSIGNAL"))
    revised.version = 2
    for step in revised.steps:
        if step.step_id == "NS-3":
            step.instruction = "REVISED STEP THREE — swap the SIM."
    service.store.save_procedure(revised)
    assert service.store.get_procedure("PROC-NOSIGNAL").version == 2

    result = _say(service, conv.conversation_id, "no, still nothing")
    assert "reseat it firmly" in result.reply  # v1 words, not v2
    assert "REVISED" not in result.reply
    run = _run(service, conv.conversation_id)
    assert run.procedure_version == 1

    fresh = service.start_conversation()
    _say(service, fresh.conversation_id, "I have no signal at all")
    result = _say(service, fresh.conversation_id, "no, still nothing")
    assert "REVISED STEP THREE" in result.reply


def test_changing_procedure_data_alone_moves_the_walker(service):
    # Raise the healthy-speed threshold in the data (v2) and the
    # same reported reading takes the fault path — no code moved.
    revised = deepcopy(_procedure("PROC-BROADBAND"))
    revised.version = 2
    for step in revised.steps:
        if step.step_id == "BB-3":
            step.threshold = Decimal("50")
    service.store.save_procedure(revised)

    conv, _ = _start_broadband(service)
    _to_measurement(service, conv.conversation_id)
    result = _say(service, conv.conversation_id, "it says 42")
    assert "Reseat the cable" in result.reply
    assert _run(service, conv.conversation_id).current_step_id == "BB-5"
