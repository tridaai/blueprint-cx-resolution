"""Round 2: mid-conversation rerouting, the clarifying-question
path, and router confidence. The desk can change mid-conversation;
identity, context and the executed-action history cannot be lost
in the move — and a proposal never survives it."""

from decimal import Decimal

from cx_agent.identity import expected_code
from cx_agent.models import Intent
from cx_agent.routing import analyze, resolve_clarification


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _verified_billing_with_credit(service):
    """Drive NB-1001 to a proposed dispute credit at billing."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "why is my bill so high")
    _say(service, conv.conversation_id, "NB-1001 postcode 02134")
    _say(service, conv.conversation_id, expected_code("NB-1001"))
    result = _say(service, conv.conversation_id,
                  "I was charged twice for device protection, that's wrong")
    assert result.action is not None and result.action.status == "proposed"
    return conv


# -- the router's evidence ------------------------------------------------


def test_analyze_names_the_tie_and_the_runner_up():
    tied = analyze("why is my bill wrong? also, I have no signal")
    assert tied.ambiguous
    assert {i.value for i in tied.leaders} == {"billing", "technical"}
    assert tied.signals[Intent.BILLING]  # evidence, not just a score
    clear = analyze("no signal and a wrong charge")
    assert not clear.ambiguous
    assert clear.runner_up == Intent.BILLING


def test_analyze_low_confidence_is_a_weak_single_word():
    assert analyze("fee").low_confidence
    assert analyze("blorp zibble").low_confidence
    assert not analyze("why is my bill so high").low_confidence


def test_resolve_clarification_by_score_and_by_name():
    pair = [Intent.BILLING, Intent.TECHNICAL]
    assert resolve_clarification("the bill", pair) == Intent.BILLING
    assert resolve_clarification("the signal one", pair) == Intent.TECHNICAL
    assert resolve_clarification("the fault, definitely", pair) == Intent.TECHNICAL
    assert resolve_clarification("whatever", pair) is None
    assert resolve_clarification("both are broken", pair) is None


# -- rerouting ---------------------------------------------------------------


def test_reroute_retires_the_old_desks_proposal(service):
    conv = _verified_billing_with_credit(service)
    _say(service, conv.conversation_id,
         "hold on — actually the real problem is I have no signal at home")
    actions = service.store.list_actions(conv.conversation_id)
    assert [a.status for a in actions] == ["superseded"]
    assert "re-confirmed" in actions[0].block_reason
    events = [e for e in service.store.list_audit(conv.conversation_id)
              if e.kind == "reroute"]
    assert len(events) == 1
    assert events[0].detail["from_desk"] == "billing"
    assert events[0].detail["to_desk"] == "technical"
    assert events[0].detail["trigger_turn"] == 5
    turns = [e for e in service.store.list_audit(conv.conversation_id)
             if e.kind == "turn"]
    assert turns[-1].detail.get("rerouted_from") == "billing"
    assert turns[-1].detail["routing"]["runner_up"] is None
    assert turns[-1].detail["routing"]["signals"]["technical"]


def test_a_stale_yes_never_executes_after_a_reroute(service):
    conv = _verified_billing_with_credit(service)
    _say(service, conv.conversation_id,
         "hold on — actually the real problem is I have no signal at home")
    _say(service, conv.conversation_id, "yes, the bars are back")
    executed = [a for a in service.store.list_actions(conv.conversation_id)
                if a.status == "executed"]
    assert executed == []
    assert service.store.get_account("NB-1001").balance_due == Decimal("94.35")


def test_verification_survives_a_reroute(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "why is my bill so high")
    _say(service, conv.conversation_id, "NB-1010 postcode 33101")
    _say(service, conv.conversation_id, expected_code("NB-1010"))
    result = _say(service, conv.conversation_id,
                  "I still can't pay it this month")
    assert result.intent == "account"
    assert result.identity_state == "verified"
    assert "one-time code" not in result.reply
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "executed"
    assert result.identity_state == "verified"


def test_same_desk_follow_up_keeps_the_proposal_alive(service):
    conv = _verified_billing_with_credit(service)
    _say(service, conv.conversation_id, "and is the bill total right?")
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "executed"
    assert service.store.get_account("NB-1001").balance_due == Decimal("84.36")


# -- clarification --------------------------------------------------------------


def test_clarification_asks_once_and_the_answer_routes(service):
    conv = service.start_conversation()
    first = _say(service, conv.conversation_id,
                 "why is my bill wrong? also, I have no signal")
    assert "right desk" in first.reply
    assert "your bill" in first.reply
    assert "a fault with your service" in first.reply
    assert first.intent == "unknown"  # no desk was guessed
    second = _say(service, conv.conversation_id, "the bill, start there")
    assert second.intent == "billing"


def test_an_ignored_clarification_commits_to_care_without_looping(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "why is my bill wrong? also, I have no signal")
    result = _say(service, conv.conversation_id, "whatever, just sort it out")
    assert result.intent == "care"
    assert "right desk" not in result.reply  # no second question
    assert not result.escalated


def test_low_confidence_turn_goes_to_care_not_a_transactional_desk(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id, "fee")
    assert result.intent == "care"
    turns = [e for e in service.store.list_audit(conv.conversation_id)
             if e.kind == "turn"]
    assert turns[-1].detail["routing"]["decision"] == "low_confidence_care"
