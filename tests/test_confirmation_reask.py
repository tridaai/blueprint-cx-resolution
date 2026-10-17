"""Regression tests for the signal-less non-answer defect
(Round 9, found by the persona eval expansion).

A reply the confirmation vocabulary cannot read — "um, yes I
think so", a fragment — carries no desk signal of any kind. Two
places in the conversation machinery treated it as a brand-new
turn, and the decision the customer was mid-answer to died
silently both times:

- the supervisor routed it as a fresh intent; the desk change
  retired the pending proposal as superseded, so a credit the
  customer went on to confirm with a plain "yes" never executed;
- the roaming bill-shock guard spent its awaiting-decision
  marker on it, so the cheaper pass the customer was deciding
  on could never be bought.

The rule now, in both places: a signal-less non-answer is not a
new topic — the question is asked again, plainly, and the
decision stays open. A reply carrying any real signal still
routes and reroutes exactly as before, and three non-answers in
a row still escalate as no progress.
"""

from decimal import Decimal

from cx_agent.identity import expected_code


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _verified_dispute(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "why is my bill so high")
    _say(service, conv.conversation_id, "NB-1001 postcode 02134")
    _say(service, conv.conversation_id, expected_code("NB-1001"))
    result = _say(
        service, conv.conversation_id,
        "I dispute two charges: you charged me twice for device "
        "protection, and the handset instalment — I don't recognise it")
    assert result.action is not None and result.action.status == "proposed"
    return conv


def test_hesitant_answer_reasks_and_the_proposal_still_executes(service):
    conv = _verified_dispute(service)
    result = _say(service, conv.conversation_id, "um, yes I think so")
    assert not result.escalated
    assert "shall I go ahead" in result.reply
    # Nothing executed, nothing retired: the proposal is still open.
    actions = service.store.list_actions(conv.conversation_id)
    assert [a.status for a in actions] == ["proposed"]
    result = _say(service, conv.conversation_id, "yes")
    assert result.action.status == "executed"
    assert service.store.get_account("NB-1001").balance_due == Decimal("84.36")


def test_a_real_topic_change_still_retires_the_proposal(service):
    conv = _verified_dispute(service)
    result = _say(
        service, conv.conversation_id,
        "hold on — actually the real problem is I have no signal at home")
    assert result.intent == "technical"
    actions = service.store.list_actions(conv.conversation_id)
    assert [a.status for a in actions] == ["superseded"]


def test_three_non_answers_escalate_with_the_proposal_open(service):
    conv = _verified_dispute(service)
    for text in ("um", "hmm"):
        result = _say(service, conv.conversation_id, text)
        assert not result.escalated
        assert "shall I go ahead" in result.reply
    result = _say(service, conv.conversation_id, "dunno really")
    assert result.escalated
    assert result.escalation_reason == "no_progress"
    # Never executed behind the customer's back, never destroyed:
    # the human picks the conversation up with the proposal open.
    actions = service.store.list_actions(conv.conversation_id)
    assert [a.status for a in actions] == ["proposed"]
    assert service.store.get_account("NB-1001").balance_due == Decimal("94.35")


def _at_the_guard(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "I'm travelling to Spain for 8 days — what will roaming "
         "cost me, is a pass cheaper?")
    _say(service, conv.conversation_id, "NB-1007 postcode 02139")
    result = _say(service, conv.conversation_id, expected_code("NB-1007"))
    assert "$61.20" in result.reply  # the guard's projection led
    return conv


def test_hesitant_answer_at_the_roaming_guard_reasks(service):
    conv = _at_the_guard(service)
    result = _say(service, conv.conversation_id, "um, yes I think so")
    assert not result.escalated
    assert "shall I add the Roam Pass" in result.reply
    result = _say(service, conv.conversation_id, "yes")
    assert result.action is not None and result.action.status == "proposed"
    result = _say(service, conv.conversation_id, "yes")
    assert result.action.status == "executed"
    assert service.store.get_account("NB-1007").balance_due == Decimal("75.00")


def test_new_destination_at_the_guard_still_moves_on(service):
    conv = _at_the_guard(service)
    result = _say(service, conv.conversation_id,
                  "actually, what would France cost for 5 days?")
    # Not held at the old decision: the new trip gets priced.
    assert "France" in result.reply or "france" in result.reply.lower()
    assert "shall I add the Roam Pass" not in result.reply
