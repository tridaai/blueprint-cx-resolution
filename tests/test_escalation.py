from cx_agent.config import Policy
from cx_agent.escalation import build_packet, evaluate
from cx_agent.models import Conversation
from cx_agent.screening import screen

POLICY = Policy()


def _conv(**kw):
    base = dict(conversation_id="CONV-T")
    base.update(kw)
    return Conversation(**base)


def _eval(conv, text, vulnerable=False):
    return evaluate(conv, screen(text), POLICY, account_vulnerable=vulnerable)


def test_first_human_ask_does_not_escalate_second_does():
    conv = _conv()
    assert not _eval(conv, "let me talk to a person").escalate
    assert conv.human_requests == 1
    decision = _eval(conv, "I said get me a human")
    assert decision.escalate and decision.reason == "customer_requested_human"


def test_threat_fraud_legal_escalate_immediately():
    assert _eval(_conv(), "I'll hurt someone").reason == "safety_threat"
    d = _eval(_conv(), "this is fraud")
    assert d.escalate and d.queue == "fraud"
    assert _eval(_conv(), "my attorney will write to you").reason == "legal_matter"


def test_vulnerable_account_escalates_to_care_queue():
    d = _eval(_conv(), "hello there", vulnerable=True)
    assert d.escalate and d.queue == "vulnerable_care"
    assert d.reason == "vulnerable_customer"


def test_no_progress_escalates_at_the_cap():
    conv = _conv(no_progress_turns=3)
    d = _eval(conv, "anything")
    assert d.escalate and d.reason == "no_progress"


def test_ordinary_turn_does_not_escalate():
    assert not _eval(_conv(), "why is my bill high").escalate


def test_packet_summarises_the_record(seeded_store):
    from cx_agent.models import ActionKind, ProposedAction
    from cx_agent import actions

    conv = _conv(customer_id="CUST-001", account_number="NB-1001", turns=4)
    seeded_store.save_conversation(conv)
    proposal = ProposedAction(
        kind=ActionKind.APPLY_CREDIT, summary="credit $9.99",
        payload={"credit_class": "dispute"})
    record = actions.new_action_record(conv, proposal, status="proposed")
    seeded_store.save_action(record)
    packet = build_packet(conv, seeded_store, "customer_requested_human", "general")
    assert packet.customer_name == "Amara Osei"
    assert packet.account_number == "NB-1001"
    assert packet.identity_state == "unverified"
    assert any("apply_credit" in a for a in packet.attempts)
    assert packet.open_items  # unresolved + unconfirmed action
