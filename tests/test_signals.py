"""Round 11: conversation signals — computed in code from the
conversation's own records, interventions fired by rule, and the
records that persist per turn."""

from cx_agent import signals
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import AuditEvent, Conversation, Message

POLICY = Policy()
OFFER = "bring in a person now"


def _conv(texts, **kw):
    conv = Conversation(conversation_id="CONV-T", **kw)
    for i, text in enumerate(texts, start=1):
        conv.messages.append(Message(role="customer", text=text, turn=i))
    conv.turns = len(texts)
    return conv


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


# -- the pure signal functions --------------------------------------


def test_lexicon_scores_weighted_entries_once_per_message():
    assert signals.frustration_score("hello, all good here") == 0
    assert signals.frustration_score("this is getting annoying") == 1
    assert signals.frustration_score(
        "this is frustrating and ridiculous") == 4
    # An entry repeated in one message still counts once.
    assert signals.frustration_score("frustrating, frustrating") == 2


def test_rising_needs_a_build_not_a_plateau_or_a_passed_peak():
    assert signals.is_rising([0, 1, 4])
    assert signals.is_rising([1, 2])
    assert not signals.is_rising([0, 2, 2])     # plateau
    assert not signals.is_rising([3, 1, 2])     # peak already passed
    assert not signals.is_rising([0, 0, 1])     # too weak to be a rise
    assert not signals.is_rising([4])          # one turn is not a trend
    assert not signals.is_rising([])


def test_trend_words():
    assert signals.frustration_trend([]) == "calm"
    assert signals.frustration_trend([0, 0]) == "calm"
    assert signals.frustration_trend([0, 1, 4]) == "rising"
    assert signals.frustration_trend([0, 1, 4, 0]) == "easing"
    assert signals.frustration_trend([2, 2]) == "steady"


def test_confusion_explicit_repeat_and_none():
    conv = _conv(["How do disputes work?"])
    conv.specialist_state["signals"] = {"explained_topic": "POL-BILL-01"}
    confused, topic, via = signals.detect_confusion(
        "I don't understand — what do you mean?", conv, POLICY)
    assert confused and topic == "POL-BILL-01" and via == "explicit"

    conv = _conv(["How do disputes work if a charge is wrong?",
                  "How do disputes work if a charge is wrong?"])
    confused, _topic, via = signals.detect_confusion(
        "How do disputes work if a charge is wrong?", conv, POLICY)
    assert confused and via == "repeat"

    conv = _conv(["How do disputes work?"])
    confused, _topic, via = signals.detect_confusion(
        "And what are the opening hours?", conv, POLICY)
    assert not confused and via == ""


def test_effort_counts_each_cost_once():
    conv = _conv(["my bill is wrong", "my bill is wrong",
                  "account NB-1001 postcode 02134"],
                 identity_attempts=2)
    audit = [AuditEvent(event_id="E1", conversation_id="CONV-T",
                        kind="reroute", detail={}, turn=2)]
    parts = signals.effort_breakdown(conv, audit, POLICY)
    # 3 turns + 2 identity attempts + 1 reroute + 1 repeated ask.
    assert parts == {"turns": 3, "identity_attempts": 2, "reroutes": 1,
                     "repeats": 1, "score": 7}


# -- the frustration offer -------------------------------------------


def _frustration_path(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    r1 = _say(service, cid, "My bill has a charge I don't recognise.")
    r2 = _say(service, cid,
              "Account NB-9999, postcode 00000. This is getting annoying.")
    r3 = _say(service, cid,
              "I already told you - NB-9999, 00000. This is frustrating "
              "and ridiculous.")
    return cid, (r1, r2, r3)


def test_frustration_offer_fires_at_the_rules_turn(service):
    cid, (_r1, r2, r3) = _frustration_path(service)
    assert OFFER not in r2.reply          # rising needs the build
    assert OFFER in r3.reply              # fires exactly here
    assert not r3.escalated               # an offer, not a transfer
    records = service.store.list_signal_records(cid)
    assert [r.frustration_score for r in records] == [0, 1, 4]
    assert records[-1].frustration_rising is True
    assert records[-1].interventions == [
        {"kind": "frustration_human_offer", "outcome": "offered"}]


def test_declined_frustration_offer_is_never_made_again(service):
    cid, _ = _frustration_path(service)
    r4 = _say(service, cid,
              "No thanks, I'll stay here. My account is NB-1001 and "
              "the postcode is 02134.")
    assert not r4.escalated and OFFER not in r4.reply
    r5 = _say(service, cid,
              "111111 - and honestly I'm furious, this is unacceptable.")
    assert not r5.escalated and OFFER not in r5.reply
    fired = [i for r in service.store.list_signal_records(cid)
             for i in r.interventions]
    assert fired == [
        {"kind": "frustration_human_offer", "outcome": "offered"},
        {"kind": "frustration_human_offer", "outcome": "declined"}]


def test_accepted_frustration_offer_escalates_with_its_own_reason(service):
    cid, _ = _frustration_path(service)
    r4 = _say(service, cid, "Yes please.")
    assert r4.escalated
    assert r4.escalation_reason == "early_human_offer_accepted"
    records = service.store.list_signal_records(cid)
    assert records[-1].interventions == [
        {"kind": "frustration_human_offer", "outcome": "accepted"}]


def test_frustration_without_no_progress_fires_nothing(service):
    # Angry, but helped at every step: the offer's rule needs
    # rising frustration AND a turn that made no progress.
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "This is annoying - why is my bill so high?")
    _say(service, cid, "My account is NB-1001 and the postcode is 02134")
    r3 = _say(service, cid,
              "This is frustrating and ridiculous. "
              + expected_code("NB-1001"))
    assert OFFER not in r3.reply
    assert all(not r.interventions
               for r in service.store.list_signal_records(cid))


# -- the identity-effort offer -----------------------------------------


def _identity_path(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "I need to update my card.")
    _say(service, cid, "My account is NB-1002 and the postcode is 90210.")
    r3 = _say(service, cid, "999999")
    r4 = _say(service, cid, "888888")
    return cid, r3, r4


def test_identity_offer_fires_at_the_threshold_attempt(service):
    _cid, r3, r4 = _identity_path(service)
    assert OFFER not in r3.reply          # one failed code: keep trying
    assert OFFER in r4.reply              # the threshold attempt: offer
    assert not r4.escalated


def test_identity_offer_lapses_when_the_customer_verifies(service):
    cid, _r3, _r4 = _identity_path(service)
    r5 = _say(service, cid, expected_code("NB-1002"))
    assert not r5.escalated
    assert r5.identity_state == "verified"
    fired = [i for r in service.store.list_signal_records(cid)
             for i in r.interventions]
    assert fired == [
        {"kind": "identity_effort_human_offer", "outcome": "offered"},
        {"kind": "identity_effort_human_offer", "outcome": "lapsed"}]


def test_identity_offer_accepted_escalates(service):
    cid, _r3, _r4 = _identity_path(service)
    r5 = _say(service, cid, "Yes please.")
    assert r5.escalated
    assert r5.escalation_reason == "early_human_offer_accepted"


def test_third_wrong_code_still_locks_instead_of_offering(service):
    # The offer never replaces the safety rule: ignore it, fail a
    # third code, and verification locks and escalates as before.
    cid, _r3, _r4 = _identity_path(service)
    r5 = _say(service, cid, "777777")
    assert r5.escalated
    assert r5.escalation_reason == "identity_verification_failed"


# -- second explanation forms -------------------------------------------


def test_second_form_served_once_then_not_again(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    r1 = _say(service, cid, "What time do your stores open on Saturday?")
    assert "9am to 6pm" in r1.reply
    r2 = _say(service, cid, "I don't understand - what do you mean?")
    assert "The short version:" in r2.reply
    assert "Northbeam stores are open" not in r2.reply
    r3 = _say(service, cid, "I still don't understand.")
    assert "The short version:" not in r3.reply  # never the same form twice
    records = service.store.list_signal_records(cid)
    assert records[1].confusion and records[1].confusion_topic == "POL-CARE-01"
    assert records[1].interventions == [
        {"kind": "second_form:POL-CARE-01", "outcome": "applied"}]
    assert records[2].interventions == []


def test_billing_second_form_is_summary_first(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "Why is my bill so high this month?")
    _say(service, cid, "My account is NB-1001 and the postcode is 02134")
    r3 = _say(service, cid, expected_code("NB-1001"))
    assert "Here is what makes it up:" in r3.reply      # the first form
    r4 = _say(service, cid, "I don't understand my bill at all.")
    assert "In one line:" in r4.reply                  # the second form
    assert "Here is what makes it up:" not in r4.reply
    assert "$94.35" in r4.reply  # the customer's own figures, re-ordered
    records = service.store.list_signal_records(cid)
    assert records[-1].interventions == [
        {"kind": "second_form:bill", "outcome": "applied"}]


def test_repeated_question_also_reaches_the_second_form(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "What time do your stores open on Saturday?")
    r2 = _say(service, cid, "What time do your stores open on Saturday?")
    assert "The short version:" in r2.reply
    record = service.store.list_signal_records(cid)[-1]
    assert record.confusion and record.confusion_topic == "POL-CARE-01"


# -- records and the negative control ------------------------------------


def test_calm_progressing_conversation_fires_no_interventions(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "Why is my bill so high this month?")
    _say(service, cid, "My account is NB-1001 and the postcode is 02134")
    r3 = _say(service, cid, expected_code("NB-1001"))
    assert r3.status == "resolved"
    _say(service, cid, "Thanks, that's clear now.")
    records = service.store.list_signal_records(cid)
    assert len(records) == 4                     # one per turn, persisted
    assert [r.turn for r in records] == [1, 2, 3, 4]
    assert all(r.interventions == [] for r in records)
    assert all(r.frustration_rising is False for r in records)
    assert records[-1].effort_score >= records[0].effort_score


def test_signal_audit_events_carry_the_intervention_trail(service):
    cid, _ = _frustration_path(service)
    _say(service, cid, "Yes please.")
    events = [e for e in service.store.list_audit(cid) if e.kind == "signal"]
    assert [(e.detail["kind"], e.detail["outcome"]) for e in events] == [
        ("frustration_human_offer", "offered"),
        ("frustration_human_offer", "accepted")]


# -- the handoff packet's signal summary ---------------------------------


def test_packet_carries_the_signal_summary(service):
    cid, _r3, _r4 = _identity_path(service)
    _say(service, cid, "Yes please.")
    packet = service.store.list_escalations()[-1].packet
    sig = packet.signals
    assert sig["turns"] == 5
    # 5 turns + 2 identity attempts, no reroutes, no repeated asks.
    assert sig["effort_score"] == 7
    assert sig["effort"] == {"turns": 5, "identity_attempts": 2,
                             "reroutes": 0, "repeats": 0}
    assert sig["frustration_trend"] == "calm"
    assert sig["frustration_latest"] == 0
    assert sig["interventions"] == [
        {"kind": "identity_effort_human_offer", "outcome": "accepted"}]
    assert sig["second_forms_used"] == []
    assert any("Customer experience" in item
               for item in packet.open_items)


def test_packet_summary_rides_a_classic_escalation_too(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "Let me talk to a real person")
    r2 = _say(service, cid, "No, I said get me a human")
    assert r2.escalated
    assert r2.escalation_reason == "customer_requested_human"
    packet = service.store.list_escalations()[-1].packet
    assert packet.signals["turns"] == 2
    assert packet.signals["effort_score"] == 2
    assert packet.signals["interventions"] == []
    assert packet.signals["second_forms_used"] == []


def test_packet_summary_names_the_second_forms_used(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    _say(service, cid, "What time do your stores open on Saturday?")
    _say(service, cid, "I don't understand - what do you mean?")
    _say(service, cid, "Let me talk to a real person")
    r4 = _say(service, cid, "No, I said get me a human")
    assert r4.escalated
    packet = service.store.list_escalations()[-1].packet
    assert packet.signals["second_forms_used"] == ["POL-CARE-01"]
    assert packet.signals["interventions"] == [
        {"kind": "second_form:POL-CARE-01", "outcome": "applied"}]


# -- the signal report ----------------------------------------------------


def _plant_report_world(service):
    # Entered at billing: frustration rose, the offer was declined,
    # the conversation never escalated. Final effort 6.
    cid_a, _ = _frustration_path(service)
    _say(service, cid_a,
         "No thanks, I'll stay here. My account is NB-1001 and "
         "the postcode is 02134.")
    _say(service, cid_a,
         "111111 - and honestly I'm furious, this is unacceptable.")
    # Entered at account: the identity offer was accepted and the
    # conversation escalated, frustration calm throughout. Effort 7.
    cid_b, _r3, _r4 = _identity_path(service)
    _say(service, cid_b, "Yes please.")
    # Entered at care: two calm turns, resolved. Effort 2.
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "What time do your stores open on Saturday?")
    _say(service, conv.conversation_id, "Thanks, that's clear now.")


def test_signal_report_aggregates(service):
    _plant_report_world(service)
    report = service.signal_report()
    assert report["conversations"] == 3
    assert report["avg_effort"] == 5.0
    assert report["avg_effort_by_entry_desk"] == {
        "account": 7.0, "billing": 6.0, "care": 2.0}
    assert report["rising_frustration_conversations"] == 1
    assert report["rising_frustration_share"] == 0.333
    assert report["interventions"] == {
        "offered": 2, "accepted": 1, "declined": 1, "lapsed": 0,
        "second_forms": 0, "acceptance_rate": 0.5}
    assert report["frustration_escalation"] == {
        "rising_and_escalated": 0, "rising_not_escalated": 1,
        "not_rising_and_escalated": 1, "not_rising_not_escalated": 1}
    assert "not evidence" in report["note"]


def test_signal_report_on_an_empty_store(service):
    report = service.signal_report()
    assert report["conversations"] == 0
    assert report["avg_effort"] == 0.0
    assert report["interventions"]["acceptance_rate"] is None


def test_signals_api_series_report_and_404():
    from fastapi.testclient import TestClient

    from cx_agent.api import app

    client = TestClient(app)
    client.post("/seed")
    cid = client.post("/conversations").json()["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "What time do your stores open on Saturday?"})
    client.post(f"/conversations/{cid}/messages",
                json={"text": "I don't understand - what do you mean?"})
    body = client.get(f"/signals/{cid}").json()
    assert [r["turn"] for r in body["signals"]] == [1, 2]
    assert body["latest"]["confusion_topic"] == "POL-CARE-01"
    assert body["latest"]["interventions"] == [
        {"kind": "second_form:POL-CARE-01", "outcome": "applied"}]
    report = client.get("/signals/report").json()
    assert report["conversations"] >= 1
    assert "avg_effort_by_entry_desk" in report
    assert client.get("/signals/CONV-NOPE").status_code == 404
