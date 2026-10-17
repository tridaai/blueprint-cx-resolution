"""Round 8: customer care depth + unhandled-intent mining.

Care stops being the desk that answers in prose and becomes the
desk that records: a complaint is a stored record with a governed
category, the customer's words verbatim and a response deadline
the SLA watch reads; compliments and feedback are kept verbatim;
an accessibility need is written on the account and honoured at
the next contact. The vulnerability flag is set only from a
statement the customer makes plainly — bereavement, serious
illness, hardship — and never from a hint. And the conversations
that ended unhandled are mined into candidate intents a human
can design from.
"""

from __future__ import annotations

from datetime import date

import pytest

from cx_agent import care_book, mining, screening
from cx_agent.config import Policy
from cx_agent.identity import expected_code
from cx_agent.models import (
    AuditEvent,
    Conversation,
    Escalation,
    HandoffPacket,
    Message,
)

POLICY = Policy()


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-10-01")


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


# ---------------------------------------------------------------------------
# Complaints as records
# ---------------------------------------------------------------------------

COMPLAINT = ("I want to make a complaint — your engineer never showed "
             "up for Tuesday's appointment and nobody told me. "
             "My account is NB-1003.")


def test_complaint_categories_are_governed():
    assert care_book.complaint_category(COMPLAINT) == "appointment"
    assert care_book.complaint_category(
        "This is a complaint: the man on the phone was rude and hung "
        "up on me") == "staff_conduct"
    assert care_book.complaint_category(
        "I am complaining about being charged twice on my bill") \
        == "billing"
    assert care_book.complaint_category(
        "A complaint about coverage — there is no signal at my house") \
        == "coverage"
    assert care_book.complaint_category(
        "I want to complain about everything, generally") == "other"
    # No marker, no complaint — unhappiness alone is not a record.
    assert care_book.complaint_category(
        "Your service is terrible and slow") is None
    assert care_book.complaint_category("My bill is too high") is None


def test_complaint_becomes_a_record_with_its_deadline(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id, COMPLAINT)
    assert result.intent == "care"
    assert not result.escalated
    rows = service.store.list_complaints()
    assert len(rows) == 1
    record = rows[0]
    assert record.category == "appointment"
    assert record.text == COMPLAINT  # the customer's words, verbatim
    assert record.account_number == "NB-1003"
    assert record.status == "open"
    assert record.created_date == "2026-10-01"
    # The deadline is computed in code from policy: created + 5 days.
    assert record.response_due == "2026-10-06"
    assert POLICY.complaint_response_days == 5
    # The acknowledgment is in the reply: reference and deadline.
    assert record.complaint_id in result.reply
    assert "2026-10-06" in result.reply
    assert "appointment" in result.reply


def test_chasing_a_complaint_answers_from_its_record(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, COMPLAINT)
    result = _say(service, conv.conversation_id,
                  "Any news on my complaint?")
    rows = service.store.list_complaints()
    assert len(rows) == 1  # answered from the record, not re-recorded
    assert rows[0].complaint_id in result.reply
    assert "2026-10-06" in result.reply
    # A second, different complaint is a second record.
    _say(service, conv.conversation_id,
         "And another complaint — there is no signal at my house at all.")
    rows = service.store.list_complaints()
    assert len(rows) == 2
    assert rows[1].category == "coverage"


def test_complaint_without_an_account_is_still_recorded(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "I have a complaint — I waited in all day for an appointment "
         "that never happened.")
    rows = service.store.list_complaints()
    assert len(rows) == 1
    assert rows[0].account_number == ""
    assert rows[0].category == "appointment"


# ---------------------------------------------------------------------------
# The SLA watch
# ---------------------------------------------------------------------------


def _record(store, complaint_id, due: str, status="open",
            created="2026-09-20"):
    from cx_agent.models import ComplaintRecord
    rec = ComplaintRecord(
        complaint_id=complaint_id, conversation_id="CONV-X",
        account_number="NB-1001", category="billing", text="words",
        status=status, created_date=created, response_due=due)
    store.save_complaint(rec)
    return rec


def test_watch_reads_breach_and_near_breach_from_the_calendar(store):
    _record(store, "CMP-OLD", due="2026-09-30")       # past due, open
    _record(store, "CMP-NEAR", due="2026-10-02")      # due within a day
    _record(store, "CMP-FINE", due="2026-10-20")      # comfortably open
    _record(store, "CMP-DONE", due="2026-09-01", status="responded")
    watch = care_book.complaint_watch(store, POLICY, on=date(2026, 10, 1))
    assert [c.complaint_id for c in watch["breached"]] == ["CMP-OLD"]
    assert [c.complaint_id for c in watch["due_soon"]] == ["CMP-NEAR"]
    assert care_book.complaint_sla_state(
        store.get_complaint("CMP-DONE"), POLICY,
        on=date(2026, 10, 1)) == "settled"


def test_watch_raises_a_breach_to_the_queue_once(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, COMPLAINT)
    record = service.store.list_complaints()[0]
    # The world moves on: the deadline passes with no response.
    record.response_due = "2026-09-30"
    service.store.save_complaint(record)
    out = service.run_complaint_watch()
    assert out["raised"] and out["breached"] == [record.complaint_id]
    raised = [e for e in service.store.list_escalations()
              if e.reason == "complaint_sla_breach"]
    assert len(raised) == 1
    packet = raised[0].packet
    assert record.complaint_id in packet.summary
    assert "2026-09-30" in packet.summary  # the record rides along
    assert service.store.get_complaint(
        record.complaint_id).sla_escalation_id == raised[0].escalation_id
    # And the watch never raises the same breach twice.
    assert service.run_complaint_watch()["raised"] == []


# ---------------------------------------------------------------------------
# Feedback and compliments, verbatim
# ---------------------------------------------------------------------------


def test_feedback_kinds(service):
    assert care_book.feedback_kind(
        "Just want to give some feedback — the engineer was brilliant") \
        == "compliment"
    assert care_book.feedback_kind("Some feedback: the app is fine") \
        == "feedback"
    # Bare thanks is a pleasantry, not data.
    assert care_book.feedback_kind("thanks, great") is None
    assert care_book.feedback_kind("thank you") is None


def test_compliment_is_recorded_verbatim(service):
    text = ("Just want to give some feedback — the engineer who fixed "
            "my line was brilliant, so patient with me.")
    conv = service.start_conversation()
    _say(service, conv.conversation_id, text)
    rows = service.store.list_feedback()
    assert len(rows) == 1
    assert rows[0].kind == "compliment"
    assert rows[0].text == text


def test_plain_thanks_stays_conversation(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id, "thanks, great")
    assert "You're welcome" in result.reply
    assert service.store.list_feedback() == []


# ---------------------------------------------------------------------------
# Accessibility: recorded on the account, honoured later
# ---------------------------------------------------------------------------


def test_accessibility_need_is_recorded_on_the_account(service):
    conv = service.start_conversation()
    text = ("I can't read small print — please send my letters in "
            "large print instead. My account is NB-1011.")
    _say(service, conv.conversation_id, text)
    account = service.store.get_account("NB-1011")
    assert account.accessibility_note == text
    assert account.preferred_contact == "large print"


def test_accessibility_need_waits_for_the_account_number(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id,
                  "Please don't phone me — I struggle on the phone, "
                  "text me instead.")
    assert "account number" in result.reply
    assert service.store.get_account("NB-1011").preferred_contact == ""
    result = _say(service, conv.conversation_id, "It's NB-1011.")
    assert "recorded" in result.reply
    account = service.store.get_account("NB-1011")
    assert account.preferred_contact == "text message"
    assert "struggle on the phone" in account.accessibility_note


def test_recorded_preference_is_honoured_at_the_next_contact(service):
    first = service.start_conversation()
    _say(service, first.conversation_id,
         "I can't read small print — please send my letters in large "
         "print instead. My account is NB-1011.")
    later = service.start_conversation()
    result = _say(service, later.conversation_id,
                  "It's NB-1011 here — when you need to write to me "
                  "about something, how will you do it?")
    assert "large print" in result.reply


# ---------------------------------------------------------------------------
# Vulnerability discipline: the flag comes from a statement, not a hint
# ---------------------------------------------------------------------------


def test_explicit_statements_are_the_governed_classes():
    for text, klass in [
        ("My husband died last month.", "bereavement"),
        ("I lost my wife in the spring.", "bereavement"),
        ("I've been diagnosed with MS.", "serious_illness"),
        ("I am terminally ill.", "serious_illness"),
        ("I can't afford food this month.", "financial_hardship"),
        ("We're facing financial hardship.", "financial_hardship"),
    ]:
        assert screening.screen(text).vulnerable_statement == klass, text


def test_hints_set_nothing():
    for text in [
        "Sorry, I'm old and slow at typing, bear with me.",
        "Everything feels a bit much today, I'm a bit sad.",
        "My typing is slow, sorry, I'm not great with phones.",
        "I'm 84, so go gently with me.",
        "My neighbour's husband died, it's been hard on her.",
    ]:
        assert screening.screen(text).vulnerable_statement == "", text


def test_explicit_statement_sets_the_flag_and_routes(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id,
                  "My husband died two weeks ago and I'm struggling "
                  "to cope. My account is NB-1006 — please be patient "
                  "with me.")
    assert result.escalated
    assert result.escalation_reason == "vulnerable_customer"
    customer = service.store.get_customer_by_account("NB-1006")
    assert customer.vulnerable is True
    # The setting is audited, with its class and its source.
    events = [e for e in service.store.list_audit(conv.conversation_id)
              if e.kind == "vulnerability"]
    assert len(events) == 1
    assert events[0].detail["event"] == "flag_set"
    assert events[0].detail["statement_class"] == "bereavement"
    assert events[0].detail["source"] == "explicit_customer_statement"
    # And the handoff carries the flag where the human can see it.
    packet = service.store.list_escalations()[-1].packet
    assert packet.queue == "vulnerable_care"
    assert packet.vulnerable is True
    assert any("Vulnerability flag set" in item
               for item in packet.open_items)


def test_hinted_vulnerability_sets_no_flag(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id,
                  "Sorry, I'm old and slow at typing, bear with me — "
                  "everything feels a bit much today.")
    assert not result.escalated
    assert result.intent == "care"
    customer = service.store.get_customer_by_account("NB-1006")
    assert customer.vulnerable is False
    assert [e for e in service.store.list_audit(conv.conversation_id)
            if e.kind == "vulnerability"] == []


def test_the_flag_persists_for_the_next_contact(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "My husband died two weeks ago. My account is NB-1006.")
    # A later, ordinary conversation: once identity passes, the
    # flag on the record routes it to the care queue on its own.
    later = service.start_conversation()
    _say(service, later.conversation_id, "Can I see my bill?")
    _say(service, later.conversation_id,
         "Account NB-1006, postcode 30301.")
    result = _say(service, later.conversation_id,
                  expected_code("NB-1006"))
    assert result.escalated
    assert result.escalation_reason == "vulnerable_customer"


# ---------------------------------------------------------------------------
# Unhandled-intent mining
# ---------------------------------------------------------------------------

GIFT = [
    "Can I buy a gift card for my daughter?",
    "Do you sell gift cards? I want one for my son.",
    "I would like a gift card for my nephew please.",
    "Where can I get a Northbeam gift card?",
]


def _unhandled(store, conv_id, text):
    store.save_conversation(Conversation(
        conversation_id=conv_id, unhandled_intent=text[:120]))


def test_mining_finds_the_repeated_need_and_nothing_else(store):
    for i, text in enumerate(GIFT):
        _unhandled(store, f"CONV-G{i}", text)
    _unhandled(store, "CONV-L1", "Can I borrow money from you?")
    _unhandled(store, "CONV-I1", "Do you sell travel insurance?")
    report = mining.mine_unhandled_intents(store, POLICY)
    assert report["unhandled_utterances"] == 6
    assert len(report["candidates"]) == 1
    candidate = report["candidates"][0]
    assert candidate["count"] == 4
    assert "gift" in candidate["label"]
    assert candidate["samples"] == GIFT[:3]
    assert candidate["sources"] == {"unhandled_intent": 4}
    # Below the threshold, clusters are reported, not candidates.
    small = [c for c in report["clusters"] if not c["candidate"]]
    assert {c["count"] for c in small} == {1}


def test_mining_is_deterministic(store):
    for i, text in enumerate(GIFT):
        _unhandled(store, f"CONV-G{i}", text)
    first = mining.mine_unhandled_intents(store, POLICY)
    second = mining.mine_unhandled_intents(store, POLICY)
    assert first == second


def test_mining_reads_low_confidence_and_no_progress(store):
    conv = Conversation(conversation_id="CONV-LC", messages=[
        Message(role="customer", text="Do you do gift cards at all?",
                turn=1)])
    store.save_conversation(conv)
    store.save_audit(AuditEvent(
        event_id="EVT-LC", conversation_id="CONV-LC", kind="turn",
        turn=1, detail={"routing": {"decision": "low_confidence_care"}}))
    stuck = Conversation(conversation_id="CONV-NP", messages=[
        Message(role="customer", text="Something odd happened here",
                turn=1),
        Message(role="customer", text="It just keeps happening",
                turn=3)])
    store.save_conversation(stuck)
    store.save_escalation(Escalation(
        escalation_id="ESC-NP", conversation_id="CONV-NP",
        queue="general", reason="no_progress", turn=3,
        packet=HandoffPacket(conversation_id="CONV-NP", queue="general",
                             reason="no_progress",
                             identity_state="unverified",
                             summary="three turns, no progress")))
    report = mining.mine_unhandled_intents(store, POLICY)
    sources = [c["sources"] for c in report["clusters"]]
    assert {"low_confidence": 1} in sources
    assert {"no_progress": 1} in sources
    np_cluster = next(c for c in report["clusters"]
                      if c["sources"].get("no_progress"))
    assert np_cluster["samples"] == ["It just keeps happening"]
