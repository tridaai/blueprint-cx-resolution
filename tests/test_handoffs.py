"""Handoff quality (round 17): the packet graded in code
against the elements its class requires, acceptance as a
recorded act, unaccepted handoffs surfaced by age, and the
transfer line that claims only what the packet holds."""

import os

os.environ.setdefault("CX_TODAY", "2026-10-01")

from datetime import date  # noqa: E402

import pytest  # noqa: E402

from cx_agent import handoffs  # noqa: E402
from cx_agent.config import Policy  # noqa: E402
from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.models import HandoffPacket  # noqa: E402
from cx_agent.quality import render_digest_markdown  # noqa: E402

POLICY = Policy()


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _escalate_human_demander(service):
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "Let me talk to a real person")
    result = _say(service, conv.conversation_id, "No, I said get me a human")
    assert result.escalated
    return conv, result


def _packet(service, conv_id):
    esc = [e for e in service.list_escalations()
           if e.conversation_id == conv_id][-1]
    return esc, esc.packet


# ---------------------------------------------------------------------------
# Completeness, graded in code
# ---------------------------------------------------------------------------

def test_gap_is_named_when_nothing_was_tried(service):
    """The human-demander leaves before any work exists: the
    packet transfers anyway, graded incomplete with the one
    missing element named — on the packet and in the audit."""
    conv, result = _escalate_human_demander(service)
    esc, packet = _packet(service, conv.conversation_id)
    assert packet.handoff_class == "standard"
    assert packet.complete is False
    assert packet.missing == ["tried"]
    assert packet.issue == "Let me talk to a real person"
    audit = [e for e in service.store.list_audit(conv.conversation_id)
             if e.kind == "escalation"]
    assert audit[-1].detail["packet_complete"] is False
    assert audit[-1].detail["packet_missing"] == ["tried"]
    assert esc.created_date == "2026-10-01"


def test_dispute_packet_is_complete_with_figures_from_the_record(service):
    """The dispute that outgrew the agent's cap hands over its
    figures and its decided lines — nothing the human needs is
    left to re-derive."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "why is my bill so high")
    _say(service, conv.conversation_id, "NB-1012 postcode 80202")
    _say(service, conv.conversation_id, expected_code("NB-1012"))
    result = _say(
        service, conv.conversation_id,
        "I dispute the roaming charge on my bill — this is the "
        "wrong charge, I was quoted $20 for Greece, not $78")
    assert result.escalated
    _, packet = _packet(service, conv.conversation_id)
    assert packet.complete is True and packet.missing == []
    assert packet.identity_state == "verified"
    assert any("$58.00" in f for f in packet.figures)
    assert any("Dispute" in t and "escalated" in t for t in packet.tried)
    assert packet.open_state_known is True


def test_technical_packet_carries_the_steps_that_were_tried(service):
    """A line test that finds a fault escalates to a field
    visit: the packet's tried list is the run's own history."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, "my broadband keeps dropping")
    _say(service, conv.conversation_id, "NB-1005 postcode 94107")
    result = _say(service, conv.conversation_id, expected_code("NB-1005"))
    assert result.escalation_reason == "field_visit_required"
    _, packet = _packet(service, conv.conversation_id)
    assert packet.complete is True
    assert any("BB-1" in t and "fault_found" in t for t in packet.tried)
    assert packet.identity_attempts >= 0


def test_immediate_packet_needs_no_history(service):
    """A fraud report on turn one: graded as its own class,
    complete with nothing behind it — leaving at once is the
    design, not a gap."""
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id,
                  "someone has used my card, this is fraud")
    assert result.escalated
    _, packet = _packet(service, conv.conversation_id)
    assert packet.handoff_class == "immediate"
    assert packet.complete is True and packet.missing == []


def test_watch_packet_is_graded_as_its_own_class(service):
    """The complaint watch raises its packet without a live
    conversation: the customer's words are the complaint, the
    open promises ride it, and no signals element is required."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id,
         "I want to make a complaint — your engineer never showed "
         "up for Tuesday's appointment and nobody told me. "
         "My account is NB-1003.")
    complaints = service.store.list_complaints(
        conversation_id=conv.conversation_id)
    assert complaints
    record = complaints[-1]
    record.response_due = "2026-09-28"
    service.store.save_complaint(record)
    raised = service.run_complaint_watch()["raised"]
    assert raised
    esc = [e for e in service.list_escalations()
           if e.escalation_id == raised[-1]][-1]
    packet = esc.packet
    assert packet.handoff_class == "watch"
    assert packet.issue == record.text[:200]
    assert packet.open_state_known is True
    assert any(record.complaint_id in p or "complaint" in p.lower()
               for p in packet.open_promises)
    assert packet.complete is True
    assert esc.created_date == "2026-10-01"


def test_grade_rules_per_class():
    """The grading table itself, at unit level: which elements
    each class is held to, and which it is not."""
    base = dict(conversation_id="C", queue="general", reason="x",
                identity_state="verified", summary="s",
                issue="the customer's own words",
                signals={"effort_score": 1})

    def graded(**kw):
        intent = kw.pop("_intent", "")
        fields = dict(base)
        fields.update(kw)
        return handoffs.grade_packet(HandoffPacket(**fields), intent)

    # Standard, nothing tried: the gap is named.
    p = graded(reason="no_progress")
    assert p.missing == ["tried"] and not p.complete
    # Billing by queue: figures are owed whichever door it
    # leaves by, even an immediate one.
    p = graded(reason="fraud_report", queue="billing")
    assert "figures" in p.missing
    # Care: tried is not required — the transfer is the design.
    p = graded(reason="vulnerable_customer")
    assert p.complete
    # Identity lockout on a billing case: figures owed, and the
    # account's open state once the account is known.
    p = graded(reason="identity_verification_failed",
               account_number="NB-1001", _intent="billing")
    assert p.missing == ["figures", "open_promises"]


# ---------------------------------------------------------------------------
# The line to the customer
# ---------------------------------------------------------------------------

def test_line_claims_only_what_the_packet_holds():
    """Zero-tolerance in spirit: a fragment appears in the
    transfer line only when the packet holds its element."""
    full = HandoffPacket(
        conversation_id="C", queue="general", reason="no_progress",
        identity_state="verified", summary="s", issue="my bill is wrong",
        tried=["step BB-1: clear"], figures=["apply_credit $5.00 (executed)"],
        open_items=["The customer's issue is not resolved."],
        known_incidents=[{"incident_id": "INC-1"}])
    line = handoffs.continuity_line(full)
    for fragment in ("identity and verification status", "in your own words",
                     "steps we tried", "amounts involved",
                     "known fault we discussed", "what is still open"):
        assert fragment in line

    bare = HandoffPacket(
        conversation_id="C", queue="general", reason="fraud_report",
        identity_state="unverified", summary="s")
    line = handoffs.continuity_line(bare)
    assert line == ("I've passed on your identity and verification "
                    "status — so you won't need to repeat yourself.")


def test_escalation_reply_uses_the_generated_line(service):
    """End to end: the demander who left before anything was
    tried is not told that steps were passed on."""
    _, result = _escalate_human_demander(service)
    assert "in your own words" in result.reply
    assert "what is still open" in result.reply
    assert "steps we tried" not in result.reply
    assert "everything we've covered" not in result.reply


# ---------------------------------------------------------------------------
# Acceptance — a recorded act
# ---------------------------------------------------------------------------

def test_acceptance_stamps_receiver_day_and_audit(service):
    conv, _ = _escalate_human_demander(service)
    esc, _ = _packet(service, conv.conversation_id)
    accepted = service.accept_handoff(esc.escalation_id, "j.riera")
    assert accepted.accepted_by == "j.riera"
    assert accepted.accepted_date == "2026-10-01"
    events = [e for e in service.store.list_audit(conv.conversation_id)
              if e.kind == "handoff"]
    assert events and events[-1].detail["event"] == "handoff_accepted"
    assert events[-1].detail["receiver"] == "j.riera"
    # Singular: the first receiver stands.
    with pytest.raises(ValueError):
        service.accept_handoff(esc.escalation_id, "someone.else")
    with pytest.raises(KeyError):
        service.accept_handoff("ESC-nope", "j.riera")
    with pytest.raises(ValueError):
        service.accept_handoff(esc.escalation_id, "  ")


def test_unaccepted_handoffs_age_by_the_shared_clock(service, monkeypatch):
    """Raised today: not overdue. Past the governed threshold
    (1 day) with no pickup: named, with its age. Accepted: gone
    from the list whatever its age."""
    conv, _ = _escalate_human_demander(service)
    esc, _ = _packet(service, conv.conversation_id)
    assert service.unaccepted_handoffs() == []
    monkeypatch.setenv("CX_TODAY", "2026-10-02")
    assert service.unaccepted_handoffs() == []  # age 1, not past 1
    monkeypatch.setenv("CX_TODAY", "2026-10-03")
    overdue = service.unaccepted_handoffs()
    assert len(overdue) == 1
    assert overdue[0]["escalation_id"] == esc.escalation_id
    assert overdue[0]["age_days"] == 2
    assert overdue[0]["complete"] is False
    assert overdue[0]["missing"] == ["tried"]
    service.accept_handoff(esc.escalation_id, "j.riera")
    assert service.unaccepted_handoffs() == []


def test_acceptance_threshold_is_governed(service, monkeypatch):
    conv, _ = _escalate_human_demander(service)
    _packet(service, conv.conversation_id)
    monkeypatch.setenv("CX_TODAY", "2026-10-03")
    lenient = Policy(handoff_acceptance_days=7)
    assert handoffs.unaccepted_overdue(service.store, lenient) == []
    assert len(handoffs.unaccepted_overdue(
        service.store, POLICY, on=date(2026, 10, 3))) == 1


# ---------------------------------------------------------------------------
# The digest surfaces both
# ---------------------------------------------------------------------------

def test_digest_names_incomplete_packets_and_overdue_handoffs(
        service, monkeypatch):
    conv, _ = _escalate_human_demander(service)
    esc, _ = _packet(service, conv.conversation_id)
    digest = service.run_daily_quality(with_spot_check=False)
    assert digest.handoffs["incomplete_count"] == 1
    row = digest.handoffs["incomplete"][0]
    assert row["escalation_id"] == esc.escalation_id
    assert row["missing"] == ["tried"]
    assert digest.handoffs["unaccepted_overdue"] == []
    markdown = render_digest_markdown(digest)
    assert "INCOMPLETE" in markdown and "missing tried" in markdown

    # Two days on, nobody picked it up: a fresh digest for the
    # day names it overdue (whole-book state), while the
    # incomplete list stays scoped to that day's conversations.
    monkeypatch.setenv("CX_TODAY", "2026-10-03")
    digest = service.run_daily_quality(on=date(2026, 10, 3),
                                       with_spot_check=False)
    assert digest.handoffs["incomplete"] == []
    overdue = digest.handoffs["unaccepted_overdue"]
    assert len(overdue) == 1 and overdue[0]["age_days"] == 2
    markdown = render_digest_markdown(digest)
    assert "UNACCEPTED" in markdown and esc.escalation_id in markdown
