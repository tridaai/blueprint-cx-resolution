"""Round 15: known-issue awareness.

The incident feed is a governed record set, matching is
deterministic code (a live incident, its declared signals in
the customer's words, the service agreeing, the postcode scope
covering the account), and a match changes the conversation in
exactly one way: the agent acknowledges the known issue from
the record itself — title, status, the operator's summary
verbatim — starts no procedure, proposes nothing unrelated,
and offers the book's callback follow-up instead. Everything
else follows the old paths, and a resolved incident matches
nothing, even mid-conversation.
"""

from __future__ import annotations

import pytest

from cx_agent import incidents as inc
from cx_agent import procedures
from cx_agent.identity import expected_code
from cx_agent.models import Conversation, Incident, ProcedureRun
from cx_agent.specialists import Ctx
from cx_agent.config import Policy


@pytest.fixture(autouse=True)
def _pin_today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-10-01")


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _drive_to_code(service, account_number, postcode, opener):
    """Report a fault, answer the identity flow, and let the
    code turn resume the request at the technical desk.
    Returns (conversation, result of the code turn)."""
    conv = service.start_conversation()
    _say(service, conv.conversation_id, opener)
    _say(service, conv.conversation_id,
         f"{account_number} postcode {postcode}")
    result = _say(service, conv.conversation_id,
                  expected_code(account_number))
    return conv, result


def _matches(service, conv):
    return service.conversation_incidents(conv.conversation_id)


SUMMARY_7001 = ("Some broadband lines served by the Northgate exchange "
                "are dropping repeatedly. Engineers are working at the "
                "exchange, and the fault is on our side, not your "
                "equipment.")


# ---------------------------------------------------------------------------
# Matching — the rules, read directly
# ---------------------------------------------------------------------------


def _conv(store, account_number=None):
    conv = Conversation(conversation_id="CONV-T")
    if account_number:
        conv.account_number = account_number
        customer = store.get_customer_by_account(account_number)
        conv.customer_id = customer.customer_id
    return conv


def test_seeded_feed_shape(seeded_store):
    feed = seeded_store.list_incidents()
    assert [i.incident_id for i in feed] == ["INC-7001", "INC-7002",
                                             "INC-7003"]
    assert feed[0].status == "active"
    assert feed[0].service == "broadband_drops"
    assert feed[0].affected_postcodes == ["02139"]
    assert feed[1].status == "monitoring"
    assert feed[2].status == "resolved"
    assert feed[2].affected_postcodes == []  # unscoped: every area


def test_match_needs_signals_service_and_area(seeded_store):
    elena = _conv(seeded_store, "NB-1007")  # postcode 02139
    matches = inc.match_incidents(
        seeded_store, elena, "my broadband keeps dropping")
    assert [i.incident_id for i, _s in matches] == ["INC-7001"]
    assert "broadband" in matches[0][1]
    # Area gate: Marcus (30301) reports the identical words.
    marcus = _conv(seeded_store, "NB-1006")
    assert inc.match_incidents(
        seeded_store, marcus, "my broadband keeps dropping") == []
    # Service gate: Elena reports a different fault entirely.
    assert inc.match_incidents(
        seeded_store, elena, "I have no signal at all") == []
    # Signal gate: her area and service, but not her words.
    assert inc.match_incidents(
        seeded_store, elena, "my phone battery drains fast") == []


def test_scoped_incident_waits_for_the_account(seeded_store):
    anon = _conv(seeded_store)
    assert inc.match_incidents(
        seeded_store, anon, "my broadband keeps dropping") == []


def test_resolved_never_matches_and_monitoring_does(seeded_store):
    anon = _conv(seeded_store)
    # INC-7003 is unscoped and its signals are exactly these
    # words — but it is resolved, so it explains nothing.
    assert inc.match_incidents(
        seeded_store, anon, "I have no signal at all") == []
    tom = _conv(seeded_store, "NB-1004")  # postcode 60614
    matches = inc.match_incidents(
        seeded_store, tom, "my internet is slow today")
    assert [i.incident_id for i, _s in matches] == ["INC-7002"]
    assert matches[0][0].status == "monitoring"


# ---------------------------------------------------------------------------
# Operator acts — recorded, validated, and effective
# ---------------------------------------------------------------------------


def test_acts_are_recorded_in_order(seeded_store):
    added = inc.add_incident(
        seeded_store, title="Signal down on the north side",
        service="no_signal", summary="A mast fault is taking "
        "signal down on the north side.",
        symptom_signals=["no signal"], actor="ops")
    assert added.incident_id.startswith("INC-")
    inc.update_incident(seeded_store, added.incident_id,
                        actor="ops", summary="A mast fault is "
                        "taking signal down across the north side.")
    resolved = inc.resolve_incident(seeded_store, added.incident_id,
                                    actor="ops")
    assert resolved.status == "resolved"
    acts = seeded_store.list_incident_acts(added.incident_id)
    assert [a.kind for a in acts] == ["add", "update", "resolve"]
    assert [a.seq for a in acts] == sorted(a.seq for a in acts)
    assert all(a.actor == "ops" for a in acts)


def test_acts_refuse_what_the_feed_cannot_carry(seeded_store):
    with pytest.raises(inc.IncidentError):
        inc.add_incident(seeded_store, incident_id="INC-7001",
                         title="duplicate", service="no_signal",
                         summary="x", symptom_signals=["no signal"])
    with pytest.raises(inc.IncidentError):
        inc.add_incident(seeded_store, title="bad service",
                         service="billing", summary="x",
                         symptom_signals=["bill"])
    with pytest.raises(inc.IncidentError):
        inc.add_incident(seeded_store, title="no signals",
                         service="no_signal", summary="x",
                         symptom_signals=[])
    with pytest.raises(inc.IncidentError):
        inc.add_incident(seeded_store, title="no summary",
                         service="no_signal", summary=" ",
                         symptom_signals=["no signal"])
    with pytest.raises(inc.IncidentError):
        inc.update_incident(seeded_store, "INC-7001",
                            started_by="nobody")
    with pytest.raises(inc.IncidentError):
        inc.resolve_incident(seeded_store, "INC-7003")  # already resolved
    with pytest.raises(inc.IncidentError):
        inc.resolve_incident(seeded_store, "INC-NOPE")
    # Nothing was written by the refused acts.
    assert seeded_store.list_incident_acts() == []
    assert seeded_store.get_incident("INC-7001").status == "active"


def test_resolve_stops_matching_at_once(seeded_store):
    elena = _conv(seeded_store, "NB-1007")
    assert inc.match_incidents(
        seeded_store, elena, "my broadband keeps dropping")
    inc.resolve_incident(seeded_store, "INC-7001", actor="ops")
    assert inc.match_incidents(
        seeded_store, elena, "my broadband keeps dropping") == []


def test_reseed_keeps_an_operators_resolution(seeded_store):
    from cx_agent.dataset import seed_store

    inc.resolve_incident(seeded_store, "INC-7001", actor="ops")
    seed_store(seeded_store)
    assert seeded_store.get_incident("INC-7001").status == "resolved"


# ---------------------------------------------------------------------------
# The acknowledgement in conversation
# ---------------------------------------------------------------------------


def test_match_acknowledges_and_books_the_followup(service):
    conv, result = _drive_to_code(
        service, "NB-1007", "02139", "my broadband keeps dropping")
    # The record's own words, up front — and no procedure ran.
    assert "INC-7001" in result.reply
    assert "Broadband drops on the Northgate exchange" in result.reply
    assert "status: active" in result.reply
    assert SUMMARY_7001 in result.reply
    assert "unplug" not in result.reply
    assert service.store.list_procedure_runs(conv.conversation_id) == []
    # The follow-up is the book's callback, as a record.
    assert result.action is not None
    assert result.action.kind.value == "book_callback"
    assert result.action.status == "proposed"
    assert result.action.payload["incident_id"] == "INC-7001"
    callbacks = service.store.list_callbacks(conv.conversation_id)
    assert len(callbacks) == 1
    assert "INC-7001" in callbacks[0].reason
    # The match is audited with its signals, and listed per conversation.
    audit = [e for e in service.store.list_audit(conv.conversation_id)
             if e.kind == "incident"]
    assert len(audit) == 1
    assert audit[0].detail["incident_id"] == "INC-7001"
    assert "broadband" in audit[0].detail["matched_signals"]
    matches = _matches(service, conv)
    assert [m["incident_id"] for m in matches] == ["INC-7001"]
    assert matches[0]["current_status"] == "active"

    done = _say(service, conv.conversation_id, "yes")
    assert done.action.status == "executed"
    assert service.store.list_callbacks(
        conv.conversation_id)[0].status == "booked"


def test_area_mismatch_troubleshoots_as_before(service):
    conv, result = _drive_to_code(
        service, "NB-1006", "30301", "my broadband keeps dropping")
    # Marcus is outside INC-7001's postcode scope: the normal
    # walk starts — line test, then the safety step.
    assert "INC-7001" not in result.reply
    assert "unplug" in result.reply
    assert _matches(service, conv) == []
    runs = service.store.list_procedure_runs(conv.conversation_id)
    assert len(runs) == 1 and runs[0].state == "active"


def test_service_mismatch_troubleshoots_as_before(service):
    conv = service.start_conversation()
    result = _say(service, conv.conversation_id,
                  "I have no signal at all")
    # Elena's area has a broadband incident; her report is a
    # signal fault, and the seeded signal incident is resolved —
    # the ordinary No signal walk starts.
    assert "Restart your phone" in result.reply
    assert "Northgate" not in result.reply
    assert _matches(service, conv) == []
    runs = service.store.list_procedure_runs(conv.conversation_id)
    assert len(runs) == 1 and runs[0].state == "active"


def test_unscoped_incident_acknowledges_before_identity(service):
    added = service.add_incident(
        title="Signal down across the north side", service="no_signal",
        summary="A mast fault is taking signal down across the "
                "north side. Engineers are on site.",
        symptom_signals=["no signal", "no service"], actor="ops")
    conv = service.start_conversation()
    first = _say(service, conv.conversation_id, "I have no signal at all")
    assert added.incident_id in first.reply
    assert "Signal down across the north side" in first.reply
    assert "account number" in first.reply  # the callback needs it
    assert service.store.list_procedure_runs(conv.conversation_id) == []
    _say(service, conv.conversation_id, "NB-1004 postcode 60614")
    resumed = _say(service, conv.conversation_id,
                   expected_code("NB-1004"))
    assert resumed.action is not None
    assert resumed.action.kind.value == "book_callback"
    assert resumed.action.payload["incident_id"] == added.incident_id


def test_resolution_mid_conversation_stops_the_acknowledgement(service):
    conv, result = _drive_to_code(
        service, "NB-1007", "02139", "my broadband keeps dropping")
    assert "INC-7001" in result.reply
    _say(service, conv.conversation_id, "no")  # decline the callback
    service.resolve_incident("INC-7001", actor="ops")
    later = _say(service, conv.conversation_id,
                 "my broadband keeps dropping again")
    # History stands — the match stays on the record — but the
    # acknowledgement stops, and troubleshooting takes over.
    assert SUMMARY_7001 not in later.reply
    assert "unplug" in later.reply
    matches = _matches(service, conv)
    assert [m["incident_id"] for m in matches] == ["INC-7001"]
    assert matches[0]["current_status"] == "resolved"
    runs = service.store.list_procedure_runs(conv.conversation_id)
    assert len(runs) == 1 and runs[0].state == "active"


def test_incident_arriving_mid_walk_stops_the_walk(service):
    conv, result = _drive_to_code(
        service, "NB-1006", "30301", "my broadband keeps dropping")
    assert "unplug" in result.reply  # the walk is live
    added = service.add_incident(
        title="Broadband faults across the Southgate exchange",
        service="broadband_drops", affected_postcodes=["30301"],
        summary="Broadband lines on the Southgate exchange are "
                "dropping. The fault is at the exchange.",
        symptom_signals=["broadband", "keeps dropping"], actor="ops")
    nxt = _say(service, conv.conversation_id, "no, still the same")
    assert added.incident_id in nxt.reply
    assert "Southgate" in nxt.reply
    runs = service.store.list_procedure_runs(conv.conversation_id)
    assert runs[0].state == "abandoned"
    assert runs[0].outcome == "matched_known_incident"
    assert nxt.action is not None
    assert nxt.action.kind.value == "book_callback"


def test_escalation_packet_carries_the_known_issue(service):
    conv, _ = _drive_to_code(
        service, "NB-1007", "02139", "my broadband keeps dropping")
    _say(service, conv.conversation_id, "no")
    _say(service, conv.conversation_id, "let me talk to a real person")
    done = _say(service, conv.conversation_id, "I said get me a human")
    assert done.escalated
    packet = service.list_escalations()[-1].packet
    assert [k["incident_id"] for k in packet.known_incidents] == ["INC-7001"]
    assert packet.known_incidents[0]["summary"] == SUMMARY_7001
    assert any("INC-7001" in item for item in packet.open_items)


def test_resolved_incident_does_not_ride_the_packet(service):
    conv, _ = _drive_to_code(
        service, "NB-1007", "02139", "my broadband keeps dropping")
    _say(service, conv.conversation_id, "no")
    service.resolve_incident("INC-7001", actor="ops")
    _say(service, conv.conversation_id, "let me talk to a real person")
    _say(service, conv.conversation_id, "I said get me a human")
    packet = service.list_escalations()[-1].packet
    assert packet.known_incidents == []


# ---------------------------------------------------------------------------
# The coverage tool reads the feed
# ---------------------------------------------------------------------------


def test_coverage_tool_reads_the_feed(seeded_store):
    conv = _conv(seeded_store, "NB-1004")
    run = ProcedureRun(run_id="RUN-T", conversation_id="CONV-T",
                       account_number="NB-1004",
                       procedure_id="PROC-NOSIGNAL")
    ctx = Ctx(store=seeded_store, policy=Policy(), retriever=None)
    result, _ = procedures.TOOLS["coverage_ok"](ctx, conv, run)
    assert result == "clear"  # nothing live covers 60614 yet
    inc.add_incident(seeded_store, title="Signal down — West Loop",
                     service="no_signal", affected_postcodes=["60614"],
                     summary="Signal is down around the West Loop.",
                     symptom_signals=["no signal"], actor="ops")
    result, finding = procedures.TOOLS["coverage_ok"](ctx, conv, run)
    assert result == "outage"
    assert "reported outage" in finding.text


def test_incident_record_round_trips_verbatim(seeded_store):
    record = Incident(
        incident_id="INC-T1", title="T", service="slow_data",
        affected_postcodes=["60614"], symptom_signals=["slow data"],
        status="monitoring", started_at="2026-09-30",
        summary="Speeds are lower than usual.", updated_date="2026-09-30")
    seeded_store.save_incident(record)
    assert seeded_store.get_incident("INC-T1") == record
    assert seeded_store.list_incidents()[-1] == record
