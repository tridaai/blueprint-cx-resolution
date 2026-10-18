"""Round 16 — continuity.

The briefing is computed in code from the account's records at
the moment identity verifies, and every element of it must trace
to a record: the traceability test walks the briefing and
resolves each element's id through the store. The promises
ledger is a derivation over the commitment records — complaint
deadlines, booked callbacks, arrangement instalments — with due
states from the shared clock; broken promises surface in the
briefing and in the daily digest, and nothing acts on them. A
procedure run left active by one conversation is adopted by the
next and continued, not restarted. A customer with no history
gets an empty briefing and an ordinary opening.

Dates pin to 2026-10-01, the dataset's present day.
"""

import os
import tempfile
from datetime import date
from decimal import Decimal as D

os.environ.setdefault("CX_TODAY", "2026-10-01")
os.environ["SQLITE_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="cx-continuity-api-"), "cx.db")
os.environ.pop("DATABASE_URL", None)

import pytest  # noqa: E402

from cx_agent import continuity, quality  # noqa: E402
from cx_agent.config import Policy  # noqa: E402
from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.models import (  # noqa: E402
    ActionKind, ActionRecord, ArrangementInstallment, AttributionRecord,
    CallbackRecord, ComplaintRecord, Conversation, DisputeCase,
    PaymentArrangement, ProcedureRun, ProposedAction)
from cx_agent.routing import Intent  # noqa: E402

DAY = date(2026, 10, 1)


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-10-01")


def _say(service, conv_id, text):
    return service.post_message(conv_id, text)


def _identify(service, conv_id, account, postcode):
    _say(service, conv_id, f"{account} postcode {postcode}")
    return _say(service, conv_id, expected_code(account))


def _complaint(store, cid="CMP-T1", account="NB-1002", conv="CONV-OLD",
               created="2026-09-28", due="2026-10-03", status="open"):
    record = ComplaintRecord(
        complaint_id=cid, conversation_id=conv, account_number=account,
        category="missed_appointment", text="The engineer never showed.",
        status=status, created_date=created, response_due=due)
    store.save_complaint(record)
    return record


def _callback(store, cid="CBK-T1", account="NB-1002", conv="CONV-OLD",
              status="booked", created="2026-09-29"):
    record = CallbackRecord(
        callback_id=cid, conversation_id=conv, account_number=account,
        procedure_id="PROC-NOSIGNAL", status=status,
        created_date=created)
    store.save_callback(record)
    return record


def _arrangement(store, aid="ARR-T1", account="NB-1002", conv="CONV-OLD",
                 installments=None, created="2026-09-05"):
    arr = PaymentArrangement(
        arrangement_id=aid, conversation_id=conv, account_number=account,
        total=D("187.50"),
        installments=installments if installments is not None else [
            ArrangementInstallment(number=1, amount=D("62.50"),
                                   due_date="2026-09-19",
                                   paid_amount=D("62.50")),
            ArrangementInstallment(number=2, amount=D("62.50"),
                                   due_date="2026-09-25"),
            ArrangementInstallment(number=3, amount=D("62.50"),
                                   due_date="2026-11-18")],
        state="active", created_date=created)
    store.save_payment_arrangement(arr)
    return arr


# ---------------------------------------------------------------------------
# The promises ledger — due states derived from the records
# ---------------------------------------------------------------------------

def test_ledger_complaint_due_states(seeded_store):
    _complaint(seeded_store, "CMP-PEND", due="2026-10-03")
    _complaint(seeded_store, "CMP-BROKE", due="2026-09-30")
    _complaint(seeded_store, "CMP-KEPT", due="2026-09-30",
               status="responded")
    entries = {e.source_id: e for e in continuity.promise_ledger(
        seeded_store, "NB-1002", on=DAY)}
    assert entries["CMP-PEND"].state == "pending"
    assert entries["CMP-BROKE"].state == "broken"
    assert entries["CMP-KEPT"].state == "kept"
    assert entries["CMP-PEND"].kind == "complaint_response"
    assert entries["CMP-PEND"].promise_id == "PRM-CMP-PEND"
    assert entries["CMP-PEND"].due_date == "2026-10-03"


def test_ledger_reads_as_of_a_date(seeded_store):
    _complaint(seeded_store, "CMP-LATE", created="2026-09-28",
               due="2026-10-03")
    earlier = continuity.promise_ledger(seeded_store, "NB-1002",
                                        on=date(2026, 9, 27))
    assert all(e.source_id != "CMP-LATE" for e in earlier)  # not yet made
    on_due = continuity.promise_ledger(seeded_store, "NB-1002",
                                       on=date(2026, 10, 3))
    entry = next(e for e in on_due if e.source_id == "CMP-LATE")
    assert entry.state == "pending"  # due that day is not yet broken


def test_ledger_callback_due_states(seeded_store):
    _callback(seeded_store, "CBK-MADE", status="booked")
    _callback(seeded_store, "CBK-STOPPED", status="blocked")
    _callback(seeded_store, "CBK-ASKED", status="proposed")
    _callback(seeded_store, "CBK-NO", status="declined")
    entries = {e.source_id: e for e in continuity.promise_ledger(
        seeded_store, "NB-1002", on=DAY)}
    assert entries["CBK-MADE"].state == "kept"  # a callback made is kept
    assert entries["CBK-MADE"].due_date == "2026-09-30"  # created + 1 day
    assert entries["CBK-STOPPED"].state == "broken"
    assert "CBK-ASKED" not in entries  # unanswered: no promise existed
    assert "CBK-NO" not in entries  # declined: released by the customer


def test_ledger_arrangement_instalments(seeded_store):
    _arrangement(seeded_store)
    entries = {e.promise_id: e for e in continuity.promise_ledger(
        seeded_store, "NB-1002", on=DAY)}
    assert entries["PRM-ARR-T1-1"].state == "kept"  # covered
    assert entries["PRM-ARR-T1-2"].state == "broken"  # due past, short
    assert entries["PRM-ARR-T1-3"].state == "pending"  # not yet due
    assert entries["PRM-ARR-T1-2"].due_date == "2026-09-25"


def test_ledger_sorts_broken_first(seeded_store):
    _complaint(seeded_store, "CMP-KEPT", due="2026-09-30",
               status="closed")
    _complaint(seeded_store, "CMP-PEND", due="2026-10-03")
    _complaint(seeded_store, "CMP-BROKE", due="2026-09-30")
    states = [e.state for e in continuity.promise_ledger(
        seeded_store, "NB-1002", on=DAY)]
    assert states == ["broken", "pending", "kept"]


def test_seeded_books_carry_martas_broken_instalment(seeded_store):
    entries = continuity.promise_ledger(seeded_store, "NB-1015", on=DAY)
    by_id = {e.promise_id: e for e in entries}
    assert by_id["PRM-ARR-HIST-1015-1"].state == "broken"
    assert by_id["PRM-ARR-HIST-1015-2"].state == "pending"
    assert by_id["PRM-ARR-HIST-1015-2"].due_date == "2026-10-19"


# ---------------------------------------------------------------------------
# The briefing — every element traces to a record
# ---------------------------------------------------------------------------

def _resolve_item(store, item):
    """The backing record for a briefing element, or None."""
    kind, ref = item["kind"], item["ref_id"]
    if kind == "complaint":
        return store.get_complaint(ref)
    if kind == "dispute":
        return store.get_dispute_case(ref)
    if kind == "callback":
        return store.get_callback(ref)
    if kind == "arrangement":
        return next((a for a in store.list_payment_arrangements()
                     if a.arrangement_id == ref), None)
    return None


def test_briefing_elements_all_trace_to_records(seeded_store):
    _complaint(seeded_store)
    _callback(seeded_store)
    _arrangement(seeded_store)
    seeded_store.save_dispute_case(DisputeCase(
        case_id="DSP-T1", conversation_id="CONV-OLD",
        account_number="NB-1002", bill_id="BILL-1", state="opened",
        created_date="2026-09-27"))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert {i["kind"] for i in briefing.open_items} == {
        "complaint", "callback", "arrangement", "dispute"}
    for item in briefing.open_items:
        assert _resolve_item(seeded_store, item) is not None, item
    for entry in briefing.broken_promises:
        assert seeded_store.get_payment_arrangement(entry["source_id"]) \
            is not None or seeded_store.get_complaint(
                entry["source_id"]) is not None
    assert briefing.lines  # an opening was composed from the elements


def test_briefing_broken_complaint_is_surfaced_not_softened(seeded_store):
    _complaint(seeded_store, due="2026-09-25")
    briefing = continuity.build_briefing(seeded_store, "NB-1002", on=DAY)
    assert [e["source_id"] for e in briefing.broken_promises] == ["CMP-T1"]
    assert any("response deadline (2026-09-25) has passed" in line
               for line in briefing.lines)


def test_briefing_flags_come_from_the_account_and_customer(seeded_store):
    account = seeded_store.get_account("NB-1008")  # Sam: vulnerable
    account.preferred_contact = "email"
    account.accessibility_note = "Needs letters in large print."
    seeded_store.save_account(account)
    briefing = continuity.build_briefing(seeded_store, "NB-1008", on=DAY)
    assert briefing.flags["vulnerable"] is True
    assert briefing.flags["preferred_contact"] == "email"
    assert briefing.flags["accessibility_note"] == \
        "Needs letters in large print."
    assert any("contact by email" in line for line in briefing.lines)


def test_briefing_last_outcome_from_the_attribution_record(seeded_store):
    seeded_store.save_conversation(Conversation(
        conversation_id="CONV-OLD", account_number="NB-1002",
        status="closed", intent=Intent.BILLING))
    seeded_store.save_attribution(AttributionRecord(
        conversation_id="CONV-OLD", outcome="resolved", desk="billing",
        attributed_date="2026-09-30"))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert briefing.last_outcome == \
        "Last time we spoke, we resolved your billing matter here."
    assert briefing.last_outcome_source == "CONV-OLD"
    assert briefing.lines[0] == briefing.last_outcome


def test_briefing_last_outcome_falls_back_to_the_conversation(seeded_store):
    seeded_store.save_conversation(Conversation(
        conversation_id="CONV-OLD", account_number="NB-1002",
        status="escalated", intent=Intent.ACCOUNT))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert briefing.last_outcome == \
        "Last time we spoke, we passed your account matter to a person."


def test_briefing_names_a_pending_proposal_from_its_record(seeded_store):
    proposal = ProposedAction(
        kind=ActionKind.CHANGE_PLAN,
        summary="switch you to the Family plan at $95.00 a month")
    seeded_store.save_conversation(Conversation(
        conversation_id="CONV-OLD", account_number="NB-1002",
        status="active", pending_action=proposal))
    seeded_store.save_action(ActionRecord(
        action_id="ACT-T1", conversation_id="CONV-OLD",
        kind=ActionKind.CHANGE_PLAN, summary=proposal.summary,
        status="proposed", account_number="NB-1002",
        created_date="2026-09-28"))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert briefing.pending_proposal["summary"] == proposal.summary
    assert any("waiting for a decision" in line
               for line in briefing.lines)


def test_briefing_ignores_a_proposal_that_was_answered(seeded_store):
    # The conversation no longer holds the proposal: every settle
    # path clears it, so nothing is pending whatever the old
    # action record's state.
    seeded_store.save_conversation(Conversation(
        conversation_id="CONV-OLD", account_number="NB-1002",
        status="resolved", pending_action=None))
    seeded_store.save_action(ActionRecord(
        action_id="ACT-T1", conversation_id="CONV-OLD",
        kind=ActionKind.CHANGE_PLAN, summary="switch plans",
        status="proposed", account_number="NB-1002",
        created_date="2026-09-28"))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert briefing.pending_proposal == {}


def test_briefing_resumable_marker_from_the_run_record(seeded_store):
    seeded_store.save_procedure_run(ProcedureRun(
        run_id="RUN-T1", conversation_id="CONV-OLD",
        account_number="NB-1002", procedure_id="PROC-BROADBAND",
        state="active", current_step_id="BB-2",
        created_date="2026-09-30", updated_date="2026-09-30"))
    briefing = continuity.build_briefing(
        seeded_store, "NB-1002", on=DAY, exclude_conversation_id="CONV-NEW")
    assert briefing.resumable_run_id == "RUN-T1"
    assert "BB-2" in briefing.resumable_summary
    assert any("partway through" in line for line in briefing.lines)


def test_briefing_empty_for_an_account_with_no_history(seeded_store):
    briefing = continuity.build_briefing(seeded_store, "NB-1009", on=DAY)
    assert briefing.open_items == []
    assert briefing.broken_promises == []
    assert briefing.last_outcome == ""
    assert briefing.resumable_run_id == ""
    assert briefing.pending_proposal == {}
    assert briefing.lines == []  # no fabricated continuity


# ---------------------------------------------------------------------------
# Flows — the briefing in the conversation, the run resumed
# ---------------------------------------------------------------------------

def test_returning_customer_is_opened_with_their_open_complaint(service):
    first = service.start_conversation()
    _say(service, first.conversation_id,
         "I want to make a complaint — your engineer never showed up "
         "for Tuesday's appointment and nobody told me. "
         "My account is NB-1003.")
    complaint = service.store.list_complaints(account_number="NB-1003")[0]

    second = service.start_conversation()
    result = _identify_after_question(service, second.conversation_id)
    assert "still open" in result.reply
    assert "due to respond by 2026-10-06" in result.reply
    snapshot = service.store.get_conversation(
        second.conversation_id).specialist_state["continuity"]["briefing"]
    assert [i["ref_id"] for i in snapshot["open_items"]] == \
        [complaint.complaint_id]

    # Delivered once: the next turn answers the question asked,
    # without re-reading the briefing.
    follow = _say(service, second.conversation_id,
                  "And what data allowance do I have?")
    assert "still open" not in follow.reply


def _identify_after_question(service, conv_id, account="NB-1003",
                             postcode="10001"):
    _say(service, conv_id, "Why is my bill higher than usual?")
    return _identify(service, conv_id, account, postcode)


def test_broken_promise_surfaces_in_the_next_conversations_opening(service):
    _complaint(service.store, account="NB-1002", conv="CONV-EARLIER",
               created="2026-09-20", due="2026-09-25")
    conv = service.start_conversation()
    result = _identify_after_question(
        service, conv.conversation_id, "NB-1002", "90210")
    assert "response deadline (2026-09-25) has passed" in result.reply
    assert "flagged on your record" in result.reply


def test_mid_procedure_conversation_resumes_the_same_run(service):
    first = service.start_conversation()
    _say(service, first.conversation_id, "my broadband keeps dropping")
    opened = _identify(service, first.conversation_id, "NB-1006", "30301")
    assert "unplug" in opened.reply  # parked at the safety step
    run = service.store.list_procedure_runs(
        conversation_id=first.conversation_id)[0]
    assert run.state == "active" and run.current_step_id == "BB-2"

    second = service.start_conversation()
    _say(service, second.conversation_id, "my broadband keeps dropping")
    resumed = _identify(service, second.conversation_id, "NB-1006", "30301")
    assert "partway through" in resumed.reply

    runs = [r for r in service.store.list_procedure_runs()
            if r.account_number == "NB-1006"]
    assert len(runs) == 1  # the stored run, not a restart
    assert runs[0].run_id == run.run_id
    assert runs[0].conversation_id == second.conversation_id  # adopted

    _say(service, second.conversation_id, "done, it's unplugged")
    done = _say(service, second.conversation_id, "it says 42")
    assert done.status == "resolved"
    final = service.store.get_procedure_run(run.run_id)
    assert final.state == "resolved"
    assert [h.step_id for h in final.history] == \
        ["BB-1", "BB-2", "BB-3", "BB-7"]


def test_callback_booked_after_a_dead_end_lands_on_the_account(service):
    # The no-signal walk dead-ends before identity, so the
    # callback record is born without an account; settling the
    # booking in the verified conversation stamps it — the
    # promise stands on the account it was made to, and the
    # account's ledger reads it as kept.
    conv = service.start_conversation()
    for text in ("I have no signal at all", "no, still nothing",
                 "nope, still no signal", "no, that didn't work either"):
        _say(service, conv.conversation_id, text)
    _identify(service, conv.conversation_id, "NB-1004", "60614")
    _say(service, conv.conversation_id, "yes")
    callback = service.store.list_callbacks()[0]
    assert callback.status == "booked"
    assert callback.account_number == "NB-1004"
    entries = continuity.promise_ledger(service.store, "NB-1004", on=DAY)
    assert [(e.kind, e.state) for e in entries] == [("callback", "kept")]


def test_no_history_customer_gets_an_ordinary_opening(service):
    conv = service.start_conversation()
    result = _identify_after_question(
        service, conv.conversation_id, "NB-1009", "98101")
    for fragment in ("still open", "partway through", "Last time we spoke",
                     "technician callback was booked", "response deadline"):
        assert fragment not in result.reply
    snapshot = service.store.get_conversation(
        conv.conversation_id).specialist_state["continuity"]["briefing"]
    assert snapshot["lines"] == [] and snapshot["open_items"] == []


# ---------------------------------------------------------------------------
# The digest and the views
# ---------------------------------------------------------------------------

def test_digest_surfaces_broken_promises(service):
    _complaint(service.store, account="NB-1002", conv="CONV-EARLIER",
               created="2026-09-20", due="2026-09-25")
    digest = quality.run_daily_pass(
        service.store, Policy(), on=DAY, with_spot_check=False)
    broken = digest.promises["broken"]
    assert digest.promises["broken_count"] == len(broken)
    assert any(e["source_id"] == "CMP-T1" for e in broken)
    assert any(e["source_id"] == "ARR-HIST-1015" for e in broken)
    markdown = quality.render_digest_markdown(digest)
    assert "## Promises at day end" in markdown
    assert "BROKEN:" in markdown and "2026-09-25" in markdown


def test_promises_service_and_continuity_views(service):
    entries = service.promise_ledger("NB-1015")
    assert any(e.state == "broken" for e in entries)
    assert service.conversation_continuity("CONV-NOPE") is None

    conv = service.start_conversation()
    view = service.conversation_continuity(conv.conversation_id)
    assert view["briefing"] is None and view["source"] == "none"
    _identify_after_question(service, conv.conversation_id,
                             "NB-1009", "98101")
    view = service.conversation_continuity(conv.conversation_id)
    assert view["source"] == "delivered"
    assert view["briefing"]["account_number"] == "NB-1009"


def test_promises_api():
    from fastapi.testclient import TestClient

    from cx_agent.api import app

    client = TestClient(app)
    client.post("/seed")
    body = client.get("/promises").json()
    assert body["broken_count"] >= 1
    assert any(p["account_number"] == "NB-1015" and p["state"] == "broken"
               for p in body["promises"])
    filtered = client.get("/promises",
                          params={"account_number": "NB-1015"}).json()
    assert all(p["account_number"] == "NB-1015"
               for p in filtered["promises"])
    conv = client.post("/conversations").json()
    view = client.get(
        f"/conversations/{conv['conversation_id']}/continuity")
    assert view.status_code == 200 and view.json()["briefing"] is None
    assert client.get("/conversations/NOPE/continuity").status_code == 404
