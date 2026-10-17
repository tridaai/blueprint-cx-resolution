"""The same store contract on every implementation: payload
round-trips keep Decimals exact, embeddings enforce their
dimension, and audit/action listings behave identically. The
PostgreSQL leg runs only when CX_TEST_DATABASE_URL names a
scratch database."""

import os
from decimal import Decimal

import pytest

from cx_agent.dataset import seed_store
from cx_agent.documents import embed
from cx_agent.models import Conversation
from cx_agent.store import (
    EmbeddingDimensionError,
    InMemoryStore,
    PostgresStore,
    SqliteStore,
)


def _stores(tmp_path):
    yield "memory", InMemoryStore()
    yield "sqlite", SqliteStore(tmp_path / "cx.db")
    dsn = os.environ.get("CX_TEST_DATABASE_URL")
    if dsn:
        yield "postgres", PostgresStore(dsn)


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_store_contract(kind, tmp_path):
    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    account = store.get_account("NB-1001")
    assert account.balance_due == Decimal("94.35")  # Decimal survived exactly
    bill = store.get_bill_for_account("NB-1001")
    assert bill.total == Decimal("94.35")
    assert store.get_customer_by_account("NB-1001").full_name == "Amara Osei"
    assert store.count_customers() == 15

    conv = Conversation(conversation_id="CONV-X")
    store.save_conversation(conv)
    loaded = store.get_conversation("CONV-X")
    assert loaded.conversation_id == "CONV-X"
    assert loaded.identity_state.value == "unverified"

    vectors = store.list_embeddings()
    assert len(vectors) == 17
    assert all(len(v) == 384 for _d, v in vectors)
    with pytest.raises(EmbeddingDimensionError):
        store.set_embedding("POL-BILL-01", [0.1, 0.2])


def test_sqlite_is_the_local_default_profile(tmp_path):
    store = SqliteStore(tmp_path / "cx.db")
    assert store.profile == "sqlite"
    assert (tmp_path / "cx.db").exists()


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_bill_history_and_dispute_cases(kind, tmp_path):
    """Round 3 records behave identically on every store: bills are
    a per-account history ordered by issue date (latest wins), and
    dispute cases round-trip with their per-line decisions."""
    from decimal import Decimal as D

    from cx_agent.models import Bill, BillCharge, DisputeCase, DisputeLine

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]

    older = Bill(bill_id="BILL-T-OLD", account_number="NB-T",
                 cycle_start="2026-08-13", cycle_end="2026-09-11",
                 issued="2026-09-12", due="2026-10-03",
                 charges=[BillCharge(charge_id="C1", kind="plan",
                                     description="Plan", amount=D("40.00"))])
    newer = Bill(bill_id="BILL-T-NEW", account_number="NB-T",
                 cycle_start="2026-09-12", cycle_end="2026-10-11",
                 issued="2026-10-12", due="2026-11-02",
                 charges=[BillCharge(charge_id="C2", kind="plan",
                                     description="Plan", amount=D("45.00"))])
    store.save_bill(newer)
    store.save_bill(older)  # saved out of order; reads must sort by issue date
    history = store.list_bills_for_account("NB-T")
    assert [b.bill_id for b in history] == ["BILL-T-OLD", "BILL-T-NEW"]
    assert store.get_bill_for_account("NB-T").bill_id == "BILL-T-NEW"
    assert store.list_bills_for_account("NB-NONE") == []
    assert store.get_bill_for_account("NB-NONE") is None

    case = DisputeCase(
        case_id="DSP-T1", conversation_id="CONV-T", account_number="NB-T",
        bill_id="BILL-T-NEW", state="decided_partial",
        lines=[DisputeLine(charge_id="C2", description="Plan",
                           amount=D("45.00"), disputed_amount=D("5.00"),
                           decision="declined", reason="matches the plan price")],
        credit_applied=D("0"), created_date="2026-10-20",
        updated_date="2026-10-20")
    store.save_dispute_case(case)
    loaded = store.get_dispute_case("DSP-T1")
    assert loaded.state == "decided_partial"
    assert loaded.lines[0].decision == "declined"
    assert store.list_dispute_cases(conversation_id="CONV-T")[0].case_id == "DSP-T1"
    assert store.list_dispute_cases(account_number="NB-T")[0].case_id == "DSP-T1"
    assert store.list_dispute_cases(account_number="NB-OTHER") == []
    assert store.get_dispute_case("DSP-NONE") is None


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_procedure_versions_runs_and_callbacks(kind, tmp_path):
    """Round 4 records behave identically on every store: procedures
    are versioned data (the plain read is the latest, a pinned read
    is exact, the list shows one row per procedure), and procedure
    runs and callback records round-trip with their transcripts."""
    from cx_agent.models import (
        CallbackRecord,
        Procedure,
        ProcedureRun,
        ProcedureStep,
        StepRecord,
    )

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]

    v1 = Procedure(procedure_id="PROC-T", version=1, title="Test",
                   symptom="test",
                   steps=[ProcedureStep(step_id="T-1", instruction="one")])
    v2 = Procedure(procedure_id="PROC-T", version=2, title="Test",
                   symptom="test",
                   steps=[ProcedureStep(step_id="T-1", instruction="one, revised")])
    store.save_procedure(v1)
    store.save_procedure(v2)
    assert store.get_procedure("PROC-T").version == 2
    assert store.get_procedure_version("PROC-T", 1).steps[0].instruction == "one"
    assert store.get_procedure_version("PROC-T", 3) is None
    assert [p.procedure_id for p in store.list_procedures()] == ["PROC-T"]

    history = [StepRecord(step_id="T-1", kind="tool_check",
                          detail="line_test", result="fault_found", turn=2)]
    run = ProcedureRun(run_id="RUN-T1", conversation_id="CONV-T",
                       account_number="NB-T", procedure_id="PROC-T",
                       procedure_version=1, state="dead_ended",
                       history=history, outcome="field_visit",
                       created_date="2026-10-20", updated_date="2026-10-20")
    store.save_procedure_run(run)
    loaded_run = store.get_procedure_run("RUN-T1")
    assert loaded_run.state == "dead_ended"
    assert loaded_run.history[0].result == "fault_found"
    assert store.list_procedure_runs(conversation_id="CONV-T")[0].run_id == "RUN-T1"
    assert store.list_procedure_runs(conversation_id="CONV-OTHER") == []
    assert store.get_procedure_run("RUN-NONE") is None

    callback = CallbackRecord(
        callback_id="CBK-T1", conversation_id="CONV-T",
        account_number="NB-T", procedure_id="PROC-T", procedure_version=1,
        run_id="RUN-T1", reason="steps exhausted", status="proposed",
        transcript=history, created_date="2026-10-20")
    store.save_callback(callback)
    loaded_cb = store.get_callback("CBK-T1")
    assert loaded_cb.transcript[0].step_id == "T-1"
    assert store.list_callbacks(conversation_id="CONV-T")[0].callback_id == "CBK-T1"
    assert store.list_callbacks() != []
    assert store.get_callback("CBK-NONE") is None


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_payment_records(kind, tmp_path):
    """Round 7 records behave identically on every store: attempts
    keep their order and their Decimal amounts, arrangements
    round-trip with their installments, notices with their text."""
    from cx_agent.models import (
        ArrangementInstallment,
        PaymentArrangement,
        PaymentAttempt,
        PaymentNotice,
    )

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]

    store.save_payment_attempt(PaymentAttempt(
        attempt_id="ATT-T1", conversation_id="CONV-T",
        account_number="NB-T", method_last4="1234",
        amount=Decimal("45.00"), outcome="declined",
        reason_code="insufficient_funds", decline_class="soft",
        source="manual", created_date="2026-10-01"))
    store.save_payment_attempt(PaymentAttempt(
        attempt_id="ATT-T2", conversation_id="CONV-T",
        account_number="NB-T", method_last4="1234",
        amount=Decimal("45.00"), outcome="approved",
        source="manual", created_date="2026-10-01"))
    rows = store.list_payment_attempts("NB-T")
    assert [t.attempt_id for t in rows] == ["ATT-T1", "ATT-T2"]
    assert rows[0].amount == Decimal("45.00")
    assert rows[0].decline_class == "soft"
    assert store.list_payment_attempts("NB-OTHER") == []
    assert len(store.list_payment_attempts(conversation_id="CONV-T")) == 2

    arrangement = PaymentArrangement(
        arrangement_id="ARR-T1", conversation_id="CONV-T",
        account_number="NB-T", total=Decimal("100.00"),
        installments=[ArrangementInstallment(
            number=1, amount=Decimal("33.34"), due_date="2026-10-15"),
            ArrangementInstallment(
                number=2, amount=Decimal("33.33"), due_date="2026-11-14")],
        state="active", created_date="2026-10-01")
    store.save_payment_arrangement(arrangement)
    loaded = store.get_payment_arrangement("ARR-T1")
    assert loaded.total == Decimal("100.00")
    assert loaded.installments[0].amount == Decimal("33.34")
    assert [a.arrangement_id for a in
            store.list_payment_arrangements("NB-T")] == ["ARR-T1"]
    assert store.get_payment_arrangement("ARR-NONE") is None

    store.save_payment_notice(PaymentNotice(
        notice_id="NTF-T9", account_number="NB-T", kind="autopay_failed",
        amount=Decimal("45.00"), method_last4="1234",
        reason_code="insufficient_funds", summary="Autopay failed…",
        created_date="2026-10-01"))
    notices = store.list_payment_notices("NB-T")
    assert len(notices) == 1 and notices[0].amount == Decimal("45.00")
    assert store.list_payment_notices("NB-OTHER") == []


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_roaming_rates_and_notices(kind, tmp_path):
    """Round 6 records behave identically on every store: rates
    key on the destination (case-insensitively), and notices
    round-trip with their Decimal figures and the unwarned flag."""
    from cx_agent.models import RoamingNotice, RoamingRate

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]

    store.save_roaming_rate(RoamingRate(destination="Spain",
                                        per_gb=Decimal("8.50")))
    loaded = store.get_roaming_rate("spain")
    assert loaded is not None and loaded.per_gb == Decimal("8.50")
    assert store.get_roaming_rate("Atlantis") is None
    assert [r.destination for r in store.list_roaming_rates()] == ["Spain"]

    notice = RoamingNotice(
        notice_id="NTF-T1", account_number="NB-T", kind="fair_use_100",
        destination="Spain", trip_start="2026-09-28",
        used_gb=Decimal("26"), cap_gb=Decimal("25"), pct=Decimal("104.0"),
        projected_overage_cost=Decimal("340.00"),
        summary="Fair-use warning…", conversation_id="CONV-T",
        unwarned_crossing=True, created_date="2026-10-01")
    store.save_roaming_notice(notice)
    rows = store.list_roaming_notices("NB-T")
    assert len(rows) == 1
    assert rows[0].projected_overage_cost == Decimal("340.00")
    assert rows[0].unwarned_crossing is True
    assert store.list_roaming_notices("NB-OTHER") == []
    assert len(store.list_roaming_notices()) == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_care_records(kind, tmp_path):
    """Round 8 records — complaints (with the watch's queue entry
    riding the record) and feedback — behave identically on every
    store, and the account carries its accessibility record."""
    from cx_agent.models import ComplaintRecord, FeedbackRecord

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)
    complaint = ComplaintRecord(
        complaint_id="CMP-T1", conversation_id="CONV-T",
        account_number="NB-1001", category="appointment",
        text="your engineer never showed up", status="open",
        created_date="2026-10-01", response_due="2026-10-06",
        sla_escalation_id="ESC-T1")
    store.save_complaint(complaint)
    assert store.get_complaint("CMP-T1") == complaint
    assert store.list_complaints(conversation_id="CONV-T") == [complaint]
    assert store.list_complaints(account_number="NB-1002") == []
    store.save_complaint(store.get_complaint("CMP-T1"))
    assert len(store.list_complaints()) == 1

    feedback = FeedbackRecord(
        record_id="FBK-T1", conversation_id="CONV-T",
        account_number="NB-1001", kind="compliment",
        text="the engineer was brilliant", created_date="2026-10-01")
    store.save_feedback(feedback)
    assert store.list_feedback(conversation_id="CONV-T") == [feedback]
    assert store.list_feedback(account_number="NB-1002") == []

    account = store.get_account("NB-1001")
    account.accessibility_note = "send my letters in large print"
    account.preferred_contact = "large print"
    store.save_account(account)
    read_account = store.get_account("NB-1001")
    assert read_account.accessibility_note == \
        "send my letters in large print"
    assert read_account.preferred_contact == "large print"


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_attribution_and_annotation_records(kind, tmp_path):
    """Round 10 records behave identically on every store:
    attributions round-trip with their evidence dicts, keyed by
    conversation, and annotations list in creation (seq) order,
    filtered by conversation."""
    from cx_agent.models import AnnotationRecord, AttributionRecord

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    record = AttributionRecord(
        conversation_id="CONV-T", outcome="escalated",
        cause="policy_limit", rule="dispute_above_cap", desk="billing",
        evidence={"reroutes": 0, "escalation_reason": "dispute_above_agent_cap"},
        attributed_date="2026-10-01")
    store.save_attribution(record)
    assert store.get_attribution("CONV-T") == record
    assert store.list_attributions() == [record]
    assert store.get_attribution("CONV-NONE") is None
    store.save_attribution(store.get_attribution("CONV-T"))
    assert len(store.list_attributions()) == 1

    second = AnnotationRecord(
        annotation_id="ANN-T2", conversation_id="CONV-T", annotator="ben",
        cause_correct=False, corrected_cause="routing_error",
        note="bounced first", seq=2, created_date="2026-10-01")
    first = AnnotationRecord(
        annotation_id="ANN-T1", conversation_id="CONV-T", annotator="amy",
        cause_correct=True, seq=1, created_date="2026-10-01")
    store.save_annotation(second)  # saved out of order; reads sort by seq
    store.save_annotation(first)
    assert store.list_annotations("CONV-T") == [first, second]
    assert store.list_annotations() == [first, second]
    assert store.list_annotations("CONV-OTHER") == []


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_signal_records(kind, tmp_path):
    """Round 11 records behave identically on every store: one
    SignalRecord per turn, listed in turn order per conversation,
    with the effort components and the intervention entries
    intact, and re-saving a turn's record replaces it."""
    from cx_agent.models import SignalRecord

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    second = SignalRecord(
        record_id="SIG-CONV-T-2", conversation_id="CONV-T", turn=2,
        desk="billing", frustration_score=4, frustration_rising=True,
        effort_score=6, effort_turns=2, effort_identity_attempts=2,
        effort_reroutes=1, effort_repeats=1, progress=False,
        interventions=[{"kind": "frustration_human_offer",
                        "outcome": "offered"}])
    first = SignalRecord(
        record_id="SIG-CONV-T-1", conversation_id="CONV-T", turn=1,
        desk="billing", frustration_score=1, effort_score=1,
        effort_turns=1, progress=True, findings_count=2)
    store.save_signal_record(second)  # saved out of order
    store.save_signal_record(first)
    other = SignalRecord(
        record_id="SIG-CONV-O-1", conversation_id="CONV-O", turn=1,
        effort_score=1, effort_turns=1)
    store.save_signal_record(other)

    assert store.list_signal_records("CONV-T") == [first, second]
    assert store.list_signal_records() == [other, first, second]
    assert store.list_signal_records("CONV-NONE") == []

    first.progress = False
    store.save_signal_record(first)
    assert len(store.list_signal_records("CONV-T")) == 2
    assert store.list_signal_records("CONV-T")[0].progress is False


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_retention_cases(kind, tmp_path):
    """Round 14 records behave identically on every store: a
    retention case round-trips with its reason, its one save
    offer and its outcome, lists filter by conversation and
    account, and the account carries its scheduled cancellation
    date in its own payload."""
    from cx_agent.models import RetentionCase

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    case = RetentionCase(
        case_id="RET-T1", conversation_id="CONV-T",
        account_number="NB-1001", reason="price",
        reason_text="I want to cancel, it's too expensive",
        state="cancelled", outcome="cancelled",
        save_offer_mechanism="plan_downgrade",
        save_offer_summary="change your plan to Basic Talk & Text",
        save_offer_outcome="declined",
        cancellation_effective="2026-10-13",
        created_date="2026-10-01", updated_date="2026-10-01")
    store.save_retention_case(case)
    assert store.get_retention_case("RET-T1") == case
    assert store.list_retention_cases(conversation_id="CONV-T") == [case]
    assert store.list_retention_cases(account_number="NB-1001") == [case]
    assert store.list_retention_cases(account_number="NB-1002") == []
    assert store.get_retention_case("RET-NONE") is None

    account = store.get_account("NB-1001")
    account.pending_cancellation_effective = "2026-10-13"
    store.save_account(account)
    assert store.get_account("NB-1001").pending_cancellation_effective \
        == "2026-10-13"


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_incident_feed_and_act_log(kind, tmp_path):
    """Round 15 records behave identically on every store: an
    incident round-trips with its scope and declared signals,
    the feed lists in id order, and the act log lists in seq
    order, filtered by incident."""
    from cx_agent.models import Incident, IncidentAct

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    record = Incident(
        incident_id="INC-T1", title="Signal down — West Loop",
        service="no_signal", affected_postcodes=["60614"],
        symptom_signals=["no signal", "no service"],
        status="active", started_at="2026-10-01",
        summary="Signal is down around the West Loop.",
        updated_date="2026-10-01")
    store.save_incident(record)
    assert store.get_incident("INC-T1") == record
    assert store.get_incident("INC-NONE") is None
    ids = [i.incident_id for i in store.list_incidents()]
    assert ids == sorted(ids) and "INC-T1" in ids

    second = IncidentAct(act_id="IACT-T2", seq=2, kind="resolve",
                         incident_id="INC-T1", actor="ops",
                         created_date="2026-10-01")
    first = IncidentAct(act_id="IACT-T1", seq=1, kind="add",
                        incident_id="INC-T1", actor="ops",
                        created_date="2026-10-01")
    store.save_incident_act(second)  # saved out of order; reads sort by seq
    store.save_incident_act(first)
    assert store.list_incident_acts("INC-T1") == [first, second]
    assert store.list_incident_acts() == [first, second]
    assert store.list_incident_acts("INC-OTHER") == []


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_quality_digests(kind, tmp_path):
    """Round 12 records behave identically on every store: one
    digest per date, listed in date order, and saving a digest
    for a date that already has one replaces it — the version
    on the record is what counts the replacement."""
    from cx_agent.models import DailyDigest

    stores = dict(_stores(tmp_path))
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    seed_store(store)

    later = DailyDigest(date="2026-10-01", version=1, conversations=12,
                        cause_mix={"policy_limit": 1},
                        spot_check={"drift_count": 0})
    earlier = DailyDigest(date="2026-09-30", version=1, conversations=8)
    store.save_digest(later)  # saved out of order; reads sort by date
    store.save_digest(earlier)
    assert store.list_digests() == [earlier, later]
    assert store.get_digest("2026-10-01") == later
    assert store.get_digest("2026-09-01") is None

    replaced = DailyDigest(date="2026-10-01", version=2, conversations=12,
                           cause_mix={"policy_limit": 1},
                           spot_check={"drift_count": 1})
    store.save_digest(replaced)
    assert len(store.list_digests()) == 2
    assert store.get_digest("2026-10-01") == replaced
