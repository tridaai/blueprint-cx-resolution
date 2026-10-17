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
    assert store.count_customers() == 14

    conv = Conversation(conversation_id="CONV-X")
    store.save_conversation(conv)
    loaded = store.get_conversation("CONV-X")
    assert loaded.conversation_id == "CONV-X"
    assert loaded.identity_state.value == "unverified"

    vectors = store.list_embeddings()
    assert len(vectors) == 14
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
