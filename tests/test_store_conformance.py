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
    assert store.count_customers() == 10

    conv = Conversation(conversation_id="CONV-X")
    store.save_conversation(conv)
    loaded = store.get_conversation("CONV-X")
    assert loaded.conversation_id == "CONV-X"
    assert loaded.identity_state.value == "unverified"

    vectors = store.list_embeddings()
    assert len(vectors) == 12
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
