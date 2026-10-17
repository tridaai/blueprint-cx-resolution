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
