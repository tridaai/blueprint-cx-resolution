import pytest

from cx_agent.service import ConversationService
from cx_agent.store import InMemoryStore


@pytest.fixture()
def store():
    return InMemoryStore()


@pytest.fixture()
def service(store):
    svc = ConversationService(store=store)
    svc.seed()
    return svc


@pytest.fixture()
def seeded_store(store):
    from cx_agent.dataset import seed_store

    seed_store(store)
    return store
