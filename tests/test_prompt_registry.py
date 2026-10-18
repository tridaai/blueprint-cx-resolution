"""Round 13 — the prompt registry: seeded with the book,
fail-closed, versioned under pins, and recorded on the
conversations it serves.

The lint's own tests live in test_prompt_lint.py; here the gate
appears only where an act depends on it.
"""

import os

import pytest

from cx_agent import prompts as prompts_mod
from cx_agent.llm import DraftContext, LiveLLM
from cx_agent.models import PromptEntry
from cx_agent.prompts import (
    COMPOSE_SYSTEM,
    INTENT_SYSTEM,
    PromptRegistry,
    PromptRegistryError,
)
from cx_agent.store import InMemoryStore, PostgresStore, SqliteStore


def _say(service, cid, text):
    return service.post_message(cid, text)


# -- the seeded registry ---------------------------------------------------


def test_seed_writes_the_whole_registry(seeded_store):
    registry = PromptRegistry(seeded_store)
    ids = registry.known_ids()
    assert len(ids) == 28  # 2 compose + 2 intent + 7 forms + 17 policies
    assert {"compose.system", "compose.user", "intent.system",
            "intent.user"} <= set(ids)
    assert sum(1 for i in ids if i.startswith("policy.")) == 17
    assert sum(1 for i in ids if i.startswith("explain.")) == 7
    for pid in ids:
        assert registry.active_version(pid) == 1
    assert registry.active_text("compose.system") == COMPOSE_SYSTEM
    assert registry.active_text("intent.system") == INTENT_SYSTEM


def test_seed_never_overwrites_an_operators_version(service):
    service.add_prompt_version(
        "intent.system", "Sort the message into exactly one label.")
    counts = service.seed()  # a re-seed
    assert counts["prompts"] == 28
    assert service.prompts.active_version("intent.system") == 2
    assert service.prompts.active_text("intent.system") == \
        "Sort the message into exactly one label."


# -- fail-closed loading -----------------------------------------------------


def test_missing_entry_refuses_with_its_name():
    registry = PromptRegistry(InMemoryStore())  # never seeded
    with pytest.raises(PromptRegistryError) as exc:
        registry.active_text("compose.system")
    assert "compose.system" in str(exc.value)


def test_malformed_entries_refuse_with_the_defect(seeded_store):
    seeded_store.save_prompt_entry(PromptEntry(
        prompt_id="compose.system", version=0, surface="compose",
        text="anything"))
    with pytest.raises(PromptRegistryError) as exc:
        PromptRegistry(seeded_store).entry_at("compose.system", 0)
    assert "malformed version" in str(exc.value)

    seeded_store.save_prompt_entry(PromptEntry(
        prompt_id="intent.system", version=7, surface="intent",
        text="   "))
    with pytest.raises(PromptRegistryError) as exc:
        PromptRegistry(seeded_store).active_entry("intent.system")
    assert "blank text" in str(exc.value)


def test_a_pin_to_a_missing_version_refuses(seeded_store):
    from cx_agent.models import PromptPin

    seeded_store.save_prompt_pin(PromptPin(
        prompt_id="compose.user", version=9))
    with pytest.raises(PromptRegistryError) as exc:
        PromptRegistry(seeded_store).active_entry("compose.user")
    assert "pinned to v9" in str(exc.value)


# -- the model boundary reads the registry -----------------------------------


def _stub_live_llm(store, reply="billing"):
    from cx_agent.config import LLMConfig

    llm = LiveLLM(LLMConfig(backend="openai", model="m", api_key="k",
                            base_url="http://stub"))
    llm.prompts = PromptRegistry(store)
    calls = []

    def fake_chat(system, user):
        calls.append((system, user))
        return reply

    llm._chat = fake_chat
    return llm, calls


def test_live_drafter_uses_the_registry_text(seeded_store):
    llm, calls = _stub_live_llm(seeded_store, reply="Here is your reply.")
    out = llm.draft_reply(DraftContext(
        findings=["Your bill is $94.35."], customer_message="why?"))
    assert out == "Here is your reply."
    system, user = calls[0]
    assert system == COMPOSE_SYSTEM
    assert user.startswith("Customer said: why?")
    assert "- Your bill is $94.35." in user
    assert "Lines that must appear verbatim:" in user


def test_live_drafter_follows_a_version_bump(service):
    llm, calls = _stub_live_llm(service.store, reply="ok")
    revised = COMPOSE_SYSTEM + " Speak plainly."
    service.add_prompt_version("compose.system", revised)
    llm.draft_reply(DraftContext(findings=["f"]))
    assert calls[0][0] == revised


def test_live_boundary_without_a_registry_refuses(seeded_store):
    from cx_agent.config import LLMConfig

    llm = LiveLLM(LLMConfig(backend="openai", model="m", api_key="k",
                            base_url="http://stub"))
    with pytest.raises(PromptRegistryError):
        llm.draft_reply(DraftContext(findings=["f"]))
    llm.prompts = PromptRegistry(InMemoryStore())  # wired but empty
    with pytest.raises(PromptRegistryError) as exc:
        llm.intent_hint("hello")
    assert "intent.system" in str(exc.value)


def test_intent_hint_uses_the_registry_text(seeded_store):
    llm, calls = _stub_live_llm(seeded_store, reply="billing")
    assert llm.intent_hint("my bill is wrong") == "billing"
    system, user = calls[0]
    assert system == INTENT_SYSTEM
    assert "Labels: billing, roaming, offers, technical, account, care" \
        in user
    assert user.endswith("Message: my bill is wrong")


# -- versioning, pins, acts ----------------------------------------------------


def test_active_is_highest_unless_pinned(service):
    registry = service.prompts
    # The replacement wordings keep the figure the lint declares
    # this form must state (three wrong codes) — a rewording that
    # dropped it would be refused by the gate, as it should be.
    service.add_prompt_version(
        "explain.POL-ID-01",
        "Short version: two steps, then the code. Three wrong "
        "codes lock the chat and a person takes over.")
    assert registry.active_version("explain.POL-ID-01") == 2
    service.rollback_prompt("explain.POL-ID-01", 1, actor="op")
    assert registry.active_version("explain.POL-ID-01") == 1
    assert registry.active_text("explain.POL-ID-01").startswith(
        "The short version: two steps.")
    service.add_prompt_version(
        "explain.POL-ID-01",
        "Short version: a third wording. Three wrong codes still "
        "lock the chat.")
    assert registry.active_version("explain.POL-ID-01") == 1  # pin holds
    service.unpin_prompt("explain.POL-ID-01")
    assert registry.active_version("explain.POL-ID-01") == 3


def test_acts_are_recorded_in_order(service):
    service.add_prompt_version(
        "intent.user", "Labels: {labels}. One label only.\n\nMessage: {text}",
        actor="op", note="tighten")
    service.rollback_prompt("intent.user", 1, actor="op")
    service.unpin_prompt("intent.user", actor="op")
    acts = service.store.list_prompt_acts("intent.user")
    assert [a.kind for a in acts] == ["add_version", "rollback", "unpin"]
    assert [a.seq for a in acts] == sorted(a.seq for a in acts)
    assert acts[0].actor == "op" and acts[0].note == "tighten"
    assert acts[1].version == 1
    assert all(a.created_date for a in acts)


def test_unknown_prompts_and_versions_are_refused(service):
    with pytest.raises(PromptRegistryError):
        service.add_prompt_version("compose.nope", "text")
    with pytest.raises(PromptRegistryError):
        service.rollback_prompt("compose.system", 99)
    with pytest.raises(PromptRegistryError):
        service.unpin_prompt("compose.system")  # not pinned


def test_policy_activation_rewrites_the_corpus(service):
    store = service.store
    before = store.get_policy("POL-ID-01")
    revised = before.text + " The code goes only to the contact on file."
    service.add_prompt_version("policy.POL-ID-01", revised)
    after = store.get_policy("POL-ID-01")
    assert after.text == revised
    assert after.title == before.title and after.section == before.section
    embeddings = dict(store.list_embeddings())
    assert len(embeddings["POL-ID-01"]) == 384
    service.rollback_prompt("policy.POL-ID-01", 1)
    assert store.get_policy("POL-ID-01").text == before.text


# -- provenance: the versions on the conversation ------------------------------


def test_conversation_records_the_versions_that_served_it(service):
    conv = service.start_conversation()
    cid = conv.conversation_id
    r1 = _say(service, cid, "What time do your stores open on Saturday?")
    assert "9am to 6pm" in r1.reply
    r2 = _say(service, cid, "I don't understand - what do you mean?")
    assert "The short version:" in r2.reply  # the second form was served
    conv = service.get_conversation(cid)
    assert conv.prompt_versions["compose.system"] == 1
    assert conv.prompt_versions["compose.user"] == 1
    assert conv.prompt_versions["policy.POL-CARE-01"] == 1
    assert conv.prompt_versions["explain.POL-CARE-01"] == 1
    rows = service.conversation_prompts(cid)
    by_id = {row["prompt_id"]: row for row in rows}
    assert by_id["explain.POL-CARE-01"]["version"] == 1
    assert "The short version:" in by_id["explain.POL-CARE-01"]["text"]


def test_bump_and_rollback_flow_through_conversations(service):
    def one_conversation():
        conv = service.start_conversation()
        _say(service, conv.conversation_id, "Hello, I need some help.")
        return service.get_conversation(conv.conversation_id)

    first = one_conversation()
    assert first.prompt_versions["compose.system"] == 1
    service.add_prompt_version(
        "compose.system", COMPOSE_SYSTEM + " Speak plainly.")
    second = one_conversation()
    assert second.prompt_versions["compose.system"] == 2
    service.rollback_prompt("compose.system", 1)
    third = one_conversation()
    assert third.prompt_versions["compose.system"] == 1
    # History does not move: the first conversation still records v1,
    # the second still records the v2 that served it.
    assert service.get_conversation(
        first.conversation_id).prompt_versions["compose.system"] == 1
    assert service.get_conversation(
        second.conversation_id).prompt_versions["compose.system"] == 2


# -- the registry on every store ------------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "sqlite", "postgres"])
def test_registry_round_trip_on_every_store(kind, tmp_path):
    stores = {"memory": InMemoryStore(),
              "sqlite": SqliteStore(tmp_path / "cx.db")}
    dsn = os.environ.get("CX_TEST_DATABASE_URL")
    if dsn:
        # The PostgreSQL leg shares one scratch database across the
        # suite; wipe it so this test's version counts start from the
        # seed alone (same helper as the store conformance file).
        from test_store_conformance import _fresh_postgres

        stores["postgres"] = _fresh_postgres(dsn)
    if kind not in stores:
        pytest.skip("CX_TEST_DATABASE_URL not set")
    store = stores[kind]
    from cx_agent.dataset import seed_store

    seed_store(store)
    prompts_mod.add_version(store, _policy(), "intent.system",
                            "Sort the message into exactly one label.",
                            actor="test")
    prompts_mod.rollback(store, _policy(), "intent.system", 1,
                         actor="test")
    registry = PromptRegistry(store)
    assert registry.active_version("intent.system") == 1
    assert len(registry.versions("intent.system")) == 2
    assert [a.kind for a in store.list_prompt_acts("intent.system")] == [
        "add_version", "rollback"]
    if kind == "sqlite":
        reopened = SqliteStore(tmp_path / "cx.db")
        assert PromptRegistry(reopened).active_version(
            "intent.system") == 1
        assert len(reopened.list_prompt_entries()) == 29


def _policy():
    from cx_agent.config import Policy

    return Policy()


# -- the API surface -----------------------------------------------------------


def test_prompts_on_the_api():
    from fastapi.testclient import TestClient

    from cx_agent.api import app

    client = TestClient(app)
    client.post("/seed")
    rows = client.get("/prompts").json()["prompts"]
    assert len(rows) == 28
    detail = client.get("/prompts/compose.system").json()
    assert detail["prompt_id"] == "compose.system"
    assert detail["versions"][0]["text"] == COMPOSE_SYSTEM
    assert client.get("/prompts/compose.nope").status_code == 404

    refused = client.post("/prompts/compose.user/versions", json={
        "text": "Facts only. We guarantee a full refund."})
    assert refused.status_code == 400
    findings = str(refused.json()["detail"])
    assert "banned_promise" in findings

    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "What time do your stores open?"})
    served = client.get(f"/conversations/{cid}/prompts").json()
    by_id = {row["prompt_id"]: row for row in served["prompts"]}
    assert by_id["compose.system"]["version"] >= 1
    assert by_id["policy.POL-CARE-01"]["text"].startswith(
        "Northbeam stores are open")
    assert client.get("/conversations/CONV-nope/prompts").status_code == 404
