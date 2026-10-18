"""Round 18 — cost & latency discipline.

Accounting as records, not counters: every composed turn writes
its model-call record onto the audit trail, and the budget, the
report and the digest all fold those records. A crossed cap
closes the NEXT turn, gracefully; a call that times out twice
closes the turn to a person without an answer being composed
from nothing. The timeout tests use the live boundary with a
stubbed transport — the only place a timeout can occur in the
suite, since the offline backend never times out.
"""

from __future__ import annotations

import httpx
import pytest

from cx_agent import attribution, costs
from cx_agent.config import LLMConfig, Policy
from cx_agent.llm import DraftContext, LiveLLM, ModelTimeoutError, estimate_tokens
from cx_agent.prompts import PromptRegistry, seed_registry
from cx_agent.service import ConversationService
from cx_agent.store import InMemoryStore


# ---------------------------------------------------------------------------
# Per-turn accounting
# ---------------------------------------------------------------------------

def test_composed_turns_write_call_records(service):
    cid = service.start_conversation().conversation_id
    service.post_message(cid, "What time do your stores open on Saturday?")
    service.post_message(cid, "And what is your complaints process?")
    events = [e for e in service.store.list_audit(cid)
              if e.kind == "model_call"]
    assert len(events) == 2
    for event in events:
        d = event.detail
        assert d["call_kind"] == "decision"
        assert d["backend"] == "offline"
        assert d["tokens_basis"] == "estimated"
        assert d["outcome"] == "ok"
        assert d["tokens"] == d["tokens_in"] + d["tokens_out"]
        assert d["latency_ms"] >= 0
        assert d["desk"] and d["date"]  # stamped by the drain


def test_usage_folds_from_records_and_the_report_reconciles(service):
    cid1 = service.start_conversation().conversation_id
    service.post_message(cid1, "What time do your stores open on Saturday?")
    service.post_message(cid1, "And what is your complaints process?")
    cid2 = service.start_conversation().conversation_id
    service.post_message(cid2, "why is my bill so high")

    total_tokens = 0
    for cid in (cid1, cid2):
        usage = costs.conversation_usage(service.store, cid)
        raw = [e.detail for e in service.store.list_audit(cid)
               if e.kind == "model_call"]
        assert usage["calls"] == len(raw)
        assert usage["decision_calls"] == len(raw)
        assert usage["tokens"] == sum(d["tokens"] for d in raw)
        assert usage["tokens_estimated"] == usage["tokens"]  # offline
        assert usage["tokens_measured"] == 0
        total_tokens += usage["tokens"]

    report = service.cost_report()
    assert report["overall"]["calls"] == 3
    assert report["overall"]["tokens"] == total_tokens
    # the legs sum to the total — one fold, no second set of books
    assert sum(r["calls"] for r in report["by_desk"].values()) == 3
    assert sum(r["tokens"] for r in report["by_day"].values()) == total_tokens
    assert report["overall"]["mean_latency_ms"] is not None
    assert report["overall"]["p95_latency_ms"] is not None
    assert report["budget_exhaustions"] == 0
    assert report["timeout_escalations"] == 0
    assert report["caps"]["max_decision_calls_per_conversation"] == 40


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


# ---------------------------------------------------------------------------
# Conversation budgets
# ---------------------------------------------------------------------------

def _budget_service(**caps) -> ConversationService:
    policy = Policy(**{"max_decision_calls_per_conversation": 40,
                       "max_estimated_tokens_per_conversation": 100_000,
                       **caps})
    svc = ConversationService(store=InMemoryStore(), policy=policy)
    svc.seed()
    return svc


def test_the_call_cap_closes_the_next_turn_gracefully():
    svc = _budget_service(max_decision_calls_per_conversation=2)
    cid = svc.start_conversation().conversation_id
    svc.post_message(cid, "What time do your stores open on Saturday?")
    svc.post_message(cid, "And what is your complaints process?")
    assert costs.conversation_usage(
        svc.store, cid)["decision_calls"] == 2

    result = svc.post_message(cid, "One more thing — do you have parking?")
    assert result.escalated is True
    assert result.escalation_reason == "budget_exhausted"
    assert "a person will pick it up from here" in result.reply
    # the closing turn spent nothing more
    assert costs.conversation_usage(
        svc.store, cid)["decision_calls"] == 2
    # the cap event is audited with the observed usage
    budget = [e for e in svc.store.list_audit(cid) if e.kind == "budget"]
    assert len(budget) == 1
    assert budget[0].detail["event"] == "budget_exhausted"
    assert budget[0].detail["decision_calls"] == 2
    assert budget[0].detail["max_decision_calls"] == 2
    # a handoff exists, and its packet is graded as usual
    escs = [e for e in svc.store.list_escalations()
            if e.conversation_id == cid]
    assert len(escs) == 1 and escs[0].reason == "budget_exhausted"
    esc_audit = [e for e in svc.store.list_audit(cid)
                 if e.kind == "escalation"]
    assert esc_audit and "packet_complete" in esc_audit[-1].detail
    # attribution reads the closing as the design working
    outcome = attribution.attribute_conversation(svc.store, cid)
    assert outcome.rule == "escalated_by_design:budget_exhausted"
    assert outcome.cause is None
    # the report counts the closing, and the conversation stays closed
    assert svc.cost_report()["budget_exhaustions"] == 1
    after = svc.post_message(cid, "hello?")
    assert after.outcome == "already_escalated"


def test_the_token_cap_closes_the_next_turn_too():
    svc = _budget_service(max_estimated_tokens_per_conversation=1)
    cid = svc.start_conversation().conversation_id
    svc.post_message(cid, "What time do your stores open on Saturday?")
    usage = costs.conversation_usage(svc.store, cid)
    assert usage["tokens"] > 1  # the first turn spent past the cap…
    result = svc.post_message(cid, "And the complaints process?")
    assert result.escalated is True  # …and completed; the NEXT turn closes
    assert result.escalation_reason == "budget_exhausted"
    budget = [e for e in svc.store.list_audit(cid) if e.kind == "budget"]
    assert len(budget) == 1
    assert budget[0].detail["tokens"] == usage["tokens"]
    assert budget[0].detail["max_tokens"] == 1


# ---------------------------------------------------------------------------
# Latency discipline at the live boundary (stubbed transport)
# ---------------------------------------------------------------------------

def _live_llm() -> LiveLLM:
    store = InMemoryStore()
    seed_registry(store)
    llm = LiveLLM(LLMConfig(backend="openai", model="gpt-stub",
                             api_key="test"))
    llm.prompts = PromptRegistry(store)
    return llm


def _timeout() -> httpx.ReadTimeout:
    return httpx.ReadTimeout(
        "slow", request=httpx.Request("POST", "http://stub.invalid"))


def test_a_timed_out_call_is_retried_once():
    llm = _live_llm()
    attempts = {"n": 0}

    def flaky(system, user):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _timeout()
        return "Recovered answer."

    llm._chat = flaky
    reply = llm.draft_reply(DraftContext(customer_message="hello"))
    assert reply == "Recovered answer."
    records = llm.drain_calls()
    assert [r["outcome"] for r in records] == ["timeout", "ok"]
    assert [r["attempt"] for r in records] == [1, 2]
    assert records[0]["call_kind"] == "decision"
    assert records[0]["tokens_out"] == 0
    assert records[1]["tokens_basis"] == "estimated"  # no usage reported


def test_a_second_timeout_leaves_the_boundary():
    llm = _live_llm()

    def always_slow(system, user):
        raise _timeout()

    llm._chat = always_slow
    with pytest.raises(ModelTimeoutError):
        llm.draft_reply(DraftContext(customer_message="hello"))
    records = llm.drain_calls()
    assert len(records) == 2
    assert all(r["outcome"] == "timeout" for r in records)


def test_an_ordinary_failure_still_degrades_to_the_composer():
    llm = _live_llm()

    def down(system, user):
        raise RuntimeError("provider down")

    llm._chat = down
    reply = llm.draft_reply(DraftContext(
        customer_message="What time do your stores open on Saturday?",
        findings=["Stores open at 9am on Saturdays."]))
    assert "9am" in reply  # the offline composer answered the turn
    records = llm.drain_calls()
    assert [r["outcome"] for r in records] == ["error"]


def test_provider_usage_is_recorded_as_measured():
    llm = _live_llm()

    def with_usage(system, user):
        llm._last_usage = (120, 30)  # what the provider reported
        return "A short answer."

    llm._chat = with_usage
    llm.draft_reply(DraftContext(customer_message="hello"))
    (record,) = llm.drain_calls()
    assert record["tokens_basis"] == "measured"
    assert record["tokens_in"] == 120
    assert record["tokens_out"] == 30
    assert record["tokens"] == 150


class _TimingOutLLM:
    """Test double (tests only): a model boundary whose decision
    call — and its one governed retry — has timed out.
    draft_reply raises ModelTimeoutError exactly as LiveLLM does
    once its retry is spent; the offline backend never does
    this, and production code never sees this class."""

    backend = "simulated-timeout"

    def draft_reply(self, ctx):
        raise ModelTimeoutError("simulated double timeout (test double)")

    def intent_hint(self, text):
        return None


def test_a_double_timeout_closes_the_turn_to_a_person():
    svc = ConversationService(store=InMemoryStore(), llm=_TimingOutLLM())
    svc.seed()
    cid = svc.start_conversation().conversation_id
    result = svc.post_message(cid, "why is my bill so high")
    assert result.escalated is True
    assert result.escalation_reason == "model_timeout"
    conv = svc.store.get_conversation(cid)
    assert conv.status == "escalated"
    escs = [e for e in svc.store.list_escalations()
            if e.conversation_id == cid]
    assert len(escs) == 1 and escs[0].reason == "model_timeout"
    outcome = attribution.attribute_conversation(svc.store, cid)
    assert outcome.rule == "escalated_by_design:model_timeout"
    assert outcome.cause is None
    assert svc.cost_report()["timeout_escalations"] == 1


# ---------------------------------------------------------------------------
# The digest section and the API surface
# ---------------------------------------------------------------------------

def test_the_daily_digest_carries_the_cost_section(service):
    from cx_agent.quality import render_digest_markdown

    cid = service.start_conversation().conversation_id
    service.post_message(cid, "What time do your stores open on Saturday?")
    digest = service.run_daily_quality(with_spot_check=False)
    assert digest.costs["calls"] >= 1
    assert digest.costs["decision_calls"] >= 1
    assert digest.costs["tokens"] > 0
    markdown = render_digest_markdown(digest)
    assert "## Cost and latency" in markdown


def test_costs_report_endpoint():
    from fastapi.testclient import TestClient

    from cx_agent.api import app

    client = TestClient(app)
    client.post("/seed")
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "What time do your stores open on Saturday?"})
    body = client.get("/costs/report").json()
    assert body["overall"]["calls"] >= 1
    assert body["overall"]["tokens_estimated"] == body["overall"]["tokens"]
    assert body["caps"]["max_decision_calls_per_conversation"] == 40
    assert body["caps"]["model_call_timeout_seconds"] == 60
    assert "note" in body
