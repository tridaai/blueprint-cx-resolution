"""Round 12 — the daily quality pass.

The day's conversations are planted through the real service
(the trajectories are the earlier rounds' own: the lockout, the
over-cap dispute, the twice-asked human, the accepted signal
offers, the complaint, the stopped unhandled ask, the gift-card
asks the mining pass clusters), so every digest figure is
asserted exactly against what those trajectories are known to
leave behind. The complaint book's other SLA states are planted
as records with stated dates — the SLA reading itself is round
8's, tested there.

Drift is proven two ways: a positive control (the spot-check
over a baseline-green scenario on the untouched build finds
nothing) and a planted regression. The plant is a test double,
stated plainly: the test monkeypatches
ConversationService.post_message so the result it returns
reports escalated=False whatever the turn did. To the
spot-check — which grades the build through the same results
the eval gate grades — that is indistinguishable from a build
that stopped escalating, which is exactly the regression the
spot-check exists to catch.

Dates pin to 2026-10-01, the dataset's present day.
"""

import os
import re

os.environ.setdefault("CX_TODAY", "2026-10-01")

import pytest  # noqa: E402

from cx_agent import quality  # noqa: E402
from cx_agent.config import Policy  # noqa: E402
from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.models import ComplaintRecord, DailyDigest  # noqa: E402
from cx_agent.service import ConversationService  # noqa: E402
from cx_agent.store import InMemoryStore  # noqa: E402

POLICY = Policy()
DAY = "2026-10-01"

GIFT_ASKS = [
    "Can I buy a gift card for my daughter?",
    "Do you sell gift cards? I want one for my son.",
    "I would like a gift card for my nephew please.",
    "Where can I get a Northbeam gift card?",
]


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", DAY)


@pytest.fixture()
def svc():
    service = ConversationService(store=InMemoryStore())
    service.seed()
    return service


def _drive(service, turns):
    conv = service.start_conversation()
    for text in turns:
        text = re.sub(r"\{code:(NB-\d+)\}",
                      lambda m: expected_code(m.group(1)), text)
        service.post_message(conv.conversation_id, text)
    return conv.conversation_id


def plant_day(service):
    """One day of Northbeam traffic, through the service."""
    # A second ask for a human: escalated by design.
    _drive(service, ["I want to speak to a human being",
                     "No. Get me a human, now."])
    # Three wrong codes: the lockout. (Passes the two-failure
    # identity-offer threshold on the way — that offer lapses.)
    _drive(service, ["my phone bill seems off", "it is NB-1005 94107",
                     "111111", "222222", "333333"])
    # A dispute above the agent's cap: policy_limit.
    _drive(service, ["I want to dispute a charge on my bill",
                     "NB-1012 postcode 80202", "{code:NB-1012}",
                     "the international call charge of $58 is wrong, "
                     "I never made that call"])
    # Frustration rising, the early offer accepted.
    _drive(service, ["My bill has a charge I don't recognise.",
                     "Account NB-9999, postcode 00000. This is getting "
                     "annoying.",
                     "I already told you - NB-9999, 00000. This is "
                     "frustrating and ridiculous.",
                     "Yes please."])
    # Identity effort at the threshold, the early offer accepted.
    _drive(service, ["I need to update my card.",
                     "My account is NB-1002 and the postcode is 90210.",
                     "999999", "888888", "Yes please."])
    # A calm care question, answered and closed.
    conv = service.start_conversation()
    service.post_message(conv.conversation_id,
                         "What time do your stores open on Saturday?")
    service.post_message(conv.conversation_id,
                         "Thanks, that's clear now.")
    # A need the agent cannot serve; the customer simply stops,
    # and the operator attributes the stopped conversation.
    stopped = _drive(service, ["Can I borrow money from you?"])
    service.attribute_conversation(stopped)
    # The same unhandled ask four ways: a mining candidate.
    for ask in GIFT_ASKS:
        _drive(service, [ask])
    # A complaint: recorded on the book the moment it lands.
    conv = service.start_conversation()
    service.post_message(
        conv.conversation_id,
        "I want to make a complaint — your engineer never showed "
        "up for Tuesday's appointment and nobody told me. "
        "My account is NB-1003.")
    # The rest of the complaint book, planted as records with
    # stated dates: one already breached, one due tomorrow, one
    # answered (settled) — the SLA states at day end.
    store = service.store
    store.save_complaint(ComplaintRecord(
        complaint_id="CMP-OLD-BREACH", category="billing",
        text="planted: open past its deadline",
        created_date="2026-09-20", response_due="2026-09-25"))
    store.save_complaint(ComplaintRecord(
        complaint_id="CMP-DUE-SOON", category="service_quality",
        text="planted: due tomorrow",
        created_date="2026-09-27", response_due="2026-10-02"))
    store.save_complaint(ComplaintRecord(
        complaint_id="CMP-SETTLED", category="other",
        text="planted: already answered", status="answered",
        created_date="2026-09-28", response_due="2026-10-03"))
    return service


# -- the digest figures -------------------------------------------------


def test_daily_pass_composes_the_day_exactly(svc):
    plant_day(svc)
    digest = svc.run_daily_quality(with_spot_check=False)

    assert digest.date == DAY
    assert digest.version == 1
    assert digest.conversations == 12
    assert digest.by_outcome == {
        "resolved": 6, "resolved_with_action": 0,
        "escalated": 5, "unresolved_ended": 1}
    assert digest.resolution_share == 0.5
    assert digest.by_entry_desk == {
        "account": 1, "billing": 3, "care": 8}
    # The lockout is identity friction by its rule; the two
    # accepted-offer escalations left the customer unverified,
    # which the round 10 chain reads as verification abandoned —
    # the digest reports the records as they stand.
    assert digest.cause_mix == {
        "identity_friction": 3, "missing_capability": 1,
        "policy_limit": 1}
    assert digest.escalated_by_design == 1  # the twice-asked human
    assert digest.no_cause == 6  # the six resolutions
    assert digest.escalations_by_reason == {
        "customer_requested_human": 1,
        "dispute_above_agent_cap": 1,
        "early_human_offer_accepted": 2,
        "identity_verification_failed": 1}
    assert digest.signals == {
        "conversations": 12,
        "avg_effort": 2.75,
        "rising_frustration_conversations": 1,
        "rising_frustration_share": 0.083,
        "interventions": {
            "offered": 3, "accepted": 2, "declined": 0,
            "lapsed": 1, "second_forms": 0, "acceptance_rate": 1.0}}
    assert digest.complaints == {
        "open": 1, "due_soon": 1, "breached": 1, "settled": 1,
        "total": 4}
    assert len(digest.mining_candidates) == 1
    candidate = digest.mining_candidates[0]
    assert candidate["label"] == "gift card"
    assert candidate["count"] == 4
    assert len(candidate["samples"]) == 3
    assert set(candidate["samples"]) <= set(GIFT_ASKS)
    assert digest.spot_check["skipped"] is True


def test_daily_pass_for_a_day_with_nothing_on_record(svc):
    from datetime import date

    digest = svc.run_daily_quality(on=date(2026, 9, 30),
                                   with_spot_check=False)
    assert digest.conversations == 0
    assert digest.resolution_share is None
    assert digest.cause_mix == {}
    assert digest.signals["avg_effort"] is None
    assert digest.complaints["total"] == 0
    assert digest.mining_candidates == []


def test_rerunning_the_pass_replaces_the_digest_and_says_so(svc):
    plant_day(svc)
    first = svc.run_daily_quality(with_spot_check=False)
    second = svc.run_daily_quality(with_spot_check=False)
    assert first.version == 1
    assert second.version == 2
    assert second.conversations == first.conversations == 12
    stored = svc.list_digests()
    assert len(stored) == 1  # replaced, never accumulated
    assert stored[0].version == 2
    assert svc.get_digest(DAY).version == 2


# -- the spot-check -------------------------------------------------------


def test_sampling_rule_covers_every_rubric_and_is_stable():
    module = quality._eval_module()
    scenarios = quality.load_scenarios(module)
    picked = quality.sample_scenario_ids(scenarios, 4)
    assert picked == quality.sample_scenario_ids(scenarios, 4)
    by_id = {s["id"]: s for s in scenarios}
    assert all(pid in by_id for pid in picked)
    rubrics = {s.get("rubric") for s in scenarios}
    assert rubrics <= {by_id[pid].get("rubric") for pid in picked}
    assert 0 < len(picked) < len(scenarios)


def test_spot_check_positive_control_finds_no_drift():
    spot = quality.spot_check(POLICY, scenario_ids=["esc-threat"])
    assert spot["baseline_found"] is True
    assert spot["drift"] == []
    assert spot["drift_count"] == 0
    assert "No drift" in spot["summary"]


def test_spot_check_catches_the_planted_regression(monkeypatch):
    """The plant (see the module docstring): post_message is
    wrapped so its result always reports escalated=False. The
    threat scenario's zero-tolerance turn assertion was green
    in the committed baseline, so the spot-check must report it
    as drift — scenario, assertion, expected vs actual."""
    original = ConversationService.post_message

    def dulled(self, conversation_id, text):
        result = original(self, conversation_id, text)
        return result.model_copy(update={"escalated": False})

    monkeypatch.setattr(ConversationService, "post_message", dulled)
    spot = quality.spot_check(POLICY, scenario_ids=["esc-threat"])
    assert spot["drift_count"] == 1
    finding = spot["drift"][0]
    assert finding["scenario_id"] == "esc-threat"
    assert finding["where"] == "turn 1"
    assert finding["assertion"] == "escalated"
    assert finding["expected"] == "True"
    assert finding["actual"] == "False"
    assert "Drift:" in spot["summary"]


def test_spot_check_never_calls_unbaselined_failure_drift(
        monkeypatch, tmp_path):
    """The same planted regression, against a baseline fixture
    that does NOT record the scenario as green: the failure is
    reported as unbaselined, and drift stays zero — the
    comparison never claims history the baseline doesn't have."""
    import json

    baseline = {"generated_at": "fixture", "scenarios": [
        {"id": "esc-threat", "rubric": "escalation", "persona": None,
         "class": None, "passed": 1, "total": 2,
         "zt_passed": 1, "zt_total": 2}]}
    fixture = tmp_path / "results.json"
    fixture.write_text(json.dumps(baseline))

    original = ConversationService.post_message

    def dulled(self, conversation_id, text):
        result = original(self, conversation_id, text)
        return result.model_copy(update={"escalated": False})

    monkeypatch.setattr(ConversationService, "post_message", dulled)
    spot = quality.spot_check(POLICY, scenario_ids=["esc-threat"],
                              baseline_path=fixture)
    assert spot["drift"] == []
    assert spot["unbaselined_failing"] == ["esc-threat"]


def test_spot_check_without_a_baseline_claims_nothing(tmp_path):
    spot = quality.spot_check(
        POLICY, scenario_ids=["esc-threat"],
        baseline_path=tmp_path / "missing.json")
    assert spot["baseline_found"] is False
    assert spot["drift"] == []
    assert "no drift is claimed" in spot["summary"]


# -- the trend view and the rendering -------------------------------------


def _digest(date, **overrides):
    base = dict(
        date=date, version=1, conversations=10,
        by_outcome={"resolved": 5, "resolved_with_action": 1,
                    "escalated": 3, "unresolved_ended": 1},
        by_entry_desk={"care": 10}, resolution_share=0.6,
        cause_mix={"policy_limit": 2, "identity_friction": 2},
        escalated_by_design=1, no_cause=6,
        signals={"conversations": 10, "avg_effort": 3.5,
                 "rising_frustration_conversations": 1,
                 "rising_frustration_share": 0.1,
                 "interventions": {"offered": 0, "accepted": 0,
                                   "declined": 0, "lapsed": 0,
                                   "second_forms": 0,
                                   "acceptance_rate": None}},
        complaints={"open": 0, "due_soon": 0, "breached": 0,
                    "settled": 0, "total": 0},
        mining_candidates=[], escalations_by_reason={},
        spot_check={"drift": [], "drift_count": 0})
    base.update(overrides)
    return DailyDigest(**base)


def test_trend_table_math_over_three_planted_digests(svc):
    store = svc.store
    store.save_digest(_digest("2026-09-29", conversations=8,
                              resolution_share=0.5,
                              cause_mix={"routing_error": 3},
                              signals={"avg_effort": 4.0},
                              spot_check={"drift_count": 2}))
    store.save_digest(_digest("2026-09-30"))
    store.save_digest(_digest("2026-10-01", conversations=12,
                              resolution_share=0.25, cause_mix={},
                              signals={"avg_effort": None},
                              spot_check={"drift_count": 0}))
    rows = quality.quality_trend(store)
    assert [r["date"] for r in rows] == [
        "2026-09-29", "2026-09-30", "2026-10-01"]
    assert rows[0] == {
        "date": "2026-09-29", "version": 1, "conversations": 8,
        "resolution_share": 0.5, "top_cause": "routing_error",
        "avg_effort": 4.0, "drift_count": 2}
    # A cause tie breaks alphabetically, so the row never
    # depends on the order the causes were counted in.
    assert rows[1]["top_cause"] == "identity_friction"
    assert rows[1]["avg_effort"] == 3.5
    assert rows[2]["top_cause"] is None
    assert rows[2]["avg_effort"] is None
    assert rows[2]["resolution_share"] == 0.25


def test_markdown_renders_the_record_including_drift(svc):
    digest = _digest(
        DAY, version=2,
        spot_check={
            "rule": "sha256(scenario id) mod 4 == 0",
            "sampled": ["esc-threat"], "baseline_file": "evals/results.json",
            "baseline_found": True,
            "drift": [{"scenario_id": "esc-threat", "where": "turn 1",
                       "assertion": "escalated", "expected": "True",
                       "actual": "False"}],
            "drift_count": 1, "unbaselined_failing": [],
            "summary": "Drift: 1 assertion(s) ... fail now."})
    text = quality.render_digest_markdown(digest)
    assert text.startswith("# Daily quality digest — 2026-10-01")
    assert "Version 2" in text
    assert "DRIFT esc-threat (turn 1): assertion `escalated`" in text
    assert "expected True, got False" in text
    assert "triggers no automated action" in text


def test_markdown_of_a_quiet_day_says_no_drift_in_one_line(svc):
    plant_day(svc)
    digest = svc.run_daily_quality(with_spot_check=False)
    text = quality.render_digest_markdown(digest)
    assert "gift card" in text
    assert "Skipped for this run." in text


# -- the API and the CLI ---------------------------------------------------


def _stub_spot_check(monkeypatch):
    """The API/CLI tests exercise the wiring, not the
    spot-check (tested directly above): stub it to a fixed
    no-drift verdict so these tests stay fast."""
    monkeypatch.setattr(
        quality, "spot_check",
        lambda policy, **kw: {"rule": "stub", "sampled": [],
                              "baseline_file": "evals/results.json",
                              "baseline_found": True, "drift": [],
                              "drift_count": 0,
                              "unbaselined_failing": [],
                              "summary": "No drift (stubbed)."})


def test_quality_api_run_read_export_trend(monkeypatch):
    from fastapi.testclient import TestClient

    import cx_agent.api as api

    fresh = ConversationService(store=InMemoryStore())
    fresh.seed()
    plant_day(fresh)
    monkeypatch.setattr(api, "service", fresh)
    _stub_spot_check(monkeypatch)
    client = TestClient(api.app)

    created = client.post("/quality/run?date=2026-10-01")
    assert created.status_code == 200
    body = created.json()
    assert body["date"] == DAY and body["version"] == 1
    assert body["conversations"] == 12

    again = client.post("/quality/run?date=2026-10-01").json()
    assert again["version"] == 2
    listing = client.get("/quality/digests").json()["digests"]
    assert len(listing) == 1 and listing[0]["version"] == 2
    one = client.get(f"/quality/digests/{DAY}").json()
    assert one["cause_mix"]["policy_limit"] == 1
    export = client.get(f"/quality/digests/{DAY}/markdown")
    assert export.status_code == 200
    assert export.text.startswith("# Daily quality digest — 2026-10-01")
    trend = client.get("/quality/trend").json()["trend"]
    assert trend[0]["date"] == DAY and trend[0]["drift_count"] == 0
    assert client.get("/quality/digests/2026-09-01").status_code == 404
    assert client.post("/quality/run?date=nope").status_code == 400


def test_quality_cli_prints_the_digest_and_the_trend(
        monkeypatch, capsys):
    import cx_agent.cli as cli

    fresh = ConversationService(store=InMemoryStore())
    fresh.seed()
    plant_day(fresh)
    monkeypatch.setattr(cli, "ConversationService", lambda: fresh)
    _stub_spot_check(monkeypatch)

    assert cli.main(["daily-quality", "--date", DAY]) == 0
    out = capsys.readouterr().out
    assert "# Daily quality digest — 2026-10-01" in out
    assert "gift card" in out
    assert cli.main(["quality-trend"]) == 0
    out = capsys.readouterr().out
    assert DAY in out and "50%" in out
    assert cli.main(["daily-quality", "--date", "nope"]) == 2
