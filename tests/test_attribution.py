"""Round 10 — failure attribution + the annotation queue.

The attribution rules are code over the audit trail, so the
tests plant the trails: scripted conversations through the real
service for every cause a conversation can produce, direct record
plants for the rules a trajectory can't reach (an internal error
mid-turn, a policy-blocked action on a conversation that never
resolved), pinned-clock windows for the report, and a planted
annotator set whose agreement — and kappa — is known exactly.

Dates pin to 2026-10-01, the dataset's present day.
"""

import os
import re
import tempfile

os.environ.setdefault("CX_TODAY", "2026-10-01")

import pytest  # noqa: E402

from cx_agent import attribution  # noqa: E402
from cx_agent.config import Policy  # noqa: E402
from cx_agent.identity import expected_code  # noqa: E402
from cx_agent.models import (  # noqa: E402
    ActionKind,
    ActionRecord,
    AuditEvent,
    Conversation,
    Escalation,
    HandoffPacket,
    Message,
    ProposedAction,
)
from cx_agent.service import ConversationService  # noqa: E402
from cx_agent.store import InMemoryStore  # noqa: E402


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    monkeypatch.setenv("CX_TODAY", "2026-10-01")


@pytest.fixture()
def svc():
    service = ConversationService(store=InMemoryStore())
    service.seed()
    return service


def drive(service, turns):
    conv = service.start_conversation()
    for text in turns:
        text = re.sub(r"\{code:(NB-\d+)\}",
                      lambda m: expected_code(m.group(1)), text)
        service.post_message(conv.conversation_id, text)
    return service.store.get_conversation(conv.conversation_id)


def attribute(service, conv):
    return attribution.attribute_conversation(service.store,
                                              conv.conversation_id)


LOCKOUT = ["my phone bill seems off", "it is NB-1005 94107",
           "111111", "222222", "333333"]
OVER_CAP = ["I want to dispute a charge on my bill",
            "NB-1012 postcode 80202", "{code:NB-1012}",
            "the international call charge of $58 is wrong, I never made that call"]
UNHANDLED = ["Can I borrow money from you?"]
HUMAN_TWICE = ["I want to speak to a human being",
               "No. Get me a human, now."]
STUCK = ["my internet is down", "no signal", "dunno", "whatever",
         "just fix it please"]
DEAD_END_DECLINE = ["I have no signal at all", "no, still nothing",
                    "nope, still no signal", "no, that didn't work either",
                    "NB-1004 postcode 60614", "{code:NB-1004}",
                    "no, I don't want a callback"]
BOUNCE = ["why is my bill so high", "NB-1001 postcode 02134",
          "{code:NB-1001}",
          "hold on — actually the real problem is I have no signal at home",
          "forget the signal — I'm travelling to Spain next week, "
          "should I get a roaming pass?"]
PLAN_LEFT = ["I want to change my plan to Unlimited Max",
             "NB-1004 postcode 60614", "{code:NB-1004}"]
CREDIT = ["why is my bill so high",
          "My account is NB-1001 and the postcode is 02134",
          "{code:NB-1001}",
          "I was charged twice for device protection, that's wrong",
          "yes"]
EXPLAIN = ["why is my bill so high", "NB-1001 postcode 02134",
           "{code:NB-1001}"]
VERIFY_LEFT = ["update my card please", "NB-1005 postcode 94107"]


@pytest.mark.parametrize("turns,outcome,cause,rule", [
    (LOCKOUT, "escalated", "identity_friction", "identity_lockout"),
    (OVER_CAP, "escalated", "policy_limit", "dispute_above_cap"),
    (UNHANDLED, "unresolved_ended", "missing_capability",
     "unhandled_intent_logged"),
    (HUMAN_TWICE, "escalated", None, "escalated_by_design:customer_requested_human"),
    (STUCK, "escalated", "procedure_dead_end", "procedure_stalled_no_progress"),
    (DEAD_END_DECLINE, "unresolved_ended", "procedure_dead_end",
     "procedure_exhausted"),
    (BOUNCE, "unresolved_ended", "routing_error", "rerouted_without_progress"),
    (PLAN_LEFT, "unresolved_ended", "customer_left", "left_with_action_pending"),
    (VERIFY_LEFT, "unresolved_ended", "identity_friction",
     "verification_abandoned"),
    (CREDIT, "resolved_with_action", None, "resolved_with_action"),
    (EXPLAIN, "resolved", None, "resolved"),
])
def test_trajectory_attribution(svc, turns, outcome, cause, rule):
    conv = drive(svc, turns)
    record = attribute(svc, conv)
    assert record.outcome == outcome
    assert record.cause == cause
    assert record.rule == rule
    assert record.attributed_date == "2026-10-01"
    # The evidence is on the record — the attribution explains
    # itself without a second look at the trail.
    assert "reroutes" in record.evidence
    assert "escalation_reason" in record.evidence


def test_evidence_carries_the_facts_the_rule_read(svc):
    conv = drive(svc, BOUNCE)
    record = attribute(svc, conv)
    assert record.evidence["reroutes"] == 2
    assert record.evidence["executed_kinds"] == []
    conv = drive(svc, CREDIT)
    record = attribute(svc, conv)
    assert record.evidence["executed_kinds"] == ["apply_credit"]


# -- rules a trajectory can't produce ----------------------------------------


def _plant_conv(store, cid, **kw):
    conv = Conversation(conversation_id=cid, **kw)
    store.save_conversation(conv)
    return conv


def test_system_error_comes_from_the_error_event(svc):
    conv = _plant_conv(svc.store, "CONV-ERR", turns=2)
    svc.store.save_audit(AuditEvent(
        event_id="EVT-ERR", conversation_id="CONV-ERR", kind="system",
        detail={"event": "internal_error", "error": "RuntimeError: boom"},
        turn=2))
    record = attribute(svc, conv)
    assert (record.outcome, record.cause, record.rule) == (
        "unresolved_ended", "system_error", "internal_error_event")


def test_a_crashed_turn_writes_the_error_event(svc, monkeypatch):
    conv = svc.start_conversation()

    def boom(conv_, text):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(svc.supervisor, "run_turn", boom)
    with pytest.raises(RuntimeError):
        svc.post_message(conv.conversation_id, "hello?")
    events = [e for e in svc.store.list_audit(conv.conversation_id)
              if e.kind == "system"]
    assert len(events) == 1
    assert events[0].detail["event"] == "internal_error"
    record = attribute(svc, svc.store.get_conversation(conv.conversation_id))
    assert record.cause == "system_error"


def test_policy_block_on_an_action_record(svc):
    _plant_conv(svc.store, "CONV-BLK", turns=3)
    svc.store.save_action(ActionRecord(
        action_id="ACT-BLK", conversation_id="CONV-BLK",
        kind=ActionKind.REFUND_PAYMENT, summary="refund to a named card",
        status="blocked", block_reason="refund_destination_not_on_file",
        turn=3))
    record = attribute(svc, svc.store.get_conversation("CONV-BLK"))
    assert (record.cause, record.rule) == (
        "policy_limit", "action_blocked:refund_destination_not_on_file")


def test_goodwill_refusal_is_policy_when_unresolved(svc):
    _plant_conv(svc.store, "CONV-GW", turns=4)
    svc.store.save_audit(AuditEvent(
        event_id="EVT-GW", conversation_id="CONV-GW", kind="billing",
        detail={"event": "goodwill_refused",
                "reason": "rolling goodwill cap reached"}, turn=4))
    record = attribute(svc, svc.store.get_conversation("CONV-GW"))
    assert (record.cause, record.rule) == ("policy_limit", "goodwill_refused")


def test_bare_no_progress_escalation_is_missing_capability(svc):
    conv = _plant_conv(svc.store, "CONV-NP", status="escalated", turns=4)
    svc.store.save_escalation(Escalation(
        escalation_id="ESC-NP", conversation_id="CONV-NP",
        reason="no_progress", queue="general",
        packet=HandoffPacket(conversation_id="CONV-NP", queue="general",
                             reason="no_progress", identity_state="unverified",
                             summary="planted"), turn=4))
    record = attribute(svc, conv)
    assert (record.cause, record.rule) == (
        "missing_capability", "no_progress_escalation")


def test_a_resolved_conversation_never_gets_a_cause(svc):
    # Even with a policy block on its trail, a conversation that
    # resolved is a resolution — causes are for the other endings.
    conv = _plant_conv(svc.store, "CONV-OK", status="resolved", turns=3)
    svc.store.save_action(ActionRecord(
        action_id="ACT-OK", conversation_id="CONV-OK",
        kind=ActionKind.APPLY_CREDIT, summary="credit",
        status="blocked", block_reason="credit_above_cap", turn=2))
    record = attribute(svc, conv)
    assert record.outcome == "resolved" and record.cause is None


# -- the pass -----------------------------------------------------------------


def test_the_pass_attributes_closed_conversations_only(svc):
    drive(svc, LOCKOUT)          # escalated — closed
    drive(svc, EXPLAIN)          # resolved — closed
    drive(svc, UNHANDLED)        # still active — open
    result = attribution.run_attribution(svc.store)
    assert len(result["attributed"]) == 2
    assert result["still_open"] == 1
    again = attribution.run_attribution(svc.store)
    assert again["attributed"] == []
    assert again["already_attributed"] == 2


def test_a_record_stands_once_written(svc):
    conv = drive(svc, UNHANDLED)
    first = attribute(svc, conv)
    second = attribute(svc, conv)
    assert first is second or first.model_dump() == second.model_dump()
    assert len(svc.store.list_attributions()) == 1


# -- the report: windows and reconciliation ------------------------------------


def _escalated_plant(store, cid, reason, desk="billing"):
    _plant_conv(store, cid, status="escalated", turns=3,
                specialist_state={"desk": desk})
    store.save_escalation(Escalation(
        escalation_id=f"ESC-{cid}", conversation_id=cid, reason=reason,
        queue="general",
        packet=HandoffPacket(conversation_id=cid, queue="general",
                             reason=reason, identity_state="verified",
                             summary="planted"), turn=3))


def test_report_windows_and_reconciliation(svc, monkeypatch):
    # Previous window (pin 2026-10-01; report window 3 days):
    # two policy caps at billing, one designed escalation.
    for cid in ("CONV-P1", "CONV-P2"):
        _escalated_plant(svc.store, cid, "dispute_above_agent_cap")
        attribution.attribute_conversation(svc.store, cid)
    _escalated_plant(svc.store, "CONV-P3", "customer_requested_human",
                     desk="care")
    attribution.attribute_conversation(svc.store, "CONV-P3")

    monkeypatch.setenv("CX_TODAY", "2026-10-04")
    # Current window: one more cap, one lockout at account.
    _escalated_plant(svc.store, "CONV-C1", "dispute_above_agent_cap")
    attribution.attribute_conversation(svc.store, "CONV-C1")
    _escalated_plant(svc.store, "CONV-C2", "identity_verification_failed",
                     desk="account")
    attribution.attribute_conversation(svc.store, "CONV-C2")

    report = attribution.attribution_report(svc.store, svc.policy, days=3)
    assert report["current_window"] == ["2026-10-02", "2026-10-04"]
    assert report["previous_window"] == ["2026-09-29", "2026-10-01"]
    cur, prev = report["current"], report["previous"]
    assert cur["total"] == 2 and prev["total"] == 3
    assert cur["by_cause"]["policy_limit"] == 1
    assert cur["by_cause"]["identity_friction"] == 1
    assert prev["by_cause"]["policy_limit"] == 2
    assert prev["by_design"] == 1
    rows = {(r["cause"], r["desk"]): r for r in report["cause_desk_rows"]}
    assert rows[("policy_limit", "billing")]["current"] == 1
    assert rows[("policy_limit", "billing")]["previous"] == 2
    assert rows[("policy_limit", "billing")]["delta"] == -1
    assert rows[("identity_friction", "account")]["current"] == 1
    # Reconciliation: every record is counted exactly once.
    for bucket in (cur, prev):
        assert bucket["total"] == sum(bucket["by_outcome"].values())
        non_resolved = bucket["total"] \
            - bucket["by_outcome"]["resolved"] \
            - bucket["by_outcome"]["resolved_with_action"]
        assert non_resolved == sum(bucket["by_cause"].values()) \
            + bucket["by_design"] + bucket["no_cause"]


def test_report_joins_mining_candidates_to_causes(svc):
    texts = ["can I get a gift card for my trouble",
             "do you sell gift cards here",
             "I want to buy a gift card please",
             "gift card — can I purchase one today"]
    for i, text in enumerate(texts):
        cid = f"CONV-GIFT-{i}"
        _plant_conv(svc.store, cid, turns=2, unhandled_intent=text,
                    messages=[Message(role="customer", text=text, turn=1)])
        attribution.attribute_conversation(svc.store, cid)
    report = attribution.attribution_report(svc.store, svc.policy, days=7)
    join = [j for j in report["mining_evidence"] if "gift" in j["label"]]
    assert len(join) == 1
    assert join[0]["count"] == 4
    assert join[0]["missing_capability"] == 4
    assert join[0]["routing_error"] == 0
    assert join[0]["unattributed"] == 0


# -- the annotation queue --------------------------------------------------------


def _attributed(svc, cid, reason, desk="billing"):
    _escalated_plant(svc.store, cid, reason, desk)
    return attribution.attribute_conversation(svc.store, cid)


def test_annotation_flow_and_the_rules(svc):
    record = _attributed(svc, "CONV-A1", "dispute_above_agent_cap")
    assert record.cause == "policy_limit"
    # Annotating before attribution is refused.
    _plant_conv(svc.store, "CONV-A0", status="escalated")
    with pytest.raises(ValueError, match="no attribution record"):
        attribution.annotate(svc.store, "CONV-A0", "amy", True)
    # A wrong mark without the correction is refused, as is a
    # correction outside the taxonomy, and a correction riding
    # on a "right" mark.
    with pytest.raises(ValueError, match="requires the corrected cause"):
        attribution.annotate(svc.store, "CONV-A1", "amy", False)
    with pytest.raises(ValueError, match="not in the taxonomy"):
        attribution.annotate(svc.store, "CONV-A1", "amy", False,
                             corrected_cause="bad_luck")
    with pytest.raises(ValueError, match="only for an annotation"):
        attribution.annotate(svc.store, "CONV-A1", "amy", True,
                             corrected_cause="routing_error")
    ann = attribution.annotate(svc.store, "CONV-A1", "amy", True,
                               note="cap fired, record agrees")
    assert ann.cause_correct and not ann.is_verdict
    # The same person can't be the second opinion.
    with pytest.raises(ValueError, match="different annotator"):
        attribution.annotate(svc.store, "CONV-A1", "amy", True)
    attribution.annotate(svc.store, "CONV-A1", "ben", True)
    agreement = attribution.annotation_agreement(svc.store)
    assert agreement["pairs"] == 1 and agreement["agreements"] == 1
    assert agreement["agreement_rate"] == 1.0 and agreement["kappa"] == 1.0
    assert agreement["disagreements"] == []
    queue = {r["conversation_id"]: r
             for r in attribution.annotation_queue(svc.store, svc.policy)}
    assert queue["CONV-A1"]["state"] == "annotated"
    assert queue["CONV-A1"]["annotations"] == 2


def test_disagreement_then_verdict_settles(svc):
    _attributed(svc, "CONV-D1", "dispute_above_agent_cap")
    attribution.annotate(svc.store, "CONV-D1", "amy", True)
    attribution.annotate(svc.store, "CONV-D1", "ben", False,
                         corrected_cause="routing_error")
    agreement = attribution.annotation_agreement(svc.store)
    assert agreement["agreements"] == 0
    assert len(agreement["disagreements"]) == 1
    item = agreement["disagreements"][0]
    assert item["attributed_cause"] == "policy_limit"
    assert item["second"]["cause"] == "routing_error"
    # A verdict from one of the pair is refused; from a third
    # person it settles, and the record closes.
    with pytest.raises(ValueError, match="third person"):
        attribution.annotate(svc.store, "CONV-D1", "amy", False,
                             corrected_cause="routing_error",
                             is_verdict=True)
    verdict = attribution.annotate(
        svc.store, "CONV-D1", "lead", False,
        corrected_cause="routing_error", note="desk bounced it first",
        is_verdict=True)
    assert verdict.is_verdict
    agreement = attribution.annotation_agreement(svc.store)
    assert agreement["disagreements"] == []
    assert agreement["settled"] == ["CONV-D1"]
    queue = {r["conversation_id"]: r
             for r in attribution.annotation_queue(svc.store, svc.policy)}
    assert queue["CONV-D1"]["state"] == "settled"
    with pytest.raises(ValueError, match="settled"):
        attribution.annotate(svc.store, "CONV-D1", "pat", True)


def test_verdict_needs_a_live_disagreement(svc):
    _attributed(svc, "CONV-V1", "dispute_above_agent_cap")
    attribution.annotate(svc.store, "CONV-V1", "amy", True)
    with pytest.raises(ValueError, match="needs two annotations"):
        attribution.annotate(svc.store, "CONV-V1", "lead", True,
                             is_verdict=True)
    attribution.annotate(svc.store, "CONV-V1", "ben", True)
    with pytest.raises(ValueError, match="agree"):
        attribution.annotate(svc.store, "CONV-V1", "lead", True,
                             is_verdict=True)


def test_kappa_on_a_planted_set_with_known_agreement(svc):
    # Four pairs, effective causes known by construction:
    #   (policy, policy) agree · (routing, routing) agree
    #   (missing, policy) disagree · (missing, missing) agree
    # Po = 3/4; marginals A {policy 1, routing 1, missing 2},
    # B {policy 2, routing 1, missing 1} → Pe = 5/16,
    # kappa = (0.75 − 0.3125) / (1 − 0.3125) = 0.6364.
    def plant(cid, reason, amy, ben):
        _attributed(svc, cid, reason)
        attribution.annotate(svc.store, cid, "amy", *amy)
        attribution.annotate(svc.store, cid, "ben", *ben)

    plant("CONV-K1", "dispute_above_agent_cap", (True,), (True,))
    conv = _plant_conv(svc.store, "CONV-K2", turns=2)
    svc.store.save_audit(AuditEvent(event_id="EVT-K2A", conversation_id="CONV-K2", kind="reroute",
                                    detail={"from_desk": "billing",
                                            "to_desk": "technical"}, turn=1))
    svc.store.save_audit(AuditEvent(event_id="EVT-K2A", conversation_id="CONV-K2", kind="reroute",
                                    detail={"from_desk": "technical",
                                            "to_desk": "roaming"}, turn=2))
    attribution.attribute_conversation(svc.store, "CONV-K2")
    attribution.annotate(svc.store, "CONV-K2", "amy", True)
    attribution.annotate(svc.store, "CONV-K2", "ben", True)
    _plant_conv(svc.store, "CONV-K3", turns=1,
                unhandled_intent="Can I borrow money from you?")
    attribution.attribute_conversation(svc.store, "CONV-K3")
    attribution.annotate(svc.store, "CONV-K3", "amy", True)
    attribution.annotate(svc.store, "CONV-K3", "ben", False,
                         corrected_cause="policy_limit")
    _plant_conv(svc.store, "CONV-K4", turns=1,
                pending_action=ProposedAction(
                    kind=ActionKind.CHANGE_PLAN, summary="plan change"))
    attribution.attribute_conversation(svc.store, "CONV-K4")
    attribution.annotate(svc.store, "CONV-K4", "amy", False,
                         corrected_cause="missing_capability")
    attribution.annotate(svc.store, "CONV-K4", "ben", False,
                         corrected_cause="missing_capability")

    agreement = attribution.annotation_agreement(svc.store)
    assert agreement["pairs"] == 4
    assert agreement["agreements"] == 3
    assert agreement["agreement_rate"] == 0.75
    assert agreement["kappa"] == 0.6364
    assert [d["conversation_id"] for d in agreement["disagreements"]] == ["CONV-K3"]


def test_queue_holds_failures_in_full_and_resolved_by_sample(svc):
    _attributed(svc, "CONV-Q1", "dispute_above_agent_cap")
    queue = attribution.annotation_queue(svc.store, svc.policy)
    assert [r["conversation_id"] for r in queue] == ["CONV-Q1"]
    assert queue[0]["sampled"] is False
    # With the modulus at 1 every resolved conversation queues.
    conv = drive(svc, EXPLAIN)
    attribution.attribute_conversation(svc.store, conv.conversation_id)
    queue = attribution.annotation_queue(
        svc.store, Policy(annotation_sample_modulus=1))
    row = {r["conversation_id"]: r for r in queue}[conv.conversation_id]
    assert row["sampled"] is True and row["outcome"] == "resolved"
    # The sampling is deterministic: same id, same decision.
    first = attribution._sampled(conv.conversation_id, 5)
    assert first == attribution._sampled(conv.conversation_id, 5)


# -- the API surface -------------------------------------------------------------


def test_attribution_and_annotations_over_http():
    os.environ["SQLITE_PATH"] = os.path.join(
        tempfile.mkdtemp(prefix="cx-api-attr-"), "cx.db")
    os.environ.pop("DATABASE_URL", None)
    from fastapi.testclient import TestClient

    from cx_agent.api import app
    client = TestClient(app)
    client.post("/seed")
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    for text in ["my phone bill seems off", "it is NB-1005 94107",
                 "111111", "222222", "333333"]:
        client.post(f"/conversations/{cid}/messages", json={"text": text})
    run = client.post("/attribution/run").json()
    assert cid in run["attributed"]
    record = client.get(f"/attribution/{cid}").json()
    assert record["cause"] == "identity_friction"
    assert record["rule"] == "identity_lockout"
    report = client.get("/attribution/report").json()
    assert report["current"]["by_cause"]["identity_friction"] >= 1
    queue = client.get("/annotations/queue").json()["queue"]
    assert any(r["conversation_id"] == cid for r in queue)
    # Annotating a conversation that isn't attributed: 404.
    other = client.post("/conversations").json()["conversation_id"]
    missing = client.post("/annotations", json={
        "conversation_id": other, "annotator": "amy", "cause_correct": True})
    assert missing.status_code == 404
    ok = client.post("/annotations", json={
        "conversation_id": cid, "annotator": "amy", "cause_correct": True})
    assert ok.status_code == 200
    dup = client.post("/annotations", json={
        "conversation_id": cid, "annotator": "amy", "cause_correct": True})
    assert dup.status_code == 400
    assert "different annotator" in dup.json()["detail"]
    client.post("/annotations", json={
        "conversation_id": cid, "annotator": "ben", "cause_correct": True})
    agreement = client.get("/annotations/agreement").json()
    assert agreement["pairs"] == 1 and agreement["kappa"] == 1.0
