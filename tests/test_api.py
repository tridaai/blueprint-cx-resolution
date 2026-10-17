import os
import tempfile

# The API builds its service at import; point the store at a scratch
# file before that import happens.
os.environ["SQLITE_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="cx-api-"), "cx.db")
os.environ.pop("DATABASE_URL", None)

from fastapi.testclient import TestClient  # noqa: E402

from cx_agent.api import app  # noqa: E402


def test_health_and_readiness_lifecycle():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/readiness").status_code == 503  # nothing seeded yet
    seeded = client.post("/seed")
    assert seeded.status_code == 200
    assert seeded.json()["seeded"]["customers"] == 15
    ready = client.get("/readiness")
    assert ready.status_code == 200
    body = ready.json()
    assert body["store"] == "sqlite" and body["llm_backend"] == "offline"


def test_conversation_over_http():
    client = TestClient(app)
    client.post("/seed")
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    turn = client.post(f"/conversations/{cid}/messages",
                       json={"text": "What time do your stores open on Saturday?"})
    assert turn.status_code == 200
    body = turn.json()
    assert body["intent"] == "care"
    assert "9am" in body["reply"]
    listing = client.get("/conversations").json()["conversations"]
    assert any(c["conversation_id"] == cid for c in listing)
    detail = client.get(f"/conversations/{cid}").json()
    assert len(detail["messages"]) == 2
    assert client.get("/conversations/NOPE").status_code == 404


def test_console_is_served_at_root():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "CX Resolution Agent" in page.text
    assert "View all blueprints" in page.text


def test_escalation_queue_endpoint():
    client = TestClient(app)
    client.post("/seed")
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "let me talk to a real person"})
    client.post(f"/conversations/{cid}/messages",
                json={"text": "get me a human"})
    queue = client.get("/escalations").json()["escalations"]
    assert len(queue) == 1
    assert queue[0]["packet"]["queue"] == "general"


def test_callbacks_and_procedure_runs_endpoints():
    client = TestClient(app)
    client.post("/seed")
    assert client.get("/callbacks").json() == {"callbacks": []}
    assert client.get("/procedure-runs").json() == {"procedure_runs": []}
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    for text in ("I have no signal at all", "no, still nothing",
                 "nope, still no signal", "no, that didn't work either"):
        client.post(f"/conversations/{cid}/messages", json={"text": text})
    callbacks = client.get("/callbacks").json()["callbacks"]
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback["status"] == "proposed"
    assert callback["procedure_id"] == "PROC-NOSIGNAL"
    assert [s["step_id"] for s in callback["transcript"]] == [
        "NS-1", "NS-2", "NS-3", "NS-4", "book_callback"]
    detail = client.get(f"/callbacks/{callback['callback_id']}")
    assert detail.status_code == 200
    assert client.get("/callbacks/NOPE").status_code == 404
    runs = client.get(
        f"/procedure-runs?conversation_id={cid}").json()["procedure_runs"]
    assert len(runs) == 1
    assert runs[0]["state"] == "dead_ended"
    assert runs[0]["procedure_version"] == 1


def test_complaint_feedback_and_mining_endpoints():
    client = TestClient(app)
    client.post("/seed")
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages", json={
        "text": "I want to make a complaint — your engineer never "
                "showed up for Tuesday's appointment. My account "
                "is NB-1003."})
    complaints = client.get("/complaints").json()["complaints"]
    assert len(complaints) == 1
    assert complaints[0]["category"] == "appointment"
    assert complaints[0]["sla_state"] == "open"
    one = client.get(f"/complaints/{complaints[0]['complaint_id']}")
    assert one.status_code == 200
    assert one.json()["text"].startswith("I want to make a complaint")
    assert client.get("/complaints/CMP-NOPE").status_code == 404
    watch = client.post("/complaints/watch").json()
    assert watch["raised"] == []  # nothing is past its deadline yet

    conv2 = client.post("/conversations").json()
    client.post(f"/conversations/{conv2['conversation_id']}/messages",
                json={"text": "Some feedback: the engineer was "
                              "brilliant, honestly."})
    feedback = client.get("/feedback").json()["feedback"]
    assert len(feedback) == 1 and feedback[0]["kind"] == "compliment"

    conv3 = client.post("/conversations").json()
    client.post(f"/conversations/{conv3['conversation_id']}/messages",
                json={"text": "Can I buy a gift card for my daughter?"})
    mining = client.get("/care/mining").json()
    assert mining["unhandled_utterances"] >= 1
    assert mining["candidates"] == []  # one utterance is no cluster


def test_incident_feed_endpoints():
    client = TestClient(app)
    client.post("/seed")
    feed = client.get("/incidents").json()["incidents"]
    assert [i["incident_id"] for i in feed] == ["INC-7001", "INC-7002",
                                                "INC-7003"]
    assert feed[0]["status"] == "active"
    assert feed[0]["affected_postcodes"] == ["02139"]
    active = client.get("/incidents?status=active").json()["incidents"]
    assert [i["incident_id"] for i in active] == ["INC-7001"]
    one = client.get("/incidents/INC-7001")
    assert one.status_code == 200 and one.json()["acts"] == []
    assert client.get("/incidents/INC-NOPE").status_code == 404

    # The operator acts: add, update, resolve — each recorded.
    added = client.post("/incidents", json={
        "incident_id": "INC-9001", "title": "Signal down — West Loop",
        "service": "no_signal",
        "summary": "Signal is down around the West Loop.",
        "symptom_signals": ["no signal"], "affected_postcodes": ["60614"],
        "actor": "ops"})
    assert added.status_code == 200
    assert added.json()["status"] == "active"
    duplicate = client.post("/incidents", json={
        "incident_id": "INC-9001", "title": "again", "service": "no_signal",
        "summary": "x", "symptom_signals": ["no signal"]})
    assert duplicate.status_code == 400
    bad = client.post("/incidents", json={
        "title": "bad service", "service": "billing", "summary": "x",
        "symptom_signals": ["bill"]})
    assert bad.status_code == 400
    updated = client.post("/incidents/INC-9001/update", json={
        "summary": "Signal is down across the West Loop.",
        "actor": "ops"})
    assert updated.status_code == 200
    assert updated.json()["summary"] == \
        "Signal is down across the West Loop."
    assert client.post("/incidents/INC-NOPE/update",
                       json={"summary": "x"}).status_code == 404
    resolved = client.post("/incidents/INC-9001/resolve",
                           json={"actor": "ops"})
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    assert client.post("/incidents/INC-9001/resolve",
                       json={}).status_code == 400  # already resolved
    detail = client.get("/incidents/INC-9001").json()
    assert [a["kind"] for a in detail["acts"]] == ["add", "update",
                                                   "resolve"]

    # A conversation's matches ride their own view.
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "my broadband keeps dropping"})
    client.post(f"/conversations/{cid}/messages",
                json={"text": "NB-1007 postcode 02139"})
    from cx_agent.identity import expected_code

    client.post(f"/conversations/{cid}/messages",
                json={"text": expected_code("NB-1007")})
    matches = client.get(f"/conversations/{cid}/incidents").json()
    assert [m["incident_id"] for m in matches["matches"]] == ["INC-7001"]
    assert "broadband" in matches["matches"][0]["matched_signals"]
    assert matches["matches"][0]["current_status"] == "active"
    assert client.get("/conversations/NOPE/incidents").status_code == 404


def test_retention_endpoints():
    client = TestClient(app)
    client.post("/seed")
    before = client.get("/retention/report").json()
    conv = client.post("/conversations").json()
    cid = conv["conversation_id"]
    client.post(f"/conversations/{cid}/messages",
                json={"text": "Just cancel my account, no offers please"})
    client.post(f"/conversations/{cid}/messages",
                json={"text": "NB-1005 postcode 94107"})
    from cx_agent.identity import expected_code

    turn = client.post(
        f"/conversations/{cid}/messages",
        json={"text": expected_code("NB-1005")}).json()
    # No offers means none: the first proposal is the cancellation.
    assert turn["action"]["kind"] == "cancel_account"
    done = client.post(f"/conversations/{cid}/messages",
                       json={"text": "yes"}).json()
    assert done["action"]["status"] == "executed"
    cases = client.get("/retention/cases").json()["cases"]
    mine = [c for c in cases if c["conversation_id"] == cid]
    assert len(mine) == 1
    assert mine[0]["no_offers_requested"] is True
    assert mine[0]["outcome"] == "cancelled"
    detail = client.get(f"/retention/cases/{mine[0]['case_id']}")
    assert detail.status_code == 200
    assert client.get("/retention/cases/RET-NOPE").status_code == 404
    report = client.get("/retention/report").json()
    assert report["cases"] == before["cases"] + 1
    assert report["cancellations_completed"] == \
        before["cancellations_completed"] + 1
    assert report["offers_made"] == before["offers_made"]
