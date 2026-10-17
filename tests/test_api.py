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
    assert seeded.json()["seeded"]["customers"] == 14
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
