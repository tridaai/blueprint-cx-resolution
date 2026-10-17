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
    assert seeded.json()["seeded"]["customers"] == 10
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
