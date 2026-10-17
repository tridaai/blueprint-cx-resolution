"""FastAPI surface: health, readiness, the conversation API, the
escalation queue, and the single-file console at /.

The app is stateless — every record lives in the store — so it
scales horizontally behind any balancer. A request-ID middleware
tags every log line a request produces.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .logging_setup import configure_logging, conversation_id_var, new_request_id, request_id_var
from .service import ConversationService

log = logging.getLogger("cx_agent.api")

configure_logging()
service = ConversationService()

app = FastAPI(title="CX Resolution Agent — Trida AI Blueprint")

_CONSOLE = Path(__file__).parent / "static" / "console.html"


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["x-request-id"] = rid
    return response


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


class MessageIn(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/readiness")
def readiness() -> JSONResponse:
    info = service.readiness()
    ready = info["customers_seeded"] > 0
    return JSONResponse(
        {"ready": ready, **info},
        status_code=200 if ready else 503)


@app.post("/seed", dependencies=[Depends(require_key)])
def seed() -> dict:
    counts = service.seed()
    log.info("dataset seeded", extra={"fields": counts})
    return {"seeded": counts}


@app.post("/conversations", dependencies=[Depends(require_key)])
def start_conversation() -> dict:
    conv = service.start_conversation()
    return {"conversation_id": conv.conversation_id, "status": conv.status}


@app.get("/conversations", dependencies=[Depends(require_key)])
def list_conversations() -> dict:
    rows = []
    for conv in service.list_conversations():
        rows.append({
            "conversation_id": conv.conversation_id,
            "status": conv.status,
            "intent": conv.intent.value,
            "identity_state": conv.identity_state.value,
            "account_number": conv.account_number,
            "turns": conv.turns,
            "last_message": (conv.messages[-1].text[:120] if conv.messages else ""),
        })
    return {"conversations": rows}


@app.get("/conversations/{conversation_id}", dependencies=[Depends(require_key)])
def get_conversation(conversation_id: str) -> dict:
    conv = service.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return conv.model_dump(mode="json")


@app.post("/conversations/{conversation_id}/messages",
          dependencies=[Depends(require_key)])
def post_message(conversation_id: str, body: MessageIn) -> dict:
    token = conversation_id_var.set(conversation_id)
    try:
        result = service.post_message(conversation_id, body.text)
    finally:
        conversation_id_var.reset(token)
    return result.model_dump(mode="json")


@app.post("/conversations/{conversation_id}/export",
          dependencies=[Depends(require_key)])
def export_transcript(conversation_id: str) -> dict:
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    stored = service.export_transcript(conversation_id)
    return {"stored": stored,
            "note": "" if stored else "S3 not configured; nothing was stored"}


@app.get("/escalations", dependencies=[Depends(require_key)])
def list_escalations() -> dict:
    return {"escalations": [e.model_dump(mode="json")
                             for e in service.list_escalations()]}


@app.get("/actions", dependencies=[Depends(require_key)])
def list_actions(conversation_id: str | None = None) -> dict:
    return {"actions": [a.model_dump(mode="json")
                         for a in service.list_actions(conversation_id)]}


@app.get("/conversations/{conversation_id}/audit",
         dependencies=[Depends(require_key)])
def list_audit(conversation_id: str) -> dict:
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return {"audit": [e.model_dump(mode="json")
                      for e in service.list_audit(conversation_id)]}


@app.get("/disputes", dependencies=[Depends(require_key)])
def list_disputes(conversation_id: str | None = None,
                  account_number: str | None = None) -> dict:
    return {"disputes": [d.model_dump(mode="json") for d in
                         service.list_disputes(conversation_id, account_number)]}


@app.get("/disputes/{case_id}", dependencies=[Depends(require_key)])
def get_dispute(case_id: str) -> dict:
    case = service.get_dispute(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="no such dispute case")
    return case.model_dump(mode="json")


@app.get("/callbacks", dependencies=[Depends(require_key)])
def list_callbacks(conversation_id: str | None = None) -> dict:
    return {"callbacks": [c.model_dump(mode="json")
                           for c in service.list_callbacks(conversation_id)]}


@app.get("/callbacks/{callback_id}", dependencies=[Depends(require_key)])
def get_callback(callback_id: str) -> dict:
    callback = service.store.get_callback(callback_id)
    if callback is None:
        raise HTTPException(status_code=404, detail="no such callback")
    return callback.model_dump(mode="json")


@app.get("/procedure-runs", dependencies=[Depends(require_key)])
def list_procedure_runs(conversation_id: str | None = None) -> dict:
    return {"procedure_runs": [r.model_dump(mode="json") for r in
                               service.list_procedure_runs(conversation_id)]}


@app.get("/")
def console() -> FileResponse:
    return FileResponse(_CONSOLE, media_type="text/html")
