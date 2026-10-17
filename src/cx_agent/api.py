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


class AnnotationIn(BaseModel):
    conversation_id: str
    annotator: str
    cause_correct: bool
    corrected_cause: str | None = None
    note: str = ""
    is_verdict: bool = False


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


@app.get("/roaming/notices", dependencies=[Depends(require_key)])
def list_roaming_notices(account_number: str | None = None) -> dict:
    return {"notices": [n.model_dump(mode="json") for n in
                        service.list_roaming_notices(account_number)]}


@app.get("/payments/attempts", dependencies=[Depends(require_key)])
def list_payment_attempts(account_number: str | None = None) -> dict:
    return {"attempts": [t.model_dump(mode="json") for t in
                         service.list_payment_attempts(account_number)]}


@app.get("/payments/arrangements", dependencies=[Depends(require_key)])
def list_payment_arrangements(account_number: str | None = None) -> dict:
    return {"arrangements": [a.model_dump(mode="json") for a in
                             service.list_payment_arrangements(account_number)]}


@app.get("/payments/notices", dependencies=[Depends(require_key)])
def list_payment_notices(account_number: str | None = None) -> dict:
    return {"notices": [n.model_dump(mode="json") for n in
                        service.list_payment_notices(account_number)]}


@app.get("/complaints", dependencies=[Depends(require_key)])
def list_complaints(conversation_id: str | None = None,
                    account_number: str | None = None) -> dict:
    from . import care_book
    rows = []
    for c in service.list_complaints(conversation_id, account_number):
        row = c.model_dump(mode="json")
        row["sla_state"] = care_book.complaint_sla_state(
            c, service.policy)
        rows.append(row)
    return {"complaints": rows}


@app.get("/complaints/{complaint_id}", dependencies=[Depends(require_key)])
def get_complaint(complaint_id: str) -> dict:
    from . import care_book
    record = service.get_complaint(complaint_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such complaint")
    row = record.model_dump(mode="json")
    row["sla_state"] = care_book.complaint_sla_state(record, service.policy)
    return row


@app.post("/complaints/watch", dependencies=[Depends(require_key)])
def run_complaint_watch() -> dict:
    """Run the SLA watch: breached complaints are raised to the
    human queue with their records attached."""
    return service.run_complaint_watch()


@app.get("/feedback", dependencies=[Depends(require_key)])
def list_feedback(conversation_id: str | None = None,
                  account_number: str | None = None) -> dict:
    return {"feedback": [f.model_dump(mode="json") for f in
                         service.list_feedback(conversation_id, account_number)]}


@app.get("/care/mining", dependencies=[Depends(require_key)])
def mining_report() -> dict:
    """The unhandled-intent mining report: candidate clusters
    for humans designing new intents. Creates nothing."""
    return service.mining_report()


@app.post("/attribution/run", dependencies=[Depends(require_key)])
def run_attribution() -> dict:
    """Run the attribution pass: every closed conversation gets
    its outcome + cause record, computed from its audit trail."""
    return service.run_attribution()


@app.get("/attribution/report", dependencies=[Depends(require_key)])
def attribution_report(days: int = 7) -> dict:
    """Attribution counts by cause × desk over a window, the
    previous window beside them, and the mining join."""
    if days < 1:
        raise HTTPException(status_code=400, detail="days must be >= 1")
    return service.attribution_report(days)


@app.get("/attribution", dependencies=[Depends(require_key)])
def list_attributions() -> dict:
    return {"attributions": [r.model_dump(mode="json")
                             for r in service.list_attributions()]}


@app.get("/attribution/{conversation_id}", dependencies=[Depends(require_key)])
def get_attribution(conversation_id: str) -> dict:
    record = service.store.get_attribution(conversation_id)
    if record is None:
        raise HTTPException(status_code=404,
                            detail="no attribution for this conversation")
    return record.model_dump(mode="json")


@app.post("/attribution/{conversation_id}", dependencies=[Depends(require_key)])
def attribute_conversation(conversation_id: str) -> dict:
    """Attribute one conversation — including one that merely
    stopped, which this act declares ended."""
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return service.attribute_conversation(
        conversation_id).model_dump(mode="json")


@app.get("/annotations/queue", dependencies=[Depends(require_key)])
def annotation_queue() -> dict:
    """The review queue: every escalated or unresolved
    conversation, a deterministic sample of the resolved ones,
    each with its review state."""
    return {"queue": service.annotation_queue()}


@app.get("/annotations/agreement", dependencies=[Depends(require_key)])
def annotation_agreement() -> dict:
    """Annotator agreement — simple rate and Cohen's kappa over
    cause labels — plus the disagreements awaiting a verdict."""
    return service.annotation_agreement()


@app.get("/annotations", dependencies=[Depends(require_key)])
def list_annotations(conversation_id: str | None = None) -> dict:
    return {"annotations": [a.model_dump(mode="json") for a in
                            service.list_annotations(conversation_id)]}


@app.post("/annotations", dependencies=[Depends(require_key)])
def annotate(body: AnnotationIn) -> dict:
    """Write one annotation. The queue's rules are enforced in
    code; a broken rule is a 400 naming it, a conversation that
    doesn't exist or isn't attributed yet is a 404."""
    if service.get_conversation(body.conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    if service.store.get_attribution(body.conversation_id) is None:
        raise HTTPException(
            status_code=404,
            detail="no attribution for this conversation — attribute "
                   "it before annotating it")
    try:
        record = service.annotate(
            body.conversation_id, body.annotator, body.cause_correct,
            body.corrected_cause, body.note, body.is_verdict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@app.get("/")
def console() -> FileResponse:
    return FileResponse(_CONSOLE, media_type="text/html")
