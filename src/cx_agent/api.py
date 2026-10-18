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


class HandoffAcceptIn(BaseModel):
    receiver: str


class AnnotationIn(BaseModel):
    conversation_id: str
    annotator: str
    cause_correct: bool
    corrected_cause: str | None = None
    note: str = ""
    is_verdict: bool = False


class PromptVersionIn(BaseModel):
    text: str
    actor: str = ""
    note: str = ""


class PromptRollbackIn(BaseModel):
    version: int
    actor: str = ""
    note: str = ""


class PromptUnpinIn(BaseModel):
    actor: str = ""
    note: str = ""


class IncidentIn(BaseModel):
    incident_id: str = ""
    title: str
    service: str
    summary: str
    symptom_signals: list[str]
    affected_postcodes: list[str] = []
    status: str = "active"
    started_at: str = ""
    actor: str = ""
    note: str = ""


class IncidentUpdateIn(BaseModel):
    title: str | None = None
    service: str | None = None
    summary: str | None = None
    symptom_signals: list[str] | None = None
    affected_postcodes: list[str] | None = None
    status: str | None = None
    started_at: str | None = None
    actor: str = ""
    note: str = ""


class IncidentResolveIn(BaseModel):
    actor: str = ""
    note: str = ""


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
                             for e in service.list_escalations()],
            # The queue's fallen-on-the-floor view (round 17):
            # handoffs nobody has picked up past the governed
            # threshold, oldest first.
            "unaccepted_overdue": service.unaccepted_handoffs()}


@app.post("/escalations/{escalation_id}/accept",
          dependencies=[Depends(require_key)])
def accept_handoff(escalation_id: str, body: HandoffAcceptIn) -> dict:
    """A person picks a handoff up — a recorded act: the
    receiver and the day land on the escalation, and the act
    in the conversation's audit trail."""
    try:
        esc = service.accept_handoff(escalation_id, body.receiver)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such escalation")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"escalation": esc.model_dump(mode="json")}


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


@app.get("/retention/report", dependencies=[Depends(require_key)])
def retention_report() -> dict:
    """The retention report: cases by reason, offers made and
    accepted, the save rate over decided cases, cancellations
    completed — counts from the case records only."""
    return service.retention_report()


@app.get("/retention/cases", dependencies=[Depends(require_key)])
def list_retention_cases(conversation_id: str | None = None,
                         account_number: str | None = None) -> dict:
    return {"cases": [c.model_dump(mode="json") for c in
                      service.list_retention_cases(conversation_id,
                                                   account_number)]}


@app.get("/retention/cases/{case_id}", dependencies=[Depends(require_key)])
def get_retention_case(case_id: str) -> dict:
    case = service.get_retention_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="no such retention case")
    return case.model_dump(mode="json")


@app.get("/callbacks", dependencies=[Depends(require_key)])
def list_callbacks(conversation_id: str | None = None) -> dict:
    return {"callbacks": [c.model_dump(mode="json")
                           for c in service.list_callbacks(conversation_id)]}


@app.get("/promises", dependencies=[Depends(require_key)])
def list_promises(account_number: str | None = None) -> dict:
    """The promises ledger (round 16): every commitment on the
    book — complaint response deadlines, booked callbacks,
    arrangement instalments — with its derived due state. Broken
    first. Derived from the records on each read; nothing here
    acts on a broken promise."""
    entries = service.promise_ledger(account_number)
    return {"promises": [e.model_dump(mode="json") for e in entries],
            "broken_count": sum(1 for e in entries
                                if e.state == "broken")}


@app.get("/conversations/{conversation_id}/continuity",
         dependencies=[Depends(require_key)])
def conversation_continuity(conversation_id: str) -> dict:
    """The continuity briefing for a conversation: the snapshot
    delivered when identity verified, or the briefing computed
    live from the account's records."""
    result = service.conversation_continuity(conversation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return result


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


@app.get("/signals/report", dependencies=[Depends(require_key)])
def signal_report() -> dict:
    """Signal aggregates over stored conversations: average
    effort by entry desk, the rising-frustration share,
    intervention outcomes, and the descriptive
    frustration/escalation counts."""
    return service.signal_report()


@app.get("/costs/report", dependencies=[Depends(require_key)])
def cost_report() -> dict:
    """Cost & latency over stored conversations: model calls by
    kind, tokens split by basis (measured vs estimated), mean
    and p95 latency — overall, per desk, per day — folded from
    the call records on the audit trails, with the budget and
    timeout closings and the governed caps named."""
    return service.cost_report()


@app.post("/quality/run", dependencies=[Depends(require_key)])
def run_daily_quality(date: str | None = None) -> dict:
    """Run the daily quality pass for a date (default: today, by
    the shared clock) and return the stored digest. Re-running
    for a date replaces the digest and raises its version."""
    from datetime import date as _date

    try:
        on = _date.fromisoformat(date) if date else None
    except ValueError as exc:
        raise HTTPException(status_code=400,
                            detail=f"not an ISO date: {date}") from exc
    return service.run_daily_quality(on).model_dump(mode="json")


@app.get("/quality/digests", dependencies=[Depends(require_key)])
def list_digests() -> dict:
    return {"digests": [d.model_dump(mode="json")
                        for d in service.list_digests()]}


@app.get("/quality/trend", dependencies=[Depends(require_key)])
def quality_trend() -> dict:
    """Headline metrics across the stored digests, oldest first."""
    return {"trend": service.quality_trend()}


@app.get("/quality/digests/{digest_date}",
         dependencies=[Depends(require_key)])
def get_digest(digest_date: str) -> dict:
    digest = service.get_digest(digest_date)
    if digest is None:
        raise HTTPException(status_code=404,
                            detail="no digest for this date — "
                                   "run the daily pass for it first")
    return digest.model_dump(mode="json")


@app.get("/quality/digests/{digest_date}/markdown",
         dependencies=[Depends(require_key)])
def get_digest_markdown(digest_date: str):
    """The digest as exportable markdown — rendered from the
    stored record by the one rendering function."""
    from fastapi.responses import PlainTextResponse

    from . import quality

    digest = service.get_digest(digest_date)
    if digest is None:
        raise HTTPException(status_code=404,
                            detail="no digest for this date — "
                                   "run the daily pass for it first")
    return PlainTextResponse(quality.render_digest_markdown(digest),
                             media_type="text/markdown")


@app.get("/prompts", dependencies=[Depends(require_key)])
def list_prompts() -> dict:
    """The prompt registry's active set: every governed text at
    its active version, with its pin if one stands."""
    return {"prompts": service.list_prompts()}


@app.get("/prompts/{prompt_id}", dependencies=[Depends(require_key)])
def get_prompt(prompt_id: str) -> dict:
    """One registry entry in full: all its versions (old versions
    stay readable forever), its pin, and its act log."""
    detail = service.get_prompt(prompt_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="no such prompt")
    return detail


def _prompt_act_or_http(fn, *args):
    """Run a registry act, translating its refusals: an unknown
    prompt is a 404 (checked by the caller), a lint refusal or a
    registry defect is a 400 naming it — nothing was written."""
    from .promptlint import PromptLintError
    from .prompts import PromptRegistryError

    try:
        return fn(*args)
    except PromptLintError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc),
                    "findings": [str(f) for f in exc.findings]}) from exc
    except PromptRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/prompts/{prompt_id}/versions",
          dependencies=[Depends(require_key)])
def add_prompt_version(prompt_id: str, body: PromptVersionIn) -> dict:
    """Add a version of a registry entry. The lint gates
    activation: a text that would break it is refused, and the
    refusal names the findings."""
    if service.get_prompt(prompt_id) is None:
        raise HTTPException(status_code=404, detail="no such prompt")
    entry = _prompt_act_or_http(
        service.add_prompt_version, prompt_id, body.text,
        body.actor, body.note)
    return entry.model_dump(mode="json")


@app.post("/prompts/{prompt_id}/rollback",
          dependencies=[Depends(require_key)])
def rollback_prompt(prompt_id: str, body: PromptRollbackIn) -> dict:
    """Roll an entry back: pin an older version active. The newer
    versions stay on the books, readable forever."""
    if service.get_prompt(prompt_id) is None:
        raise HTTPException(status_code=404, detail="no such prompt")
    pin = _prompt_act_or_http(
        service.rollback_prompt, prompt_id, body.version,
        body.actor, body.note)
    return pin.model_dump(mode="json")


@app.post("/prompts/{prompt_id}/unpin",
          dependencies=[Depends(require_key)])
def unpin_prompt(prompt_id: str, body: PromptUnpinIn) -> dict:
    """Lift an entry's pin: the highest version activates again."""
    if service.get_prompt(prompt_id) is None:
        raise HTTPException(status_code=404, detail="no such prompt")
    _prompt_act_or_http(
        service.unpin_prompt, prompt_id, body.actor, body.note)
    return service.get_prompt(prompt_id)


@app.get("/conversations/{conversation_id}/prompts",
         dependencies=[Depends(require_key)])
def conversation_prompts(conversation_id: str) -> dict:
    """The exact registry texts that served this conversation, at
    the versions recorded on it — the trace from a historical
    reply to the words behind it."""
    rows = service.conversation_prompts(conversation_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return {"conversation_id": conversation_id, "prompts": rows}


@app.get("/incidents", dependencies=[Depends(require_key)])
def list_incidents(status: str | None = None) -> dict:
    """The operator incident feed: every known-issue record, in
    feed order — the service, the areas it covers, the declared
    signals, the status, and the operator's summary the agent
    serves verbatim."""
    rows = []
    for incident in service.list_incidents(status):
        row = incident.model_dump(mode="json")
        row["acts"] = len(service.list_incident_acts(incident.incident_id))
        rows.append(row)
    return {"incidents": rows}


@app.get("/incidents/{incident_id}", dependencies=[Depends(require_key)])
def get_incident(incident_id: str) -> dict:
    """One incident in full, with its operator act log."""
    incident = service.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="no such incident")
    row = incident.model_dump(mode="json")
    row["acts"] = [a.model_dump(mode="json")
                   for a in service.list_incident_acts(incident_id)]
    return row


def _incident_act_or_http(fn, *args, **kwargs):
    """Run an incident-feed act, translating its refusal: a
    broken rule is a 400 naming it — nothing was written."""
    from .incidents import IncidentError

    try:
        return fn(*args, **kwargs)
    except IncidentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/incidents", dependencies=[Depends(require_key)])
def add_incident(body: IncidentIn) -> dict:
    """Add an incident to the feed — a recorded operator act.
    The record is validated against the governed vocabularies
    (service, status, declared signals, a customer-safe summary)
    before anything is written."""
    incident = _incident_act_or_http(
        service.add_incident,
        incident_id=body.incident_id, title=body.title,
        service=body.service, summary=body.summary,
        symptom_signals=body.symptom_signals,
        affected_postcodes=body.affected_postcodes,
        status=body.status, started_at=body.started_at,
        actor=body.actor, note=body.note)
    return incident.model_dump(mode="json")


@app.post("/incidents/{incident_id}/update",
          dependencies=[Depends(require_key)])
def update_incident(incident_id: str, body: IncidentUpdateIn) -> dict:
    """Update an incident's governed fields — a recorded
    operator act. Only the fields the caller sends change."""
    if service.get_incident(incident_id) is None:
        raise HTTPException(status_code=404, detail="no such incident")
    fields = {key: value for key, value in body.model_dump().items()
              if key not in ("actor", "note") and value is not None}
    incident = _incident_act_or_http(
        service.update_incident, incident_id, body.actor, body.note,
        **fields)
    return incident.model_dump(mode="json")


@app.post("/incidents/{incident_id}/resolve",
          dependencies=[Depends(require_key)])
def resolve_incident(incident_id: str, body: IncidentResolveIn) -> dict:
    """Resolve an incident — a recorded operator act. From this
    act on it matches nothing, including in conversations
    already open."""
    if service.get_incident(incident_id) is None:
        raise HTTPException(status_code=404, detail="no such incident")
    incident = _incident_act_or_http(
        service.resolve_incident, incident_id, body.actor, body.note)
    return incident.model_dump(mode="json")


@app.get("/conversations/{conversation_id}/incidents",
         dependencies=[Depends(require_key)])
def conversation_incidents(conversation_id: str) -> dict:
    """The incidents this conversation matched — incident id,
    the signals that matched, the turn — each with the
    incident's live status."""
    rows = service.conversation_incidents(conversation_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return {"conversation_id": conversation_id, "matches": rows}


@app.get("/signals/{conversation_id}", dependencies=[Depends(require_key)])
def get_signals(conversation_id: str) -> dict:
    """One conversation's per-turn signal series; the last record
    is the latest snapshot."""
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    records = service.list_signals(conversation_id)
    return {"conversation_id": conversation_id,
            "signals": [r.model_dump(mode="json") for r in records],
            "latest": (records[-1].model_dump(mode="json")
                       if records else None)}


@app.get("/")
def console() -> FileResponse:
    return FileResponse(_CONSOLE, media_type="text/html")
