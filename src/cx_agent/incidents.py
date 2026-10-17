"""Known-issue awareness: the operator incident feed, and the
deterministic matching that reads it (round 15).

The feed is a governed record set (models.Incident): an operator
writes what is known — the service affected, the postcodes it
covers, the symptom signals that identify it, its status, and a
customer-safe summary in the operator's own words. The dataset
seeds the book's standing incidents the same way it seeds
everything else; after that the feed is operable — an incident
is added, updated or resolved by a recorded operator act (the
act log at the bottom of this module, in the prompt registry's
pattern), never by an edit in place that history cannot see.

Matching is code, and deliberately narrow:

- the incident is **live** — status `active` or `monitoring`;
  a `resolved` incident never matches, so resolving one stops
  every later acknowledgement, including mid-conversation;
- the customer's report carries at least one of the incident's
  declared **symptom signals** (a phrase match over their words,
  the same closed-vocabulary discipline the router uses);
- the **service agrees**: when the report names a procedure's
  symptom, that symptom must be the incident's service — an
  incident about broadband never acknowledges a signal fault,
  however the words overlap;
- the **area agrees**: an incident scoped to postcodes matches
  only a customer whose account postcode is one of them (an
  unscoped incident matches every area). With no account known
  yet, a scoped incident cannot match — the acknowledgement
  waits for the identity the troubleshooting desk asks for
  anyway, and the resumed request is matched again with the
  postcode on record.

That is the whole of it. Matching is bounded by the declared
signals: a report worded outside them follows the normal
troubleshooting path, and the agent has no incident
intelligence beyond the feed — no inferred causes, no
estimated fix times. What the customer is told comes from the
record itself: the title, the status, the start date and the
operator's summary, verbatim.

On a match the supervisor does not start (or continue) a
troubleshooting procedure for the matched symptom and proposes
no unrelated action. The follow-up it offers is the one the
book already has: a technician callback as a CallbackRecord,
proposed through the normal confirmation protocol and settled
when the customer answers it.
"""

from __future__ import annotations

import uuid

from .clock import today
from .models import (
    ActionKind,
    AuditEvent,
    CallbackRecord,
    Conversation,
    Finding,
    Incident,
    IncidentAct,
    ProposedAction,
    SpecialistResult,
)
from . import procedures

# The governed vocabularies. Services are the procedures' own
# symptom keys (dataset.build_procedures) — an incident is about
# a fault the troubleshooting desk would otherwise walk.
SERVICES = ("no_signal", "slow_data", "wifi_calling", "broadband_drops")
STATUSES = ("active", "monitoring", "resolved")
# The statuses that still explain a customer's report. Resolved
# is history: it stays readable on the feed forever and matches
# nothing.
MATCHABLE_STATUSES = ("active", "monitoring")

_CALLBACK_SUMMARY = "book a technician callback within one working day"


class IncidentError(ValueError):
    """An operator act the feed refuses: an unknown incident, a
    duplicate id, a value outside the governed vocabularies, a
    record missing the words it exists to carry. The message
    names the defect; a refused act writes nothing."""


# ---------------------------------------------------------------------------
# Reading the feed
# ---------------------------------------------------------------------------


def customer_postcode(store, conv: Conversation) -> str:
    """The postcode this conversation's account carries, when an
    account is known: the customer's own record, resolved through
    whichever link the conversation holds. "" when no account has
    been identified — a scoped incident cannot match yet."""
    customer = None
    if conv.customer_id:
        customer = store.get_customer(conv.customer_id)
    if customer is None and conv.account_number:
        customer = store.get_customer_by_account(conv.account_number)
    return customer.postcode if customer else ""


def _signals_in(incident: Incident, text: str) -> list[str]:
    lowered = text.lower()
    return [s for s in incident.symptom_signals if s and s.lower() in lowered]


def _area_matches(store, conv: Conversation, incident: Incident) -> bool:
    if not incident.affected_postcodes:
        return True
    return customer_postcode(store, conv) in incident.affected_postcodes


def match_incidents(store, conv: Conversation,
                    text: str) -> list[tuple[Incident, list[str]]]:
    """The live incidents that explain this report, as
    (incident, matched signals) in feed order. Deterministic:
    the same feed, conversation and words always give the same
    answer, and no model is consulted."""
    proc = procedures.match_procedure(store, text)
    out: list[tuple[Incident, list[str]]] = []
    for incident in store.list_incidents():
        if incident.status not in MATCHABLE_STATUSES:
            continue
        if proc is not None and proc.symptom != incident.service:
            continue
        signals = _signals_in(incident, text)
        if not signals:
            continue
        if not _area_matches(store, conv, incident):
            continue
        out.append((incident, signals))
    return out


def match_for_active_run(store, conv: Conversation
                         ) -> tuple[Incident, list[str]] | None:
    """The continuation read: the conversation is already walking
    a procedure when an incident for that very fault is (or
    becomes) known. The run itself is the symptom context — its
    pinned procedure's symptom must be the incident's service,
    the area must agree, and one of the incident's declared
    signals must appear in the customer's own words earlier in
    this conversation (the report that started the walk). A walk
    for a different fault is never swept up by it."""
    state = conv.specialist_state.get("technical", {})
    run_id = state.get("run_id")
    if not run_id:
        return None
    run = store.get_procedure_run(run_id)
    if run is None or run.state != "active":
        return None
    proc = store.get_procedure_version(run.procedure_id,
                                       run.procedure_version)
    if proc is None:
        proc = store.get_procedure(run.procedure_id)
    if proc is None:
        return None
    history = "\n".join(m.text for m in conv.messages if m.role == "customer")
    for incident in store.list_incidents():
        if incident.status not in MATCHABLE_STATUSES:
            continue
        if incident.service != proc.symptom:
            continue
        signals = _signals_in(incident, history)
        if not signals:
            continue
        if not _area_matches(store, conv, incident):
            continue
        return incident, signals
    return None


def match_for_turn(store, conv: Conversation,
                   text: str) -> tuple[Incident, list[str]] | None:
    """The supervisor's one read per technical turn: this
    message's report first, then the live walk's continuation.
    At most one incident is acknowledged per turn — the first
    the feed order gives — so a reply never stacks two known
    issues."""
    matches = match_incidents(store, conv, text)
    if matches:
        return matches[0]
    return match_for_active_run(store, conv)


# ---------------------------------------------------------------------------
# The acknowledgement
# ---------------------------------------------------------------------------


def incident_line(incident: Incident) -> str:
    """The record's own facts as one verbatim-safe line — id,
    title, status and the start date it states, nothing added."""
    return (f"Incident {incident.incident_id}: {incident.title} "
            f"(status: {incident.status}, since {incident.started_at}).")


def acknowledgement_result(ctx, conv: Conversation, incident: Incident,
                           matched_signals: list[str]) -> SpecialistResult:
    """The reply a match produces: the known issue up front, in
    the record's words, and the book's existing follow-up — a
    technician callback, as a record — in place of any
    troubleshooting step or unrelated proposal.

    The callback record is created now, carrying the incident
    (not a procedure transcript: no walk ran), and settles when
    the customer answers the proposal, exactly as a procedure
    dead-end's record settles. With no account known yet the
    desk asks for identity first and the proposal follows on
    the resumed turn, the procedure engine's own pattern."""
    findings = [Finding(
        topic="known_issue",
        text=(f"Before we troubleshoot this: it's a known issue "
              f"on our side. {incident_line(incident)}"))]
    must_include = [incident.summary]
    if not conv.account_number:
        findings.append(Finding(
            topic="identity",
            text=("So I can book that follow-up on the right "
                  "account, tell me your account number (it "
                  "starts NB-) and the postcode on the account.")))
        return SpecialistResult(
            specialist="technical", findings=findings,
            must_include=must_include, needs_identity=True,
            progress=True)
    callback = CallbackRecord(
        callback_id=f"CBK-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=conv.account_number or "",
        procedure_id="",
        procedure_version=0,
        run_id="",
        reason=(f"Known issue {incident.incident_id}: "
                f"{incident.title} — {incident.summary}"),
        status="proposed",
        transcript=[],
        created_date=today().isoformat(),
    )
    ctx.store.save_callback(callback)
    state = conv.specialist_state.setdefault("technical", {})
    state["callback_id"] = callback.callback_id
    findings.append(Finding(
        topic="known_issue_followup",
        text=("Running the device steps won't bite while this is "
              "open, so I won't walk you through them. What I "
              "can do is book a technician callback about this "
              "known issue — the callback record carries the "
              "incident, so the technician starts from it, not "
              "from zero.")))
    return SpecialistResult(
        specialist="technical", findings=findings,
        proposed_action=ProposedAction(
            kind=ActionKind.BOOK_CALLBACK,
            summary=_CALLBACK_SUMMARY,
            payload={"reason": f"known_issue:{incident.incident_id}",
                     "callback_id": callback.callback_id,
                     "incident_id": incident.incident_id},
            basis=(f"Known issue {incident.incident_id} matched on "
                   f"signals {matched_signals} (incident feed).")),
        must_include=must_include, progress=True)


# ---------------------------------------------------------------------------
# Per-conversation matches — the record the surfaces read
# ---------------------------------------------------------------------------


def record_match(store, conv: Conversation, incident: Incident,
                 matched_signals: list[str]) -> None:
    """Write the match where the surfaces look for it: an audit
    event on the conversation (incident id + the matched
    signals) and the conversation's own match list, which the
    handoff packet and the per-conversation view read."""
    store.save_audit(AuditEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id, kind="incident",
        detail={"event": "incident_matched",
                "incident_id": incident.incident_id,
                "matched_signals": list(matched_signals),
                "service": incident.service,
                "status": incident.status},
        turn=conv.turns))
    bucket = conv.specialist_state.setdefault("incidents", {})
    matches = bucket.setdefault("matches", [])
    matches.append({
        "incident_id": incident.incident_id,
        "turn": conv.turns,
        "matched_signals": list(matched_signals),
        "status": incident.status,
    })


def conversation_matches(store, conv: Conversation) -> list[dict]:
    """This conversation's incident matches, each enriched with
    the incident's live record — a match whose incident has
    since resolved reads as resolved here, however it stood when
    it matched."""
    bucket = conv.specialist_state.get("incidents", {})
    out = []
    for match in bucket.get("matches", []):
        incident = store.get_incident(match["incident_id"])
        out.append({
            **match,
            "title": incident.title if incident else "",
            "current_status": incident.status if incident else "",
        })
    return out


def packet_incidents(store, conv: Conversation) -> list[dict]:
    """The known-issue note for a handoff packet: the incidents
    this conversation matched that are STILL live on the feed,
    as the records state them now (id, title, status, summary).
    A match whose incident resolved before the handoff is
    history, and does not ride the packet as an open issue."""
    seen: list[str] = []
    out: list[dict] = []
    bucket = conv.specialist_state.get("incidents", {})
    for match in bucket.get("matches", []):
        incident_id = match["incident_id"]
        if incident_id in seen:
            continue
        seen.append(incident_id)
        incident = store.get_incident(incident_id)
        if incident is None or incident.status not in MATCHABLE_STATUSES:
            continue
        out.append({
            "incident_id": incident.incident_id,
            "title": incident.title,
            "status": incident.status,
            "summary": incident.summary,
        })
    return out


# ---------------------------------------------------------------------------
# Operator acts — the feed is operable, and every act is recorded
# ---------------------------------------------------------------------------

_UPDATE_FIELDS = ("title", "service", "affected_postcodes",
                  "symptom_signals", "status", "started_at", "summary")


def _validate(incident: Incident) -> None:
    if not incident.title.strip():
        raise IncidentError(
            f"incident {incident.incident_id!r}: a title is required")
    if incident.service not in SERVICES:
        raise IncidentError(
            f"incident {incident.incident_id!r}: unknown service "
            f"{incident.service!r} (expected one of "
            f"{', '.join(SERVICES)})")
    if incident.status not in STATUSES:
        raise IncidentError(
            f"incident {incident.incident_id!r}: unknown status "
            f"{incident.status!r} (expected one of "
            f"{', '.join(STATUSES)})")
    if not incident.symptom_signals:
        raise IncidentError(
            f"incident {incident.incident_id!r}: at least one "
            f"symptom signal is required — matching is bounded by "
            f"the declared signals, so an incident without any "
            f"could never honestly match")
    if not incident.summary.strip():
        raise IncidentError(
            f"incident {incident.incident_id!r}: a customer-safe "
            f"summary is required — it is the only account of the "
            f"issue the agent may give")


def _record_act(store, kind: str, incident_id: str, actor: str,
                note: str) -> IncidentAct:
    seq = max((a.seq for a in store.list_incident_acts()), default=0) + 1
    act = IncidentAct(
        act_id=f"IACT-{uuid.uuid4().hex[:10]}", seq=seq, kind=kind,
        incident_id=incident_id, actor=actor, note=note,
        created_date=today().isoformat())
    store.save_incident_act(act)
    return act


def add_incident(store, *, title: str, service: str, summary: str,
                 symptom_signals: list[str],
                 affected_postcodes: list[str] | None = None,
                 status: str = "active", started_at: str = "",
                 incident_id: str = "", actor: str = "",
                 note: str = "") -> Incident:
    """Add an incident to the feed. The id is the operator's
    when given (and must be new); otherwise one is minted. The
    record is validated against the governed vocabularies
    before anything is written."""
    incident = Incident(
        incident_id=incident_id or f"INC-{uuid.uuid4().hex[:8].upper()}",
        title=title, service=service,
        affected_postcodes=list(affected_postcodes or []),
        symptom_signals=list(symptom_signals), status=status,
        started_at=started_at or today().isoformat(), summary=summary,
        updated_date=today().isoformat())
    _validate(incident)
    if store.get_incident(incident.incident_id) is not None:
        raise IncidentError(
            f"incident {incident.incident_id!r} already exists — "
            f"update it instead of adding it again")
    store.save_incident(incident)
    _record_act(store, "add", incident.incident_id, actor, note)
    return incident


def update_incident(store, incident_id: str, actor: str = "",
                    note: str = "", **fields) -> Incident:
    """Update an incident's fields — the summary the agent
    serves, its signals, its scope, its status. Only the
    governed fields may change; the updated record is validated
    as a whole, so an act cannot leave the feed carrying a
    record matching could not honestly read."""
    incident = store.get_incident(incident_id)
    if incident is None:
        raise IncidentError(f"unknown incident {incident_id!r}")
    unknown = sorted(set(fields) - set(_UPDATE_FIELDS))
    if unknown:
        raise IncidentError(
            f"incident {incident_id!r}: cannot update "
            f"{', '.join(unknown)} (governed fields: "
            f"{', '.join(_UPDATE_FIELDS)})")
    for key, value in fields.items():
        setattr(incident, key, value)
    _validate(incident)
    incident.updated_date = today().isoformat()
    store.save_incident(incident)
    _record_act(store, "update", incident_id, actor, note)
    return incident


def resolve_incident(store, incident_id: str, actor: str = "",
                     note: str = "") -> Incident:
    """Resolve an incident: its status becomes `resolved`,
    which stops it matching from this act on — including in
    conversations already open. The record stays on the feed,
    readable forever; resolving an already-resolved incident is
    refused, so the act log never claims a change that did not
    happen."""
    incident = store.get_incident(incident_id)
    if incident is None:
        raise IncidentError(f"unknown incident {incident_id!r}")
    if incident.status == "resolved":
        raise IncidentError(
            f"incident {incident_id!r} is already resolved")
    incident.status = "resolved"
    incident.updated_date = today().isoformat()
    store.save_incident(incident)
    _record_act(store, "resolve", incident_id, actor, note)
    return incident
