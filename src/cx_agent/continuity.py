"""Continuity (round 16): the briefing a returning customer is
opened with, and the promises ledger — both computed in code
from the account's records, never from anything a model
remembers about the customer.

The briefing is built at the moment identity verifies: the open
items with their deadlines, the standing flags honoured from the
first turn, the last conversation's outcome in one line, the
promises already broken, a proposal left pending, and the
resumable marker a previous conversation left behind. Every
element carries the id of the record it came from; a test
asserts no element exists without its backing record. An
account with no history gets an empty briefing and an ordinary
opening — continuity is never fabricated.

The promises ledger is a read, not a second store — the same
discipline as the complaint watch. Each commitment already in
the book (a complaint's response deadline, a booked technician
callback, an arrangement instalment) derives a due state from
its own record and the shared clock: kept, pending, or broken.
Nothing in this module acts on a broken promise: no apology is
sent and no compensation is raised. The ledger surfaces — in
the next conversation's briefing, in the daily digest, and on
the promises views — and a person decides what to do.
"""

from __future__ import annotations

from datetime import date, timedelta

from . import payments_book
from .clock import today
from .models import ContinuityBriefing, PromiseEntry

_STATE_ORDER = {"broken": 0, "pending": 1, "kept": 2}


def _before(record_date: str, on: date) -> bool:
    """A record dated after the as-of date is not on the book yet."""
    return not record_date or record_date <= on.isoformat()


# ---------------------------------------------------------------------------
# The promises ledger — a derivation over the commitment records
# ---------------------------------------------------------------------------

def _complaint_promise(record, on: date) -> PromiseEntry | None:
    if not _before(record.created_date, on):
        return None
    if record.status != "open":
        state = "kept"  # responded or closed: the deadline was met
    elif record.response_due and date.fromisoformat(record.response_due) < on:
        state = "broken"  # the date passed with the complaint still open
    else:
        state = "pending"
    return PromiseEntry(
        promise_id=f"PRM-{record.complaint_id}",
        kind="complaint_response",
        account_number=record.account_number,
        source_id=record.complaint_id,
        conversation_id=record.conversation_id,
        summary=(f"Substantive response to complaint {record.complaint_id} "
                 f"({record.category.replace('_', ' ')})"),
        due_date=record.response_due,
        state=state)


def _callback_promise(store, record, on: date) -> PromiseEntry | None:
    # Only a settled booking is a commitment: a proposal the
    # customer hasn't answered made no promise, and a declined
    # one was released by the customer themselves.
    if record.status not in ("booked", "blocked"):
        return None
    if not _before(record.created_date, on):
        return None
    proc = store.get_procedure(record.procedure_id)
    title = proc.title if proc else "troubleshooting"
    # The booking's own words promise the callback "within one
    # working day"; the ledger carries the next calendar day as
    # the due date the commitment is read against.
    due = ""
    if record.created_date:
        due = (date.fromisoformat(record.created_date)
               + timedelta(days=1)).isoformat()
    return PromiseEntry(
        promise_id=f"PRM-{record.callback_id}",
        kind="callback",
        account_number=record.account_number,
        source_id=record.callback_id,
        conversation_id=record.conversation_id,
        summary=f"Technician callback for the {title} problem",
        due_date=due,
        # A callback made is kept; a booking stopped at execution
        # is a promise the system made and did not keep.
        state="kept" if record.status == "booked" else "broken")


def _arrangement_promises(arr, on: date) -> list[PromiseEntry]:
    if not _before(arr.created_date, on):
        return []
    out = []
    for inst in arr.installments:
        if inst.paid_amount >= inst.amount:
            state = "kept"
        elif date.fromisoformat(inst.due_date) < on:
            state = "broken"
        else:
            state = "pending"
        out.append(PromiseEntry(
            promise_id=f"PRM-{arr.arrangement_id}-{inst.number}",
            kind="arrangement_instalment",
            account_number=arr.account_number,
            source_id=arr.arrangement_id,
            conversation_id=arr.conversation_id,
            summary=(f"Arrangement {arr.arrangement_id} instalment "
                     f"{inst.number} of ${inst.amount}"),
            due_date=inst.due_date,
            state=state))
    return out


def promise_ledger(store, account_number: str | None = None,
                   on: date | None = None) -> list[PromiseEntry]:
    """Every commitment on the book, derived from its record:
    broken first, then pending, then kept — each group by due
    date. `on` reads the ledger as of a date (the digest's day
    end); the default is today, by the shared clock."""
    on = on or today()
    entries: list[PromiseEntry] = []
    for record in store.list_complaints(account_number=account_number):
        entry = _complaint_promise(record, on)
        if entry is not None:
            entries.append(entry)
    for record in store.list_callbacks():
        if account_number and record.account_number != account_number:
            continue
        entry = _callback_promise(store, record, on)
        if entry is not None:
            entries.append(entry)
    for arr in store.list_payment_arrangements(account_number=account_number):
        entries.extend(_arrangement_promises(arr, on))
    entries.sort(key=lambda e: (_STATE_ORDER[e.state], e.due_date,
                                e.promise_id))
    return entries


def promise_summary(store, on: date | None = None) -> dict:
    """The digest's promises section: the ledger's state at day
    end, with the broken entries in full — summaries and due
    dates from their records, for a person to act on."""
    entries = promise_ledger(store, on=on)
    broken = [e for e in entries if e.state == "broken"]
    return {
        "broken_count": len(broken),
        "pending_count": sum(1 for e in entries if e.state == "pending"),
        "kept_count": sum(1 for e in entries if e.state == "kept"),
        "broken": [e.model_dump(mode="json") for e in broken],
    }


# ---------------------------------------------------------------------------
# The continuity briefing — the account's records, as an opening
# ---------------------------------------------------------------------------

def _outcome_line(outcome: str, desk: str) -> str:
    matter = f"your {desk} matter" if desk else "your last matter"
    if outcome in ("resolved", "resolved_with_action", "resolved_goodwill"):
        return f"Last time we spoke, we resolved {matter} here."
    if outcome == "escalated":
        return f"Last time we spoke, we passed {matter} to a person."
    return f"Last time we spoke, {matter} didn't reach an answer."


def _last_outcome(store, account_number: str,
                  exclude_conversation_id: str) -> tuple[str, str]:
    """The last conversation's outcome in one line: the
    attribution record when the daily attribution has run,
    otherwise the conversation's own closing state. One line,
    from one record — (line, source conversation id)."""
    others = [c for c in store.list_conversations()
              if c.account_number == account_number
              and c.conversation_id != exclude_conversation_id]
    attributed = []
    for conv in others:
        record = store.get_attribution(conv.conversation_id)
        if record is not None:
            attributed.append(record)
    if attributed:
        record = max(attributed,
                     key=lambda r: (r.attributed_date, r.conversation_id))
        return _outcome_line(record.outcome, record.desk), record.conversation_id
    for conv in reversed(others):
        desk = conv.intent.value if conv.intent.value != "unknown" else ""
        if conv.status == "resolved":
            return _outcome_line("resolved", desk), conv.conversation_id
        if conv.status == "escalated":
            return _outcome_line("escalated", desk), conv.conversation_id
    return "", ""


def build_briefing(store, account_number: str, on: date | None = None,
                   exclude_conversation_id: str = "") -> ContinuityBriefing:
    """The briefing for one account, from its records alone.
    Pure: it writes nothing. (Adopting a resumable run into the
    live conversation is the supervisor's act, not the read's.)"""
    on = on or today()
    briefing = ContinuityBriefing(account_number=account_number)
    account = store.get_account(account_number)
    customer = (store.get_customer(account.customer_id)
                if account is not None else None)
    briefing.flags = {
        "accessibility_note": account.accessibility_note if account else "",
        "preferred_contact": account.preferred_contact if account else "",
        "vulnerable": bool(customer and customer.vulnerable),
    }
    briefing.broken_promises = [
        e.model_dump(mode="json")
        for e in promise_ledger(store, account_number, on)
        if e.state == "broken"]
    broken_sources = {e["source_id"] for e in briefing.broken_promises}

    broken_lines: list[str] = []
    item_lines: list[str] = []

    for record in store.list_complaints(account_number=account_number):
        if record.status != "open":
            continue
        briefing.open_items.append({
            "kind": "complaint", "ref_id": record.complaint_id,
            "summary": f"complaint ({record.category.replace('_', ' ')})",
            "due_date": record.response_due})
        if record.complaint_id in broken_sources:
            broken_lines.append(
                f"I can see your complaint from {record.created_date} is "
                f"still open, and its response deadline "
                f"({record.response_due}) has passed — that missed "
                f"deadline is flagged on your record.")
        else:
            item_lines.append(
                f"I can see your complaint from {record.created_date} is "
                f"still open — we're due to respond by "
                f"{record.response_due}.")

    for case in store.list_dispute_cases(account_number=account_number):
        if case.state not in ("opened", "escalated_above_cap"):
            continue
        briefing.open_items.append({
            "kind": "dispute", "ref_id": case.case_id,
            "summary": "billing dispute", "due_date": ""})
        if case.state == "escalated_above_cap":
            item_lines.append(
                f"Your billing dispute from {case.created_date} is with "
                f"a person on the billing team.")
        else:
            item_lines.append(
                f"Your billing dispute from {case.created_date} is "
                f"still open.")

    for record in store.list_callbacks():
        if (record.account_number != account_number
                or record.status != "booked"):
            continue
        proc = store.get_procedure(record.procedure_id)
        title = proc.title if proc else "troubleshooting"
        briefing.open_items.append({
            "kind": "callback", "ref_id": record.callback_id,
            "summary": f"technician callback ({title})",
            "due_date": ""})
        item_lines.append(
            f"A technician callback was booked on {record.created_date} "
            f"for the {title} problem — that's still on the book.")

    for arr in store.list_payment_arrangements(account_number=account_number):
        state = payments_book.arrangement_state(arr, on)
        if state == "completed":
            continue
        next_due = next((i for i in arr.installments
                         if i.paid_amount < i.amount), None)
        if next_due is None:
            continue
        briefing.open_items.append({
            "kind": "arrangement", "ref_id": arr.arrangement_id,
            "summary": "payment arrangement",
            "due_date": next_due.due_date})
        if state == "broken":
            broken_lines.append(
                f"Your payment arrangement has an instalment of "
                f"${next_due.amount} that was due on {next_due.due_date} "
                f"and hasn't been covered — it's flagged on your record.")
        else:
            item_lines.append(
                f"Your payment arrangement has ${next_due.amount} due "
                f"on {next_due.due_date}.")

    # The resumable marker: an active procedure run another
    # conversation left behind. The run is the record — the next
    # conversation continues it, it does not start over.
    runs = [r for r in store.list_procedure_runs()
            if r.account_number == account_number
            and r.state == "active"
            and r.conversation_id != exclude_conversation_id]
    resume_line = ""
    if runs:
        run = max(runs, key=lambda r: (r.updated_date, r.created_date,
                                       r.run_id))
        proc = store.get_procedure(run.procedure_id)
        title = proc.title if proc else run.procedure_id
        briefing.resumable_run_id = run.run_id
        briefing.resumable_summary = (
            f"{title} walk, waiting at step {run.current_step_id}")
        resume_line = (
            f"We were partway through the {title} troubleshooting last "
            f"time — the walk is saved where we left it, and we can "
            f"pick it up from the step we were on.")

    # A proposal left unconfirmed in an earlier conversation is
    # NAMED from its record, never carried over: acting on it
    # needs a fresh proposal from live figures and a fresh yes
    # (the round 2 / round 5 discipline). It counts as still
    # pending only while both records say so — the conversation
    # still holds it (every settle path clears it) and its action
    # record still stands in proposed state (the same matching
    # the supervisor's retire uses).
    pending_line = ""
    for conv in reversed(store.list_conversations()):
        if (conv.conversation_id == exclude_conversation_id
                or conv.account_number != account_number
                or conv.pending_action is None):
            continue
        pending = conv.pending_action
        still_proposed = any(
            r.status == "proposed" and r.kind == pending.kind
            and r.summary == pending.summary
            for r in store.list_actions(conv.conversation_id))
        if not still_proposed:
            continue
        briefing.pending_proposal = {
            "kind": pending.kind.value,
            "summary": pending.summary,
            "conversation_id": conv.conversation_id}
        pending_line = (
            f"You also had this waiting for a decision last time: "
            f"{pending.summary}. We can look at it again "
            f"with today's figures whenever you're ready.")
        break

    outcome, source = _last_outcome(store, account_number,
                                    exclude_conversation_id)
    briefing.last_outcome = outcome
    briefing.last_outcome_source = source

    lines: list[str] = []
    if outcome:
        lines.append(outcome)
    lines.extend(broken_lines)
    if resume_line:
        lines.append(resume_line)
    lines.extend(item_lines)
    if pending_line:
        lines.append(pending_line)
    if briefing.flags["preferred_contact"]:
        lines.append(
            f"I've noted your preference for contact by "
            f"{briefing.flags['preferred_contact']}.")
    briefing.lines = lines
    return briefing
