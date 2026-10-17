"""The care ledger: what the care desk writes down, and the watch
that reads it back.

Three of the things a customer says at the catch-all desk are
records, not conversation filler, and this module is where they
become records:

- **Complaints.** Recognised by a complaint marker in the
  customer's words, categorised against a governed list by rules
  in code, and written the moment they're recognised — words
  verbatim, the response deadline policy promises, status open.
  The SLA watch below scans the open records: breached ones are
  the queue's business, near-breach ones are flagged for the team.
- **Compliments and feedback.** Recorded verbatim, kind decided by
  whether the words carry praise. Bare "thanks" is small talk and
  stays conversation; a named subject or an explicit feedback
  marker makes it data.
- **Accessibility needs.** A stated need ("send my letters in
  large print") is recorded on the account — the words verbatim
  and the derived contact preference — and later contacts honour
  the preference from the record.

Everything here is deterministic: no model decides what a
complaint is, which category it falls in, or when it breaches.
"""

from __future__ import annotations

import re
import uuid
from datetime import date, timedelta

from .clock import today
from .models import ComplaintRecord, FeedbackRecord

# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------

# A complaint is a complaint when the customer says it is: the
# marker words. Unhappiness without the marker is a service
# problem for the desks, not a complaint record.
_COMPLAINT_MARKERS = [
    r"\bcomplain\b", r"\bcomplaint\b", r"\bcomplaining\b",
]

# The governed category list, in the order the rules read. Each
# category is a set of plain keyword classes over the customer's
# words; the first class that matches decides, so a "rude engineer
# who never showed up" is staff conduct, not an appointment.
COMPLAINT_CATEGORIES: dict[str, list[str]] = {
    "staff_conduct": [
        "rude", "unhelpful", "hung up", "attitude", "shouted",
        "swore at", "dismissive", "patronising", "patronizing",
    ],
    "appointment": [
        "appointment", "no-show", "no show", "never showed",
        "didn't show", "did not show", "didn't turn up",
        "did not turn up", "never turned up", "missed the visit",
        "missed visit", "engineer didn't", "engineer did not",
        "technician didn't", "nobody showed", "no one showed",
        "waited in all day",
    ],
    "billing": [
        "bill", "charge", "invoice", "refund", "overcharged",
        "direct debit", "payment taken",
    ],
    "coverage": [
        "coverage", "no signal", "no service", "reception",
        "blackspot", "black spot", "signal",
    ],
    "service_quality": [
        "slow", "keeps dropping", "not working", "fault", "app",
        "website", "service", "broadband", "connection",
    ],
}


def complaint_category(text: str) -> str | None:
    """The governed category for a complaint, or None when the
    text carries no complaint marker at all — the caller is not
    looking at a complaint. Category rules read the whole text;
    the first matching class in the governed order decides, and a
    complaint about nothing in particular is "other"."""
    lowered = f" {text.lower()} "
    if not any(re.search(p, lowered) for p in _COMPLAINT_MARKERS):
        return None
    for category, words in COMPLAINT_CATEGORIES.items():
        if any(w in lowered for w in words):
            return category
    return "other"


def open_complaint(store, conv, text: str, category: str, policy,
                   account_number: str = "") -> ComplaintRecord:
    """Write the complaint record the moment it's recognised:
    the customer's words verbatim, and the response deadline
    computed from policy (created + complaint_response_days).
    The record attaches to the conversation's account, or to the
    account the customer names in the message — identification,
    exactly as the accessibility record attaches; a complaint
    needs no verification to be written down."""
    number = account_number or conv.account_number \
        or account_number_in(text)
    if number and conv.account_number is None \
            and store.get_account(number) is not None:
        customer = store.get_customer_by_account(number)
        conv.account_number = number
        if customer is not None:
            conv.customer_id = customer.customer_id
    created = today()
    record = ComplaintRecord(
        complaint_id=f"CMP-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=number or "",
        category=category,
        text=text,
        status="open",
        created_date=created.isoformat(),
        response_due=(created + timedelta(
            days=policy.complaint_response_days)).isoformat(),
    )
    store.save_complaint(record)
    return record


def complaint_sla_state(record: ComplaintRecord, policy,
                        on: date | None = None) -> str:
    """Where one complaint stands against its deadline, derived
    from the record and the calendar: closed/responded records are
    settled; an open record past its due date is breached; within
    the near-breach window it is due_soon; otherwise open."""
    if record.status != "open":
        return "settled"
    on = on or today()
    due = date.fromisoformat(record.response_due)
    if due < on:
        return "breached"
    if (due - on).days <= policy.complaint_near_breach_days:
        return "due_soon"
    return "open"


def complaint_watch(store, policy, on: date | None = None) -> dict:
    """The SLA watch scan: every still-open complaint, split by
    where it stands against its deadline. Pure — it raises
    nothing; raising queue entries is the service's act
    (service.run_complaint_watch), so the scan can be read by the
    API, the console and the eval gate alike."""
    on = on or today()
    breached: list[ComplaintRecord] = []
    due_soon: list[ComplaintRecord] = []
    for record in store.list_complaints():
        state = complaint_sla_state(record, policy, on)
        if state == "breached":
            breached.append(record)
        elif state == "due_soon":
            due_soon.append(record)
    return {"on": on.isoformat(), "breached": breached,
            "due_soon": due_soon}


# ---------------------------------------------------------------------------
# Compliments and feedback
# ---------------------------------------------------------------------------

_FEEDBACK_MARKERS = [
    r"\bfeedback\b", r"\bcompliment\b", r"\bkudos\b",
    r"\bwell done\b", r"\bcredit where\b", r"\bwant to praise\b",
    r"\bpass on my thanks\b", r"\bsay well done\b",
]

# Praise words and the subjects praise attaches to: either an
# explicit marker, or praise aimed at a named subject, makes a
# message feedback data rather than a conversational thank-you.
_PRAISE_WORDS = [
    "brilliant", "excellent", "fantastic", "amazing", "superb",
    "wonderful", "great service", "lovely", "outstanding",
    "really helpful", "so helpful", "so patient",
]
_FEEDBACK_SUBJECTS = [
    "engineer", "technician", "agent", "advisor", "team",
    "service", "app", "website", "man", "woman", "lady", "guy",
    "person", "staff",
]


def feedback_kind(text: str) -> str | None:
    """compliment | feedback | None (not feedback at all — an
    ordinary thank-you or closing pleasantry). An explicit marker
    or praise aimed at a subject records it; praise in the words
    makes it a compliment, a bare marker records it as feedback."""
    lowered = f" {text.lower()} "
    marked = any(re.search(p, lowered) for p in _FEEDBACK_MARKERS)
    praised = any(w in lowered for w in _PRAISE_WORDS)
    aimed = any(w in lowered for w in _FEEDBACK_SUBJECTS)
    if not (marked or (praised and aimed)):
        return None
    return "compliment" if praised else "feedback"


def record_feedback(store, conv, text: str, kind: str,
                    account_number: str = "") -> FeedbackRecord:
    record = FeedbackRecord(
        record_id=f"FBK-{uuid.uuid4().hex[:10]}",
        conversation_id=conv.conversation_id,
        account_number=account_number or conv.account_number or "",
        kind=kind,
        text=text,
        created_date=today().isoformat(),
    )
    store.save_feedback(record)
    return record


# ---------------------------------------------------------------------------
# Accessibility and contact preferences
# ---------------------------------------------------------------------------

# The ways a customer states an access need, as phrases. A match
# means "record this": the need is stated, not hinted — small
# print the customer cannot read, a channel they cannot use.
_ACCESSIBILITY_MARKERS = [
    "large print", "small print", "braille", "easy read",
    "can't read", "cannot read", "struggle to read",
    "struggling to read", "hard of hearing", "hearing loop",
    "text relay", "sign language", "letters too small",
    "print is too small", "can't see the letters",
    "cannot see the letters", "visually impaired",
    "email me instead", "by email instead", "text me instead",
    "don't phone me", "do not phone me", "can't hear on the phone",
    "cannot hear on the phone", "struggle on the phone",
]

# The preference a stated need derives, in reading order: the
# first the customer's words support. This is the instruction
# later contacts follow — the note keeps the words themselves.
_PREFERENCE_RULES: list[tuple[str, list[str]]] = [
    ("large print", ["large print", "small print", "letters too small",
                     "print is too small"]),
    ("braille", ["braille"]),
    ("easy read", ["easy read"]),
    ("text relay", ["text relay"]),
    ("sign language", ["sign language"]),
    ("email", ["email"]),
    ("text message", ["text me", "text message", "by text"]),
]

_ACCOUNT_RE = re.compile(r"\b(NB-\d{4,})\b", re.IGNORECASE)


def accessibility_need(text: str) -> bool:
    """True when the customer states an accessibility or contact
    need in so many words. Tone, age and apologies are not needs
    and never match — the same discipline the vulnerability flag
    follows."""
    lowered = f" {text.lower()} "
    return any(m in lowered for m in _ACCESSIBILITY_MARKERS)


def preferred_contact_for(text: str) -> str:
    """The contact preference a stated need derives — the first
    rule the customer's words support. "" when the need names no
    channel (the verbatim note still carries it)."""
    lowered = f" {text.lower()} "
    for preference, words in _PREFERENCE_RULES:
        if any(w in lowered for w in words):
            return preference
    return ""


def account_number_in(text: str) -> str:
    """The account number stated in a message, if one is — care
    records attach to the account the customer names, exactly as
    the identity flow reads it."""
    match = _ACCOUNT_RE.search(text)
    return match.group(1).upper() if match else ""


def record_accessibility(store, conv, text: str,
                         account_number: str = ""):
    """Record a stated need on the account: the words verbatim in
    the note, the derived preference beside them. Sets the
    conversation's account link (identification — the customer
    named their own account for their own record). Returns the
    updated account, or None when no account can be resolved —
    the caller asks for the number and holds the note meanwhile."""
    number = account_number or conv.account_number \
        or account_number_in(text)
    if not number:
        return None
    account = store.get_account(number)
    if account is None:
        return None
    account.accessibility_note = text
    account.preferred_contact = preferred_contact_for(text)
    store.save_account(account)
    if conv.account_number is None:
        customer = store.get_customer_by_account(number)
        conv.account_number = number
        if customer is not None:
            conv.customer_id = customer.customer_id
    return account


def contact_preference_line(account) -> str:
    """The line a later contact honours the record with, verbatim-
    safe for must_include: what the customer asked for, stated as
    the commitment it became."""
    if account is None or not account.preferred_contact:
        return ""
    return (f"You asked us to contact you in {account.preferred_contact} — "
            f"that's what's on your record, so everything we send you "
            f"will be in {account.preferred_contact}.")
