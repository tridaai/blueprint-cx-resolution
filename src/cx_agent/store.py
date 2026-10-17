"""Persistence behind one interface.

Three implementations of the same protocol:

- **PostgreSQL** — production. JSONB payloads, numbered SQL
  migrations in `migrations/` run at startup, pgvector for policy
  embeddings.
- **SQLite** — the local default. One file (`./data/cx.db` unless
  `SQLITE_PATH` says otherwise), the same tables and payload shapes
  via the SQLite-dialect migrations in `migrations/sqlite/`, money
  held as decimal strings inside the JSON payloads, embeddings
  stored as JSON with similarity computed in code — no pgvector,
  and no database server to install.
- **In-memory** — a test double for tests and evals, never the
  production or local story.

Selection happens in exactly one place: `make_store()` (via
`store_profile()` in config). `DATABASE_URL` set means Postgres;
otherwise the SQLite file. Nothing else in the app knows which store
is running.

Embeddings carry one contract on every store: EMBEDDING_DIM
dimensions (documents.py; the Postgres column is `vector(384)`).
`set_embedding` enforces it in code on all three implementations.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import sqlite3
import threading
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Protocol

from .config import store_profile
from .documents import EMBEDDING_DIM
from .models import (
    Account,
    ActionRecord,
    AnnotationRecord,
    AttributionRecord,
    AuditEvent,
    Bill,
    CallbackRecord,
    ComplaintRecord,
    Conversation,
    Customer,
    DisputeCase,
    Escalation,
    FeedbackRecord,
    Offer,
    PaymentArrangement,
    PaymentAttempt,
    PaymentNotice,
    Plan,
    PolicyDocument,
    Procedure,
    ProcedureRun,
    RoamingNotice,
    RoamingRate,
)

DEFAULT_SQLITE_PATH = "data/cx.db"


def _migrations_dir(dialect: str = "") -> Path:
    """Locate the numbered migrations for a dialect ("" = PostgreSQL,
    "sqlite" = SQLite). `./migrations` relative to the working
    directory wins (the compose image and a repo checkout both run
    from the root); otherwise the path anchored at this package's
    repo root."""
    local = Path("migrations") / dialect if dialect else Path("migrations")
    if local.exists():
        return local
    return Path(__file__).resolve().parents[2] / "migrations" / dialect


def _json_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"not serialisable: {type(obj)!r}")


def dump(model) -> str:
    return json.dumps(model.model_dump(), default=_json_default)


class EmbeddingDimensionError(ValueError):
    """An embedding was offered at the wrong dimension. Raised by
    every store before anything is written, so the failure is this
    deliberate message everywhere, not only where pgvector rejects
    it."""


def _check_dim(vector: list[float]) -> None:
    if len(vector) != EMBEDDING_DIM:
        raise EmbeddingDimensionError(
            f"embedding must be {EMBEDDING_DIM} dimensions, got {len(vector)}"
        )


class Store(Protocol):
    profile: str

    # reference book
    def save_customer(self, c: Customer) -> None: ...
    def get_customer(self, customer_id: str) -> Customer | None: ...
    def get_customer_by_account(self, account_number: str) -> Customer | None: ...
    def save_account(self, a: Account) -> None: ...
    def get_account(self, account_number: str) -> Account | None: ...
    def save_plan(self, p: Plan) -> None: ...
    def get_plan(self, plan_id: str) -> Plan | None: ...
    def list_plans(self) -> list[Plan]: ...
    def save_offer(self, o: Offer) -> None: ...
    def get_offer(self, offer_id: str) -> Offer | None: ...
    def list_offers(self) -> list[Offer]: ...
    def save_bill(self, b: Bill) -> None: ...
    def get_bill_for_account(self, account_number: str) -> Bill | None: ...
    def list_bills_for_account(self, account_number: str) -> list[Bill]: ...
    def save_policy(self, d: PolicyDocument) -> None: ...
    def get_policy(self, doc_id: str) -> PolicyDocument | None: ...
    def list_policies(self) -> list[PolicyDocument]: ...
    def set_embedding(self, doc_id: str, vector: list[float]) -> None: ...
    def list_embeddings(self) -> list[tuple[str, list[float]]]: ...
    def save_procedure(self, p: Procedure) -> None: ...
    def get_procedure(self, procedure_id: str) -> Procedure | None: ...
    def get_procedure_version(self, procedure_id: str,
                              version: int) -> Procedure | None: ...
    def list_procedures(self) -> list[Procedure]: ...
    def save_procedure_run(self, r: ProcedureRun) -> None: ...
    def get_procedure_run(self, run_id: str) -> ProcedureRun | None: ...
    def list_procedure_runs(self, conversation_id: str | None = None
                            ) -> list[ProcedureRun]: ...
    def save_callback(self, c: CallbackRecord) -> None: ...
    def get_callback(self, callback_id: str) -> CallbackRecord | None: ...
    def list_callbacks(self, conversation_id: str | None = None
                       ) -> list[CallbackRecord]: ...
    def save_roaming_rate(self, r: RoamingRate) -> None: ...
    def get_roaming_rate(self, destination: str) -> RoamingRate | None: ...
    def list_roaming_rates(self) -> list[RoamingRate]: ...
    def save_roaming_notice(self, n: RoamingNotice) -> None: ...
    def list_roaming_notices(self, account_number: str | None = None
                             ) -> list[RoamingNotice]: ...
    def save_payment_attempt(self, t: PaymentAttempt) -> None: ...
    def list_payment_attempts(self, account_number: str | None = None,
                              conversation_id: str | None = None
                              ) -> list[PaymentAttempt]: ...
    def save_payment_arrangement(self, a: PaymentArrangement) -> None: ...
    def get_payment_arrangement(self, arrangement_id: str
                                ) -> PaymentArrangement | None: ...
    def list_payment_arrangements(self, account_number: str | None = None
                                  ) -> list[PaymentArrangement]: ...
    def save_payment_notice(self, n: PaymentNotice) -> None: ...
    def list_payment_notices(self, account_number: str | None = None
                             ) -> list[PaymentNotice]: ...
    def save_complaint(self, c: ComplaintRecord) -> None: ...
    def get_complaint(self, complaint_id: str) -> ComplaintRecord | None: ...
    def list_complaints(self, conversation_id: str | None = None,
                        account_number: str | None = None
                        ) -> list[ComplaintRecord]: ...
    def save_feedback(self, f: FeedbackRecord) -> None: ...
    def list_feedback(self, conversation_id: str | None = None,
                      account_number: str | None = None
                      ) -> list[FeedbackRecord]: ...

    # conversation records
    def save_conversation(self, c: Conversation) -> None: ...
    def get_conversation(self, conversation_id: str) -> Conversation | None: ...
    def list_conversations(self) -> list[Conversation]: ...
    def save_action(self, a: ActionRecord) -> None: ...
    def list_actions(self, conversation_id: str | None = None) -> list[ActionRecord]: ...
    def save_dispute_case(self, d: DisputeCase) -> None: ...
    def get_dispute_case(self, case_id: str) -> DisputeCase | None: ...
    def list_dispute_cases(self, conversation_id: str | None = None,
                           account_number: str | None = None) -> list[DisputeCase]: ...
    def save_escalation(self, e: Escalation) -> None: ...
    def list_escalations(self) -> list[Escalation]: ...
    def save_attribution(self, r: AttributionRecord) -> None: ...
    def get_attribution(self, conversation_id: str) -> AttributionRecord | None: ...
    def list_attributions(self) -> list[AttributionRecord]: ...
    def save_annotation(self, a: AnnotationRecord) -> None: ...
    def list_annotations(self, conversation_id: str | None = None
                         ) -> list[AnnotationRecord]: ...
    def save_audit(self, e: AuditEvent) -> None: ...
    def list_audit(self, conversation_id: str) -> list[AuditEvent]: ...
    def count_customers(self) -> int: ...


class InMemoryStore:
    profile = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.customers: dict[str, Customer] = {}
        self.accounts: dict[str, Account] = {}
        self.plans: dict[str, Plan] = {}
        self.offers: dict[str, Offer] = {}
        self.bills: dict[str, Bill] = {}  # by bill_id — an account has a history
        self.policies: dict[str, PolicyDocument] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.procedures: dict[str, Procedure] = {}
        self.procedure_runs: dict[str, ProcedureRun] = {}
        self.callbacks: dict[str, CallbackRecord] = {}
        self.roaming_rates: dict[str, RoamingRate] = {}
        self.roaming_notices: dict[str, RoamingNotice] = {}
        self.payment_attempts: dict[str, PaymentAttempt] = {}
        self.payment_arrangements: dict[str, PaymentArrangement] = {}
        self.payment_notices: dict[str, PaymentNotice] = {}
        self.complaints: dict[str, ComplaintRecord] = {}
        self.feedback: dict[str, FeedbackRecord] = {}
        self.conversations: dict[str, Conversation] = {}
        self.actions: dict[str, ActionRecord] = {}
        self.dispute_cases: dict[str, DisputeCase] = {}
        self.escalations: dict[str, Escalation] = {}
        self.attributions: dict[str, AttributionRecord] = {}
        self.annotations: dict[str, AnnotationRecord] = {}
        self.audit: list[AuditEvent] = []

    def _put(self, table: dict, key: str, model) -> None:
        with self._lock:
            table[key] = copy.deepcopy(model)

    def _get(self, table: dict, key: str):
        with self._lock:
            model = table.get(key)
            return copy.deepcopy(model) if model is not None else None

    def _all(self, table: dict) -> list:
        with self._lock:
            return [copy.deepcopy(m) for m in table.values()]

    def save_customer(self, c): self._put(self.customers, c.customer_id, c)
    def get_customer(self, customer_id): return self._get(self.customers, customer_id)

    def get_customer_by_account(self, account_number):
        for c in self._all(self.customers):
            if c.account_number == account_number:
                return c
        return None

    def save_account(self, a): self._put(self.accounts, a.account_number, a)
    def get_account(self, account_number): return self._get(self.accounts, account_number)
    def save_plan(self, p): self._put(self.plans, p.plan_id, p)
    def get_plan(self, plan_id): return self._get(self.plans, plan_id)
    def list_plans(self): return self._all(self.plans)
    def save_offer(self, o): self._put(self.offers, o.offer_id, o)
    def get_offer(self, offer_id): return self._get(self.offers, offer_id)
    def list_offers(self): return self._all(self.offers)
    def save_bill(self, b): self._put(self.bills, b.bill_id, b)

    def list_bills_for_account(self, account_number):
        rows = [b for b in self._all(self.bills)
                if b.account_number == account_number]
        return sorted(rows, key=lambda b: (b.issued, b.bill_id))

    def get_bill_for_account(self, account_number):
        rows = self.list_bills_for_account(account_number)
        return rows[-1] if rows else None

    def save_policy(self, d): self._put(self.policies, d.doc_id, d)
    def get_policy(self, doc_id): return self._get(self.policies, doc_id)
    def list_policies(self): return self._all(self.policies)

    def set_embedding(self, doc_id, vector):
        _check_dim(vector)
        with self._lock:
            self.embeddings[doc_id] = list(vector)

    def list_embeddings(self):
        with self._lock:
            return [(k, list(v)) for k, v in self.embeddings.items()]

    # Procedures are versioned data: rows key on id + version, the
    # plain read returns the latest version, and a run pins the
    # version it started on through get_procedure_version.
    def save_procedure(self, p):
        self._put(self.procedures, _procedure_key(p.procedure_id, p.version), p)

    def get_procedure(self, procedure_id):
        versions = [p for p in self._all(self.procedures)
                    if p.procedure_id == procedure_id]
        return max(versions, key=lambda p: p.version) if versions else None

    def get_procedure_version(self, procedure_id, version):
        return self._get(self.procedures, _procedure_key(procedure_id, version))

    def list_procedures(self):
        latest: dict[str, Procedure] = {}
        for p in self._all(self.procedures):
            current = latest.get(p.procedure_id)
            if current is None or p.version > current.version:
                latest[p.procedure_id] = p
        return [latest[k] for k in sorted(latest)]

    def save_procedure_run(self, r): self._put(self.procedure_runs, r.run_id, r)
    def get_procedure_run(self, run_id): return self._get(self.procedure_runs, run_id)

    def list_procedure_runs(self, conversation_id=None):
        rows = self._all(self.procedure_runs)
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: r.run_id)

    def save_callback(self, c): self._put(self.callbacks, c.callback_id, c)
    def get_callback(self, callback_id): return self._get(self.callbacks, callback_id)

    def list_callbacks(self, conversation_id=None):
        rows = self._all(self.callbacks)
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: r.callback_id)

    def save_roaming_rate(self, r):
        self._put(self.roaming_rates, r.destination.lower(), r)

    def get_roaming_rate(self, destination):
        return self._get(self.roaming_rates, destination.lower())

    def list_roaming_rates(self):
        return sorted(self._all(self.roaming_rates),
                      key=lambda r: r.destination)

    def save_roaming_notice(self, n):
        self._put(self.roaming_notices, n.notice_id, n)

    def list_roaming_notices(self, account_number=None):
        rows = self._all(self.roaming_notices)
        if account_number:
            rows = [n for n in rows if n.account_number == account_number]
        return sorted(rows, key=lambda n: (n.created_date, n.notice_id))

    def save_payment_attempt(self, t):
        self._put(self.payment_attempts, t.attempt_id, t)

    def list_payment_attempts(self, account_number=None, conversation_id=None):
        rows = self._all(self.payment_attempts)
        if account_number:
            rows = [t for t in rows if t.account_number == account_number]
        if conversation_id:
            rows = [t for t in rows if t.conversation_id == conversation_id]
        return sorted(rows, key=lambda t: t.created_date)

    def save_payment_arrangement(self, a):
        self._put(self.payment_arrangements, a.arrangement_id, a)

    def get_payment_arrangement(self, arrangement_id):
        return self._get(self.payment_arrangements, arrangement_id)

    def list_payment_arrangements(self, account_number=None):
        rows = self._all(self.payment_arrangements)
        if account_number:
            rows = [a for a in rows if a.account_number == account_number]
        return sorted(rows, key=lambda a: a.created_date)

    def save_payment_notice(self, n):
        self._put(self.payment_notices, n.notice_id, n)

    def list_payment_notices(self, account_number=None):
        rows = self._all(self.payment_notices)
        if account_number:
            rows = [n for n in rows if n.account_number == account_number]
        return sorted(rows, key=lambda n: (n.created_date, n.notice_id))

    def save_complaint(self, c): self._put(self.complaints, c.complaint_id, c)
    def get_complaint(self, complaint_id):
        return self._get(self.complaints, complaint_id)

    def list_complaints(self, conversation_id=None, account_number=None):
        rows = self._all(self.complaints)
        if conversation_id:
            rows = [c for c in rows if c.conversation_id == conversation_id]
        if account_number:
            rows = [c for c in rows if c.account_number == account_number]
        return sorted(rows, key=lambda c: (c.created_date, c.complaint_id))

    def save_feedback(self, f): self._put(self.feedback, f.record_id, f)

    def list_feedback(self, conversation_id=None, account_number=None):
        rows = self._all(self.feedback)
        if conversation_id:
            rows = [f for f in rows if f.conversation_id == conversation_id]
        if account_number:
            rows = [f for f in rows if f.account_number == account_number]
        return sorted(rows, key=lambda f: (f.created_date, f.record_id))

    def save_conversation(self, c): self._put(self.conversations, c.conversation_id, c)
    def get_conversation(self, cid): return self._get(self.conversations, cid)
    def list_conversations(self): return self._all(self.conversations)
    def save_action(self, a): self._put(self.actions, a.action_id, a)

    def list_actions(self, conversation_id=None):
        rows = self._all(self.actions)
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: (r.turn, r.action_id))

    def save_dispute_case(self, d): self._put(self.dispute_cases, d.case_id, d)
    def get_dispute_case(self, case_id): return self._get(self.dispute_cases, case_id)

    def list_dispute_cases(self, conversation_id=None, account_number=None):
        rows = self._all(self.dispute_cases)
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        if account_number:
            rows = [r for r in rows if r.account_number == account_number]
        return sorted(rows, key=lambda r: r.case_id)

    def save_escalation(self, e): self._put(self.escalations, e.escalation_id, e)
    def list_escalations(self): return self._all(self.escalations)

    def save_attribution(self, r):
        self._put(self.attributions, r.conversation_id, r)

    def get_attribution(self, conversation_id):
        return self._get(self.attributions, conversation_id)

    def list_attributions(self):
        return sorted(self._all(self.attributions),
                      key=lambda r: r.conversation_id)

    def save_annotation(self, a):
        self._put(self.annotations, a.annotation_id, a)

    def list_annotations(self, conversation_id=None):
        rows = self._all(self.annotations)
        if conversation_id:
            rows = [a for a in rows if a.conversation_id == conversation_id]
        return sorted(rows, key=lambda a: a.seq)

    def save_audit(self, e):
        with self._lock:
            self.audit.append(copy.deepcopy(e))

    def list_audit(self, conversation_id):
        with self._lock:
            return [copy.deepcopy(e) for e in self.audit
                    if e.conversation_id == conversation_id]

    def count_customers(self): return len(self.customers)


# ---------------------------------------------------------------------------
# SQL stores (SQLite + PostgreSQL share the payload-table shape)
# ---------------------------------------------------------------------------

# (table, key column, model) for every JSON-payload table.
_TABLES = {
    "customers": ("customer_id", Customer),
    "accounts": ("account_number", Account),
    "plans": ("plan_id", Plan),
    "offers": ("offer_id", Offer),
    "bills": ("bill_id", Bill),
    "policies": ("doc_id", PolicyDocument),
    "procedures": ("procedure_id", Procedure),
    "procedure_runs": ("run_id", ProcedureRun),
    "callbacks": ("callback_id", CallbackRecord),
    "roaming_rates": ("destination", RoamingRate),
    "roaming_notices": ("notice_id", RoamingNotice),
    "payment_attempts": ("attempt_id", PaymentAttempt),
    "payment_arrangements": ("arrangement_id", PaymentArrangement),
    "payment_notices": ("notice_id", PaymentNotice),
    "complaints": ("complaint_id", ComplaintRecord),
    "feedback_records": ("record_id", FeedbackRecord),
    "conversations": ("conversation_id", Conversation),
    "actions": ("action_id", ActionRecord),
    "dispute_cases": ("case_id", DisputeCase),
    "escalations": ("escalation_id", Escalation),
    "attributions": ("conversation_id", AttributionRecord),
    "annotations": ("annotation_id", AnnotationRecord),
}


def _procedure_key(procedure_id: str, version: int) -> str:
    """The row key for one version of a procedure. The procedures
    table's key column holds `id:vN`, so versions coexist and the
    latest is a read-time choice, never an overwrite."""
    return f"{procedure_id}:v{version}"


class _SqlStore:
    profile = "sql"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # -- driver hooks -------------------------------------------------
    def _execute(self, sql: str, params: tuple = ()) -> None:
        raise NotImplementedError

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        raise NotImplementedError

    # -- generic payload access --------------------------------------
    def _save(self, table: str, key: str, model) -> None:
        key_col, _ = _TABLES[table]
        self._execute(
            f"INSERT INTO {table} ({key_col}, payload) VALUES (?, ?) "
            f"ON CONFLICT ({key_col}) DO UPDATE SET payload = excluded.payload",
            (key, dump(model)),
        )

    def _load(self, table: str, key: str):
        _, model_cls = _TABLES[table]
        rows = self._query(
            f"SELECT payload FROM {table} WHERE {_TABLES[table][0]} = ?", (key,)
        )
        if not rows:
            return None
        return model_cls.model_validate_json(rows[0][0])

    def _load_all(self, table: str) -> list:
        _, model_cls = _TABLES[table]
        return [model_cls.model_validate_json(r[0])
                for r in self._query(f"SELECT payload FROM {table}")]

    # -- protocol ------------------------------------------------------
    def save_customer(self, c): self._save("customers", c.customer_id, c)
    def get_customer(self, customer_id): return self._load("customers", customer_id)

    def get_customer_by_account(self, account_number):
        for c in self._load_all("customers"):
            if c.account_number == account_number:
                return c
        return None

    def save_account(self, a): self._save("accounts", a.account_number, a)
    def get_account(self, account_number): return self._load("accounts", account_number)
    def save_plan(self, p): self._save("plans", p.plan_id, p)
    def get_plan(self, plan_id): return self._load("plans", plan_id)
    def list_plans(self): return self._load_all("plans")
    def save_offer(self, o): self._save("offers", o.offer_id, o)
    def get_offer(self, offer_id): return self._load("offers", offer_id)
    def list_offers(self): return self._load_all("offers")
    def save_bill(self, b): self._save("bills", b.bill_id, b)

    def list_bills_for_account(self, account_number):
        rows = [b for b in self._load_all("bills")
                if b.account_number == account_number]
        return sorted(rows, key=lambda b: (b.issued, b.bill_id))

    def get_bill_for_account(self, account_number):
        rows = self.list_bills_for_account(account_number)
        return rows[-1] if rows else None

    def save_policy(self, d): self._save("policies", d.doc_id, d)
    def get_policy(self, doc_id): return self._load("policies", doc_id)
    def list_policies(self): return self._load_all("policies")
    def save_procedure(self, p):
        self._save("procedures", _procedure_key(p.procedure_id, p.version), p)

    def get_procedure(self, procedure_id):
        versions = [p for p in self._load_all("procedures")
                    if p.procedure_id == procedure_id]
        return max(versions, key=lambda p: p.version) if versions else None

    def get_procedure_version(self, procedure_id, version):
        return self._load("procedures", _procedure_key(procedure_id, version))

    def list_procedures(self):
        latest: dict[str, Procedure] = {}
        for p in self._load_all("procedures"):
            current = latest.get(p.procedure_id)
            if current is None or p.version > current.version:
                latest[p.procedure_id] = p
        return [latest[k] for k in sorted(latest)]

    def save_procedure_run(self, r): self._save("procedure_runs", r.run_id, r)
    def get_procedure_run(self, run_id): return self._load("procedure_runs", run_id)

    def list_procedure_runs(self, conversation_id=None):
        rows = self._load_all("procedure_runs")
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: r.run_id)

    def save_callback(self, c): self._save("callbacks", c.callback_id, c)
    def get_callback(self, callback_id): return self._load("callbacks", callback_id)

    def list_callbacks(self, conversation_id=None):
        rows = self._load_all("callbacks")
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: r.callback_id)

    # Roaming rates key on the destination, lower-cased like the
    # in-memory store, so a lookup never misses on capitalisation.
    def save_roaming_rate(self, r):
        self._save("roaming_rates", r.destination.lower(), r)

    def get_roaming_rate(self, destination):
        return self._load("roaming_rates", destination.lower())

    def list_roaming_rates(self):
        return sorted(self._load_all("roaming_rates"),
                      key=lambda r: r.destination)

    def save_roaming_notice(self, n): self._save("roaming_notices", n.notice_id, n)

    def list_roaming_notices(self, account_number=None):
        rows = self._load_all("roaming_notices")
        if account_number:
            rows = [n for n in rows if n.account_number == account_number]
        return sorted(rows, key=lambda n: (n.created_date, n.notice_id))

    def save_payment_attempt(self, t):
        self._save("payment_attempts", t.attempt_id, t)

    def list_payment_attempts(self, account_number=None, conversation_id=None):
        rows = self._load_all("payment_attempts")
        if account_number:
            rows = [t for t in rows if t.account_number == account_number]
        if conversation_id:
            rows = [t for t in rows if t.conversation_id == conversation_id]
        return sorted(rows, key=lambda t: t.created_date)

    def save_payment_arrangement(self, a):
        self._save("payment_arrangements", a.arrangement_id, a)

    def get_payment_arrangement(self, arrangement_id):
        return self._load("payment_arrangements", arrangement_id)

    def list_payment_arrangements(self, account_number=None):
        rows = self._load_all("payment_arrangements")
        if account_number:
            rows = [a for a in rows if a.account_number == account_number]
        return sorted(rows, key=lambda a: a.created_date)

    def save_payment_notice(self, n):
        self._save("payment_notices", n.notice_id, n)

    def list_payment_notices(self, account_number=None):
        rows = self._load_all("payment_notices")
        if account_number:
            rows = [n for n in rows if n.account_number == account_number]
        return sorted(rows, key=lambda n: (n.created_date, n.notice_id))

    def save_complaint(self, c): self._save("complaints", c.complaint_id, c)
    def get_complaint(self, complaint_id):
        return self._load("complaints", complaint_id)

    def list_complaints(self, conversation_id=None, account_number=None):
        rows = self._load_all("complaints")
        if conversation_id:
            rows = [c for c in rows if c.conversation_id == conversation_id]
        if account_number:
            rows = [c for c in rows if c.account_number == account_number]
        return sorted(rows, key=lambda c: (c.created_date, c.complaint_id))

    def save_feedback(self, f):
        self._save("feedback_records", f.record_id, f)

    def list_feedback(self, conversation_id=None, account_number=None):
        rows = self._load_all("feedback_records")
        if conversation_id:
            rows = [f for f in rows if f.conversation_id == conversation_id]
        if account_number:
            rows = [f for f in rows if f.account_number == account_number]
        return sorted(rows, key=lambda f: (f.created_date, f.record_id))

    def save_conversation(self, c): self._save("conversations", c.conversation_id, c)
    def get_conversation(self, cid): return self._load("conversations", cid)
    def list_conversations(self): return self._load_all("conversations")
    def save_action(self, a): self._save("actions", a.action_id, a)

    def list_actions(self, conversation_id=None):
        rows = self._load_all("actions")
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        return sorted(rows, key=lambda r: (r.turn, r.action_id))

    def save_dispute_case(self, d): self._save("dispute_cases", d.case_id, d)
    def get_dispute_case(self, case_id): return self._load("dispute_cases", case_id)

    def list_dispute_cases(self, conversation_id=None, account_number=None):
        rows = self._load_all("dispute_cases")
        if conversation_id:
            rows = [r for r in rows if r.conversation_id == conversation_id]
        if account_number:
            rows = [r for r in rows if r.account_number == account_number]
        return sorted(rows, key=lambda r: r.case_id)

    def save_escalation(self, e): self._save("escalations", e.escalation_id, e)
    def list_escalations(self): return self._load_all("escalations")

    def save_attribution(self, r):
        self._save("attributions", r.conversation_id, r)

    def get_attribution(self, conversation_id):
        return self._load("attributions", conversation_id)

    def list_attributions(self):
        return sorted(self._load_all("attributions"),
                      key=lambda r: r.conversation_id)

    def save_annotation(self, a):
        self._save("annotations", a.annotation_id, a)

    def list_annotations(self, conversation_id=None):
        rows = self._load_all("annotations")
        if conversation_id:
            rows = [a for a in rows if a.conversation_id == conversation_id]
        return sorted(rows, key=lambda a: a.seq)

    def save_audit(self, e):
        self._execute(
            "INSERT INTO audit_events (event_id, conversation_id, payload) VALUES (?, ?, ?)",
            (e.event_id, e.conversation_id, dump(e)),
        )

    def list_audit(self, conversation_id):
        rows = self._query(
            "SELECT payload FROM audit_events WHERE conversation_id = ?", (conversation_id,)
        )
        return [AuditEvent.model_validate_json(r[0]) for r in rows]

    def count_customers(self):
        return self._query("SELECT COUNT(*) FROM customers")[0][0]

    def set_embedding(self, doc_id, vector):
        _check_dim(vector)
        self._execute(
            "INSERT INTO policy_embeddings (doc_id, embedding) VALUES (?, ?) "
            "ON CONFLICT (doc_id) DO UPDATE SET embedding = excluded.embedding",
            (doc_id, json.dumps(vector)),
        )

    def list_embeddings(self):
        return [(r[0], json.loads(r[1])) for r in
                self._query("SELECT doc_id, embedding FROM policy_embeddings")]


def _run_migrations(execute_script, dialect: str) -> list[str]:
    """Apply numbered migrations once each, tracked in schema_migrations."""
    execute_script(
        "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY)"
    )
    applied = []
    folder = _migrations_dir(dialect)
    for path in sorted(folder.glob("*.sql")):
        rows = _applied_names(execute_script, path.name)
        if rows:
            continue
        execute_script(path.read_text(), record=path.name)
        applied.append(path.name)
    return applied


def _applied_names(execute_script, name: str) -> list:
    return execute_script("SELECT name FROM schema_migrations WHERE name = ?", (name,))


class SqliteStore(_SqlStore):
    profile = "sqlite"

    def __init__(self, path: str | Path = DEFAULT_SQLITE_PATH) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _execute(self, sql, params=()):
        with self._lock:
            self._conn.execute(sql.replace("%s", "?"), params)
            self._conn.commit()

    def _query(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql.replace("%s", "?"), params)
            return cur.fetchall()

    def _script(self, sql, params=(), record=None):
        """Migration runner hook: executes a full script, or a query
        when params are given; returns rows for the applied-check."""
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchall()
            self._conn.executescript(sql)
            if record:
                self._conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES (?)", (record,))
            self._conn.commit()
            return []

    def _migrate(self):
        _run_migrations(self._script, "sqlite")

    def set_embedding(self, doc_id, vector):
        _check_dim(vector)
        self._execute(
            "INSERT INTO policy_embeddings (doc_id, embedding) VALUES (?, ?) "
            "ON CONFLICT (doc_id) DO UPDATE SET embedding = excluded.embedding",
            (doc_id, json.dumps(vector)),
        )

    def list_embeddings(self):
        return [(r[0], json.loads(r[1])) for r in
                self._query("SELECT doc_id, embedding FROM policy_embeddings")]


class PostgresStore(_SqlStore):
    profile = "postgres"

    def __init__(self, dsn: str) -> None:
        super().__init__()
        import psycopg
        from psycopg.rows import tuple_row

        self._psycopg = psycopg
        self._conn = psycopg.connect(dsn, row_factory=tuple_row, autocommit=True)
        self._migrate()

    def _execute(self, sql, params=()):
        sql = sql.replace("?", "%s")
        # SQLite-flavoured upsert tail -> Postgres spelling.
        sql = sql.replace("excluded.payload", "EXCLUDED.payload")
        with self._lock:
            self._conn.execute(sql, params)

    def _query(self, sql, params=()):
        sql = sql.replace("?", "%s")
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _script(self, sql, params=(), record=None):
        with self._lock:
            if params:
                return self._conn.execute(sql, params).fetchall()
            self._conn.execute(sql)
            if record:
                self._conn.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s)", (record,))
            return []

    def _migrate(self):
        _run_migrations(self._script, "")

    def set_embedding(self, doc_id, vector):
        _check_dim(vector)
        # No pgvector Python adapter is bundled with psycopg, so the
        # vector crosses the wire as its text literal and Postgres
        # casts it into the vector(384) column; reads come back the
        # same way and are parsed below.
        literal = "[" + ",".join(repr(float(v)) for v in vector) + "]"
        with self._lock:
            self._conn.execute(
                "INSERT INTO policy_embeddings (doc_id, embedding) VALUES (%s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                (doc_id, literal),
            )

    def list_embeddings(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, embedding FROM policy_embeddings").fetchall()
        out = []
        for doc_id, raw in rows:
            if isinstance(raw, str):
                vec = [float(x) for x in raw.strip("[]").split(",") if x]
            else:  # an adapter returned the vector already parsed
                vec = [float(x) for x in raw]
            out.append((doc_id, vec))
        return out


def make_store() -> Store:
    kind, target = store_profile()
    if kind == "postgres":
        return PostgresStore(target)
    return SqliteStore(target)


@contextlib.contextmanager
def fresh_memory_store():
    """Tests/evals helper: an empty in-memory store."""
    yield InMemoryStore()
