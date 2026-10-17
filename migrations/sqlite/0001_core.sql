-- Core records for the CX Resolution Agent — SQLite dialect of
-- migrations/0001_core.sql. Same tables and payload shapes; payloads
-- are JSON text and money rides inside them as decimal strings.
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    payload     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_number TEXT PRIMARY KEY,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS offers (
    offer_id TEXT PRIMARY KEY,
    payload  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bills (
    account_number TEXT PRIMARY KEY,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    doc_id  TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS procedures (
    procedure_id TEXT PRIMARY KEY,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    payload         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    action_id TEXT PRIMARY KEY,
    payload   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id TEXT PRIMARY KEY,
    payload       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    payload         TEXT NOT NULL
);
