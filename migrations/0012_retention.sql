-- Retention cases (round 14): a customer's stated intent to
-- leave as a record of its own — the reason classified against a
-- governed list with the customer's words verbatim, the one
-- save offer the policy allows, and the outcome the case
-- settled into (saved, cancelled at the cycle boundary, or
-- escalated to humans). Payload-shaped like every other case
-- table; the cancellation schedule itself rides the account
-- record (pending_cancellation_effective, in its payload).

CREATE TABLE IF NOT EXISTS retention_cases (
    case_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
