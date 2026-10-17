-- Payments depth (round 7): payment attempts (the decline log the
-- retry rules read), payment arrangements as records with their
-- dated installments, and payment notices — the failed-autopay
-- artifact, sibling of the roaming fair-use notices.

CREATE TABLE IF NOT EXISTS payment_attempts (
    attempt_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_arrangements (
    arrangement_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_notices (
    notice_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
