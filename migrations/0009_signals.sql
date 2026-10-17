-- Conversation signals (round 11): one record per customer turn —
-- frustration score and trend, the effort score and its counted
-- components, progress, confusion — with the interventions the
-- signal rules fired on that turn.

CREATE TABLE IF NOT EXISTS signals (
    record_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
