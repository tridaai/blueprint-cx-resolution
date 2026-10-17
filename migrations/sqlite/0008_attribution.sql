-- Failure attribution + annotation queue (round 10): why each
-- closed conversation ended the way it did, computed from its
-- audit trail, and the human annotations that review those calls.

CREATE TABLE IF NOT EXISTS attributions (
    conversation_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
