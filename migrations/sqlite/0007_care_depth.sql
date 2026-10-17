-- Customer care depth (round 8): complaints as records with their
-- SLA response deadlines (the watch reads them), and feedback /
-- compliment records — the care desk's data, not chat filler.

CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback_records (
    record_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
