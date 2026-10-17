-- Daily quality digests (round 12): one stored snapshot per
-- date — the daily pass's composed view of that day's closed
-- conversations, its signal aggregates, the complaint book at
-- day end, the mining candidates, and the spot-check verdict.
-- Re-running the pass for a date replaces the row; the version
-- on the payload counts the replacement.

CREATE TABLE IF NOT EXISTS quality_digests (
    digest_date TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
