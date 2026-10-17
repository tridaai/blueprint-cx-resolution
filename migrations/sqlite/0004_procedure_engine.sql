-- Troubleshooting engine (round 4): a procedure walk is a record
-- of its own — the run, with its step history — and a dead-ended
-- procedure books its callback as a record carrying the full step
-- transcript, the way a dispute is a case. Procedures themselves
-- stay in the existing table; their rows now key on id + version.

CREATE TABLE IF NOT EXISTS procedure_runs (
    run_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS callbacks (
    callback_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
