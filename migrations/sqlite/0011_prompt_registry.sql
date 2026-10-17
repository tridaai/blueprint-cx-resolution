-- Prompt registry (round 13): every governed text — the model's
-- compose and intent instructions, the second explanation forms,
-- the policy corpus documents — as versioned entries. Rows key
-- on `prompt_id` holding `id:vN`, so versions coexist and the
-- active one (highest, unless an operator pin stands in
-- prompt_pins) is a read-time choice. prompt_acts is the
-- append-only log of the operator acts on the registry: versions
-- added, rollbacks (pins), unpins.

CREATE TABLE IF NOT EXISTS prompt_entries (
    prompt_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_pins (
    prompt_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_acts (
    act_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
