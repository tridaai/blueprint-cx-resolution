-- Known-issue awareness (round 15): the operator incident feed
-- as governed records — one row per incident (its service, its
-- affected postcodes, its declared symptom signals, its status,
-- and the operator's customer-safe summary the agent serves
-- verbatim), plus the append-only log of the operator acts that
-- added, updated or resolved them. Payload-shaped like every
-- other record table.

CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS incident_acts (
    act_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
