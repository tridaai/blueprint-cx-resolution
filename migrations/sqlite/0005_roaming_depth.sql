-- Roaming depth (round 6): the destination rate table the trip
-- quotes are priced from, and the fair-use notices — the
-- proactive warnings as stored records, which a later roaming
-- dispute reads back as evidence.

CREATE TABLE IF NOT EXISTS roaming_rates (
    destination TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS roaming_notices (
    notice_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
