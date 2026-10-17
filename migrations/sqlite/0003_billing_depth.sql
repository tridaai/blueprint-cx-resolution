-- Billing depth (round 3): bills become a history — an account has
-- more than one bill, and explanations compare against the previous
-- one — so the bills table re-keys from account_number to bill_id.
-- Dispute cases get their own table: a dispute is a record, not a
-- by-product of a conversation.

CREATE TABLE IF NOT EXISTS bills_by_id (
    bill_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

INSERT OR IGNORE INTO bills_by_id (bill_id, payload)
SELECT json_extract(payload, '$.bill_id'), payload FROM bills;

DROP TABLE bills;
ALTER TABLE bills_by_id RENAME TO bills;

CREATE TABLE IF NOT EXISTS dispute_cases (
    case_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
