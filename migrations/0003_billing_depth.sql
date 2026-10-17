-- Billing depth (round 3): bills become a history — an account has
-- more than one bill, and explanations compare against the previous
-- one — so the bills table re-keys from account_number to bill_id.
-- Dispute cases get their own table: a dispute is a record, not a
-- by-product of a conversation.

CREATE TABLE IF NOT EXISTS bills_by_id (
    bill_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);

INSERT INTO bills_by_id (bill_id, payload)
SELECT payload ->> 'bill_id', payload FROM bills
ON CONFLICT (bill_id) DO NOTHING;

DROP TABLE bills;
ALTER TABLE bills_by_id RENAME TO bills;

CREATE TABLE IF NOT EXISTS dispute_cases (
    case_id TEXT PRIMARY KEY,
    payload JSONB NOT NULL
);
