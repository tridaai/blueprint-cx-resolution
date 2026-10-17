-- Policy corpus embeddings — SQLite dialect of
-- migrations/0002_policy_embeddings.sql. SQLite has no pgvector, so
-- the vector is stored as JSON text and similarity is computed in
-- code (cx_agent/retrieval.py) over the stored vectors — the
-- results are identical: same embedder, same cosine, same ranking.
-- Only the machinery differs.
CREATE TABLE IF NOT EXISTS policy_embeddings (
    doc_id    TEXT PRIMARY KEY,
    embedding TEXT NOT NULL
);
