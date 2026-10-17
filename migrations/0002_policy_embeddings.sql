-- Policy corpus embeddings (pgvector). One embedding per policy
-- document, so the supervisor's grounding retrieval ranks the corpus
-- the same way in every profile. The offline embedder is a
-- deterministic hashing embedder (cx_agent/documents.py); a
-- configured model embedder can replace it behind the same port.
CREATE TABLE IF NOT EXISTS policy_embeddings (
    doc_id    TEXT PRIMARY KEY,
    embedding vector(384) NOT NULL
);
