"""Grounding retrieval over the policy corpus.

The supervisor's reply may assert policy only with a citation, and a
citation means: a document this retriever returned for this turn.
Ranking is cosine over the stored embeddings — pgvector holds them
in production, JSON in SQLite, but the embedder and the cosine are
the same code on every profile, so rankings (and eval results) do
not drift between them.
"""

from __future__ import annotations

from .documents import cosine, embed

# Below this similarity a document is not evidence; the honest
# answer is "I can't confirm that", not a stretch.
MIN_SIMILARITY = 0.18


class PolicyRetriever:
    def __init__(self, store) -> None:
        self.store = store

    def search(self, query: str, limit: int = 3,
               section: str | None = None) -> list[dict]:
        """Top policy documents for a query, each as
        {doc_id, title, text, score}. Empty when nothing clears the
        floor — the caller treats that as 'no policy basis'.

        `section` scopes the search to one corpus section (billing,
        roaming, plans, ...): a specialist citing policy searches its
        own desk's shelf, the way a human agent would, so a billing
        rule is never 'grounded' in a roaming document."""
        q = embed(query)
        scored = []
        for doc_id, vector in self.store.list_embeddings():
            if section is not None:
                doc = self.store.get_policy(doc_id)
                if doc is None or doc.section != section:
                    continue
            score = cosine(q, vector)
            if score >= MIN_SIMILARITY:
                scored.append((score, doc_id))
        scored.sort(reverse=True)
        out = []
        for score, doc_id in scored[:limit]:
            doc = self.store.get_policy(doc_id)
            if doc is not None:
                out.append({"doc_id": doc.doc_id, "title": doc.title,
                            "text": doc.text, "score": round(score, 4)})
        return out
