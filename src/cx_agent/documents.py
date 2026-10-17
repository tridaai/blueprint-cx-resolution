"""Policy-document embeddings.

One contract on every store: EMBEDDING_DIM dimensions. The offline
embedder is a deterministic hashing embedder — token hashes folded
into the vector with a sign, L2-normalised — so retrieval behaves
identically in tests, evals and production-shaped runs with no
model call. A configured model embedder can replace it behind the
same two functions (`embed`, `cosine`) without touching retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re

EMBEDDING_DIM = 384

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words carry no topic; left in, they let an unrelated
# question ("what is the weather") collide with any document that
# happens to use "the" a lot — which is all of them.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "i", "me", "my", "you", "your", "we", "it", "this", "that", "to",
    "of", "in", "on", "for", "and", "or", "if", "so", "at", "by",
    "with", "from", "can", "could", "will", "would", "should", "what",
    "how", "when", "why", "there", "here", "be", "been", "have", "has",
}


def _stem(token: str) -> str:
    """Crude suffix stripping so 'charges', 'charged' and 'charging'
    land in the same bucket. Applied identically to documents and
    queries, which is all a hashing embedder needs."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(tok) for tok in _TOKEN.findall(text.lower())
            if tok not in _STOPWORDS]


def embed(text: str) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    for tok in _tokens(text):
        digest = hashlib.sha256(tok.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
        # A second fold of bigrams' first token keeps short queries
        # from collapsing onto single words quite so hard.
        bucket2 = int.from_bytes(digest[5:9], "little") % EMBEDDING_DIM
        vec[bucket2] += 0.5 * sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))
