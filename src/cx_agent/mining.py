"""Unhandled-intent mining: the report that designs the next
desk.

Some conversations end without the agent understanding what was
asked — the care desk logged it as unhandled, the router fell
back to care on low confidence, or the conversation escalated
with no progress. Those endings are the raw material for new
intents, and this pass mines them: it gathers one utterance per
such conversation (the ask itself), clusters the utterances with
the repo's own embedder, and reports the clusters big enough to
be candidate intents — a label drawn from the cluster's own most
characteristic phrase, the count, and verbatim samples.

The pass is deterministic end to end: conversations are read in
id order, clustering is greedy in that order against the first
member's embedding, ties break the same way on every run. It is
also strictly a report — nothing here creates an intent, changes
a route, or writes anything back. A person designs the new
intent; the system's job is to make sure they can see the need.
"""

from __future__ import annotations

from collections import Counter

from .documents import _tokens, cosine, embed

# Where an unhandled utterance came from, in the order the
# sources are read for one conversation.
SOURCES = ("unhandled_intent", "low_confidence", "no_progress")


def _utterance_for(store, conv, no_progress_ids: set) -> tuple[str, str] | None:
    """The ask that went unhandled in one conversation, with its
    source — or None when the conversation was handled. One
    utterance per conversation, the first unhandled signal that
    conversation produced: the care desk's own log, else the
    first turn the router sent to care on low confidence, else
    the last ask before a no-progress escalation."""
    if conv.unhandled_intent:
        return conv.unhandled_intent, "unhandled_intent"
    audit = store.list_audit(conv.conversation_id)
    customer_turns = {m.turn: m.text for m in conv.messages
                      if m.role == "customer"}
    for event in audit:
        routing = (event.detail or {}).get("routing") or {}
        if event.kind == "turn" \
                and routing.get("decision") == "low_confidence_care":
            text = customer_turns.get(event.turn)
            if text:
                return text[:120], "low_confidence"
    if conv.conversation_id in no_progress_ids and customer_turns:
        return customer_turns[max(customer_turns)][:120], "no_progress"
    return None


def _label(texts: list[str]) -> str:
    """A cluster's most characteristic phrase: the most common
    adjacent word pair across its utterances (ties go to the pair
    from the earliest utterance). Falls back to the most common
    single word, then to the first utterance itself."""
    pair_counts: Counter = Counter()
    pair_first: dict[tuple[str, str], int] = {}
    word_counts: Counter = Counter()
    for idx, text in enumerate(texts):
        tokens = _tokens(text)
        word_counts.update(tokens)
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += 1
            pair_first.setdefault(pair, idx)
    if pair_counts:
        best = min(pair_counts,
                   key=lambda p: (-pair_counts[p], pair_first[p], p))
        return " ".join(best)
    if word_counts:
        return word_counts.most_common(1)[0][0]
    return texts[0][:40] if texts else ""


def mine_unhandled_intents(store, policy) -> dict:
    """Run the mining pass over every stored conversation and
    return the report: candidate clusters (at or above the policy
    threshold) first-class, smaller clusters counted beneath it,
    and the totals a reader needs to trust the numbers."""
    no_progress_ids = {e.conversation_id for e in store.list_escalations()
                       if e.reason == "no_progress"}
    gathered: list[tuple[str, str, str]] = []  # (utterance, source, conv id)
    for conv in sorted(store.list_conversations(),
                       key=lambda c: c.conversation_id):
        found = _utterance_for(store, conv, no_progress_ids)
        if found:
            gathered.append((found[0], found[1], conv.conversation_id))

    clusters: list[dict] = []
    for text, source, conv_id in gathered:
        vector = embed(text)
        best, best_score = None, -1.0
        for cluster in clusters:
            score = cosine(vector, cluster["_vector"])
            if score > best_score:
                best, best_score = cluster, score
        if best is not None and best_score >= policy.mining_similarity:
            best["members"].append((text, source, conv_id))
        else:
            clusters.append({"_vector": vector,
                             "members": [(text, source, conv_id)]})

    report_clusters = []
    for cluster in clusters:
        texts = [m[0] for m in cluster["members"]]
        sources = Counter(m[1] for m in cluster["members"])
        report_clusters.append({
            "label": _label(texts),
            "count": len(texts),
            "candidate": len(texts) >= policy.mining_min_cluster,
            "sources": {s: sources.get(s, 0) for s in SOURCES
                        if sources.get(s)},
            "samples": texts[:3],
        })
    report_clusters.sort(key=lambda c: (-c["count"], c["label"]))
    candidates = [c for c in report_clusters if c["candidate"]]
    return {
        "conversations_scanned": len(store.list_conversations()),
        "unhandled_utterances": len(gathered),
        "min_cluster": policy.mining_min_cluster,
        "similarity": policy.mining_similarity,
        "candidates": candidates,
        "clusters": report_clusters,
    }
