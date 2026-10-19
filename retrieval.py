"""Hybrid retrieval: dense vector search + BM25 keyword search, merged with Reciprocal Rank Fusion."""
from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi

from talkai.embeddings import tokenize
from talkai.store import Hit, VectorStore

RRF_K = 60  # standard constant from the original RRF paper


def bm25_search(store: VectorStore, query: str, k: int, doc_ids: Optional[list[str]] = None) -> list[Hit]:
    chunks = store.all_chunks(doc_ids)
    q = tokenize(query)
    if not chunks or not q:
        return []
    bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])
    scores = bm25.get_scores(q)
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
    return [c for c, s in ranked[:k] if s > 0]


def hybrid_search(store: VectorStore, query: str, k: int = 5, doc_ids: Optional[list[str]] = None,
                  pool: int = 20) -> list[Hit]:
    """Merge dense and keyword rankings. Dense search finds meaning; BM25 finds exact terms like 'Kubernetes'."""
    lists = [store.vector_search(query, pool, doc_ids), bm25_search(store, query, pool, doc_ids)]
    fused: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for ranking in lists:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id[hit.chunk_id] = hit
    best_possible = len(lists) / (RRF_K + 1)  # rank 1 in every list
    out = []
    for cid, score in sorted(fused.items(), key=lambda x: -x[1])[:k]:
        hit = by_id[cid]
        hit.score = round(score / best_possible, 3)
        out.append(hit)
    return out
