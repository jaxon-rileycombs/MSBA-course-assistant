"""Hybrid retrieval: combine keyword + text-embedding + visual-embedding
candidates and rerank, keeping text and visual indexes separate in storage.

Fusion = Reciprocal Rank Fusion (RRF). Scores from different retrievers
live on incomparable scales (BM25 raw scores vs cosine vs the class
reranker), and min-max normalization lets one noisy retriever flip a
near-tie; RRF fuses RANKS (``weight / (K + rank)``), which is scale-immune
and much less sensitive to tie noise. The reranker (off by default here,
wired in online mode) then reorders the fused list with its own model score.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .evidence import Evidence

# reranker: query, ordered texts -> scores in the same order (higher = better)
Reranker = Callable[[str, list[str]], list[float]]

RRF_K = 60  # standard RRF constant (dampens rank differences)


class HybridRetriever:
    def __init__(
        self,
        keyword,
        text_embedding,
        visual_embedding,
        *,
        weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
        rrf_k: float = RRF_K,
    ):
        self.keyword = keyword
        self.text_embedding = text_embedding
        self.visual_embedding = visual_embedding
        self.weights = weights or {"keyword": 1.0, "text": 1.0, "visual": 1.0}
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 8, per_retriever: int = 10) -> list[Evidence]:
        sources = {
            "keyword": self.keyword.top_k(query, per_retriever),
            "text": self.text_embedding.top_k(query, per_retriever),
            "visual": self.visual_embedding.top_k(query, per_retriever),
        }
        merged: dict[tuple[str, int], Evidence] = {}
        agg: dict[tuple[str, int], float] = {}
        for name, pairs in sources.items():
            # a page may yield several chunks; a retriever contributes that
            # page ONCE, at its best rank (otherwise long pages double-count)
            best_rank: dict[tuple[str, int], int] = {}
            best_pairs: dict[tuple[str, int], tuple[Evidence, float]] = {}
            for rank, (ev, score) in enumerate(pairs):
                if ev.key not in best_rank:
                    best_rank[ev.key] = rank
                    best_pairs[ev.key] = (ev, score)
            for key, rank in best_rank.items():
                ev, score = best_pairs[key]
                contribution = self.weights.get(name, 1.0) / (self.rrf_k + rank + 1)
                agg[key] = agg.get(key, 0.0) + contribution
                if key not in merged:
                    merged[key] = ev
                else:
                    # merge provenance: visual hit adds the image to the text hit
                    if ev.image_path and not merged[key].image_path:
                        merged[key].image_path = ev.image_path
                    if ev.text and not merged[key].text:
                        merged[key].text = ev.text
                merged[key].scores.setdefault(name, score)

        ranked = sorted(merged.values(), key=lambda e: agg[e.key], reverse=True)

        if self.reranker is not None:
            texts = [e.text for e in ranked]
            try:
                scores = self.reranker(query, texts)
                if len(scores) == len(ranked):
                    ranked = [e for _, e in sorted(
                        zip(scores, ranked), key=lambda p: p[0], reverse=True
                    )]
            except Exception:
                pass  # rerank failure must not kill retrieval; keep fused order

        return ranked[:k]
