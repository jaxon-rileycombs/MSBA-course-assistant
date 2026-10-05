"""Hybrid retrieval: combine keyword + text-embedding + visual-embedding
candidates, score-normalize, weight, merge, optionally rerank.

Combining (assignment): keep text and visual indexes separate in storage,
but combine their search results at query time and rerank the candidates so
the best evidence — including relevant images — is supplied to the answer
step alongside the question.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .evidence import Evidence

# reranker: query, ordered texts -> scores in the same order (higher = better)
Reranker = Callable[[str, list[str]], list[float]]


def _normalize(pairs: Sequence[tuple[Evidence, float]]) -> list[tuple[Evidence, float]]:
    """Min-max normalize scores to [0, 1]; empty list -> []."""
    if not pairs:
        return []
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [(e, 1.0) for e, _ in pairs]
    return [(e, (s - lo) / (hi - lo)) for e, s in pairs]


class HybridRetriever:
    def __init__(
        self,
        keyword,
        text_embedding,
        visual_embedding,
        *,
        weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
    ):
        self.keyword = keyword
        self.text_embedding = text_embedding
        self.visual_embedding = visual_embedding
        self.weights = weights or {"keyword": 1.0, "text": 1.0, "visual": 1.0}
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 8, per_retriever: int = 10) -> list[Evidence]:
        sources = {
            "keyword": self.keyword.top_k(query, per_retriever),
            "text": self.text_embedding.top_k(query, per_retriever),
            "visual": self.visual_embedding.top_k(query, per_retriever),
        }
        merged: dict[tuple[str, int], Evidence] = {}
        agg: dict[tuple[str, int], float] = {}
        for name, pairs in sources.items():
            for ev, score in _normalize(pairs):
                weight = self.weights.get(name, 1.0)
                agg[ev.key] = agg.get(ev.key, 0.0) + weight * score
                if ev.key not in merged:
                    merged[ev.key] = ev
                else:
                    # merge provenance: visual hit adds the image to the text hit
                    if ev.image_path and not merged[ev.key].image_path:
                        merged[ev.key].image_path = ev.image_path
                    if ev.text and not merged[ev.key].text:
                        merged[ev.key].text = ev.text
                merged[ev.key].scores[name] = max(
                    merged[ev.key].scores.get(name, 0.0), score
                )

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
