"""Reranker adapter: the class multimodal reranker (9004) as a callable the
hybrid retriever can use. Handles service batching and maps the service's
(index, score) results back to the original order."""

from __future__ import annotations


class ServiceReranker:
    def __init__(self, client, batch_size: int = 32):
        self.client = client
        self.batch_size = batch_size

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        """Return one score per text (original order; higher = better)."""
        scores = [0.0] * len(texts)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for idx, score in self.client.rerank(query, batch):
                scores[i + idx] = score
        return scores
