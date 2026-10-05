"""Embedding retrievers over the ChromaDB collections.

- :class:`TextEmbeddingRetriever` searches ``text_chunks`` with a text query
  embedding (Nemotron on 9002 for the real service).
- :class:`VisualEmbeddingRetriever` searches ``visual_pages``. The query is
  embedded by the SAME multimodal model that embedded the images (Qwen3-VL
  on 9003), because a multimodal embedding model puts text and images in one
  space — a text embedding from a different model would be incomparable.
  This choice is documented in the plan; validated against the class
  service's docs once keys are available.
"""

from __future__ import annotations

from typing import Callable

from .evidence import Evidence

QueryEmbedder = Callable[[str], list[float]]


class _BaseEmbeddingRetriever:
    def __init__(self, collection, query_embedder: QueryEmbedder):
        self._col = collection
        self._embed = query_embedder

    def _rows(self, query: str, k: int) -> list[tuple[dict, str, float]]:
        """Return (metadata, document_text, similarity) rows, best first."""
        qv = self._embed(query)
        res = self._col.query(
            query_embeddings=[qv],
            n_results=min(k, max(1, self._col.count())),
            include=["metadatas", "documents", "distances"],
        )
        out = []
        for meta, doc_text, dist in zip(
            res["metadatas"][0], res["documents"][0], res["distances"][0]
        ):
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            out.append((meta, doc_text, sim))
        return out


class TextEmbeddingRetriever(_BaseEmbeddingRetriever):
    def __init__(self, index, query_embedder: QueryEmbedder):
        super().__init__(index.text_col, query_embedder)

    def top_k(self, query: str, k: int = 8) -> list[tuple[Evidence, float]]:
        return [
            (
                Evidence(
                    doc_id=m["doc_id"],
                    document=m["document"],
                    page=m["page"],
                    section=m.get("section", ""),
                    text=doc_text,
                ),
                sim,
            )
            for m, doc_text, sim in self._rows(query, k)
        ]


class VisualEmbeddingRetriever(_BaseEmbeddingRetriever):
    def __init__(self, index, query_embedder: QueryEmbedder):
        super().__init__(index.visual_col, query_embedder)

    def top_k(self, query: str, k: int = 8) -> list[tuple[Evidence, float]]:
        return [
            (
                Evidence(
                    doc_id=m["doc_id"],
                    document=m["document"],
                    page=m["page"],
                    section=m.get("section", ""),
                    text=doc_text,
                    image_path=m.get("image"),
                ),
                sim,
            )
            for m, doc_text, sim in self._rows(query, k)
        ]
