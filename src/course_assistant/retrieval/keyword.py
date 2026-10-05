"""Keyword retriever (BM25 via bm25s), built from the persisted index.

The chunk texts live in ChromaDB's ``text_chunks`` collection, so the keyword
index is rebuilt from there — one source of truth, no second corpus file.
"""

from __future__ import annotations

from typing import Callable, Sequence

import bm25s

from .evidence import Evidence

_STEMMER = None


def _get_stemmer():
    """PyStemmer (Snowball English) when available; None → exact tokens.
    Keeps plural/verb forms matching ("grades" ↔ "graded" ↔ "grading")."""
    global _STEMMER
    if _STEMMER is None:
        try:
            import Stemmer

            _STEMMER = Stemmer.Stemmer("english")
        except Exception:  # pragma: no cover — optional dependency
            _STEMMER = False
    return _STEMMER or None


class KeywordRetriever:
    def __init__(self, chunks: Sequence[Evidence]):
        self.chunks = list(chunks)
        corpus = [c.text for c in self.chunks]
        self._bm25 = bm25s.BM25()
        if corpus:
            self._bm25.index(bm25s.tokenize(corpus, stopwords="en", stemmer=_get_stemmer()))

    @classmethod
    def from_index(cls, index, max_docs: int = 100_000) -> "KeywordRetriever":
        """Rebuild the keyword index from the ChromaDB text collection."""
        res = index.text_col.get(limit=max_docs)
        chunks = []
        for doc_text, meta in zip(res["documents"], res["metadatas"]):
            chunks.append(
                Evidence(
                    doc_id=meta["doc_id"],
                    document=meta["document"],
                    page=meta["page"],
                    section=meta.get("section", ""),
                    text=doc_text,
                    image_path=meta.get("image") or None,  # page image rides with the text hit
                )
            )
        return cls(chunks)

    def top_k(self, query: str, k: int = 8) -> list[tuple[Evidence, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        hits, scores = self._bm25.retrieve(
            bm25s.tokenize([query], stopwords="en", stemmer=_get_stemmer()), k=k
        )
        out = []
        for doc_token, score in zip(hits[0], scores[0]):
            out.append((self.chunks[int(doc_token)], float(score)))
        return out
