"""Embedder adapters.

The real class-service embedders (Nemotron on 9002 for text, Qwen3-VL on
9003 for images) are wired in the retrieval layer once the team's API keys
are available. Until then — and for tests, the dev loop, and demo mode — we
use :class:`HashEmbedder`, a deterministic local embedding that lets the
entire pipeline (ingest → index → retrieve → answer) run end to end. It is
NOT semantically meaningful; it exists to make the plumbing real and
testable, and is clearly labeled as a stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

TextEmbedder = Callable[[Sequence[str]], list[list[float]]]
ImageEmbedder = Callable[[Sequence[str]], list[list[float]]]  # image file paths

_HASH_DIM = 256


def _hash_vector(data: bytes) -> list[float]:
    """Hashed character-bigram vector (L2-normalized).

    Lexically meaningful: strings sharing character bigrams get similar
    vectors. Deliberately NOT semantic — this is the deterministic stand-in
    for tests/dev/demo until the class embedding services are available.
    """
    v = [0.0] * _HASH_DIM
    prev = 32  # space
    for b in data.lower():
        v[(prev * 31 + b) % _HASH_DIM] += 1.0
        prev = b
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class HashEmbedder:
    """Deterministic, dependency-free embeddings for tests/dev/demo.

    Character-bigram hashing gives lexical (not semantic) similarity.
    The real class-service embedders swap in behind the same interfaces.
    """

    dim = _HASH_DIM

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(t.encode("utf-8")) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Single-text embedding for query search (one-model one-space)."""
        return self.embed_texts([text])[0]

    def embed_images(self, paths: Sequence[str]) -> list[list[float]]:
        out = []
        for p in paths:
            try:
                data = Path(p).read_bytes()
            except OSError:
                # Dev/test/demo only: hash the path instead of the (missing) file.
                data = p.encode("utf-8")
            out.append(_hash_vector(data))
        return out
