"""Adapters presenting the class embedding services behind the retriever and
index interfaces (batched calls, query-vs-image modes).

- :class:`TextEmbedderAdapter`   — Nemotron (9002), text->vectors
- :class:`VisualEmbedderAdapter` — Qwen3-VL-Embedding (9003): images for
  indexing, TEXT for queries — one multimodal model, one vector space
  (see retrieval/embedding_retriever module docs for why).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


class TextEmbedderAdapter:
    def __init__(self, client, batch_size: int = 64):
        self.client = client
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self.client.embed(list(texts[i : i + self.batch_size])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(text)[0]


class VisualEmbedderAdapter:
    def __init__(self, client, batch_size: int = 16):
        self.client = client
        self.batch_size = batch_size

    def embed_images(self, image_paths: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in self._batches(image_paths):
            out.extend(self.client.embed_images(batch))
        return out

    @staticmethod
    def _batches(
        image_paths: Sequence[str],
        max_count: int = 4,
        max_chars: int = 700_000,
    ) -> list[list[str]]:
        """Split paths into batches the embedding service can accept.

        The server's 8192-token window bounds two things: total input size
        (~1M characters) and IMAGE-TOKEN COUNT — empirically the Qwen
        vision embedder consumes ~1K tokens per 512px image, so more than
        4 images per request overflows (verified live: 4 ok, 8 fails).
        Budget by count and by decoded byte size."""
        cap_bytes = int(max_chars * 3 / 4)
        batches: list[list[str]] = [[]]
        budget = cap_bytes
        for p in image_paths:
            size = max(1, Path(p).stat().st_size)
            if batches[-1] and (len(batches[-1]) >= max_count or budget - size < 0):
                batches.append([])
                budget = cap_bytes
            batches[-1].append(p)
            budget -= size
        return [b for b in batches if b]

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(text)[0]
