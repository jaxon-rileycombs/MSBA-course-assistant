"""Adapters presenting the class embedding services behind the retriever and
index interfaces (batched calls, query-vs-image modes).

- :class:`TextEmbedderAdapter`   — Nemotron (9002), text->vectors
- :class:`VisualEmbedderAdapter` — Qwen3-VL-Embedding (9003): images for
  indexing, TEXT for queries — one multimodal model, one vector space
  (see retrieval/embedding_retriever module docs for why).
"""

from __future__ import annotations

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
        for i in range(0, len(image_paths), self.batch_size):
            out.extend(self.client.embed_images(list(image_paths[i : i + self.batch_size])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed(text)[0]
