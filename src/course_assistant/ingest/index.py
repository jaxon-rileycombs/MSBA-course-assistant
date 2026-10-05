"""ChromaDB index layer.

The assignment requires text and visual indexes to be kept SEPARATE, so this
module maintains two collections:

- ``text_chunks`` — one vector per chunk (text embeddings), metadata:
  doc_id, document, page, section
- ``visual_pages`` — one vector per page/slide image (visual embeddings),
  metadata: doc_id, document, page, image path

Deleting a document removes its vectors from BOTH collections (requirement
1a: a removed document must no longer be searchable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb

from .chunking import Chunk
from .documents import Document
from .embeddings import ImageEmbedder, TextEmbedder
from ..clients import ClassAPIError


class IndexStore:
    TEXT_COLLECTION = "text_chunks"
    VISUAL_COLLECTION = "visual_pages"

    def __init__(
        self,
        vector_dir: str | Path,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
    ):
        self._client = chromadb.PersistentClient(path=str(vector_dir))
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.text_col = self._client.get_or_create_collection(
            self.TEXT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.visual_col = self._client.get_or_create_collection(
            self.VISUAL_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # ---- writes ------------------------------------------------------

    def upsert_document(self, doc: Document, chunks: Sequence[Chunk]) -> dict:
        """Embed and upsert a document's chunks (text) and page images (visual).

        Returns counts {"text": n, "visual": n}. Idempotent: same chunk IDs
        are overwritten, never duplicated.
        """
        text_ids: list[str] = []
        text_embeddings: list[list[float]] = []
        text_documents: list[str] = []
        text_metadatas: list[dict] = []
        for c in chunks:
            text_ids.append(c.chunk_id)
            text_documents.append(c.text)
            page = doc.pages[c.page - 1]
            text_metadatas.append(
                {
                    "doc_id": c.doc_id,
                    "document": c.document,
                    "page": c.page,
                    "section": c.section,
                    "image": page.image_paths[0] if page.image_paths else "",
                }
            )
        if text_ids:
            text_embeddings = self.text_embedder(text_documents)
            self.text_col.upsert(
                ids=text_ids,
                embeddings=text_embeddings,
                documents=text_documents,
                metadatas=text_metadatas,
            )

        visual_ids: list[str] = []
        visual_embeddings: list[list[float]] = []
        visual_documents: list[str] = []
        visual_metadatas: list[dict] = []
        for page in doc.pages:
            if not page.image_paths:
                continue  # no visual evidence for this page — skip visual index
            image = page.image_paths[0]
            vid = f"{doc.doc_id}:{page.number}"
            try:
                visual_embeddings.append(self.image_embedder([image])[0])
            except ClassAPIError as exc:
                # a picture the embedding service rejects (e.g. over its
                # token budget at any resolution) must not block the corpus
                print(f"  ! skip visual vector {doc.filename} p{page.number}: {exc}", flush=True)
                continue
            visual_ids.append(vid)
            visual_documents.append(f"[{doc.filename} p{page.number}] {page.section}")
            visual_metadatas.append(
                {
                    "doc_id": doc.doc_id,
                    "document": doc.filename,
                    "page": page.number,
                    "image": image,
                }
            )
        if visual_ids:
            self.visual_col.upsert(
                ids=visual_ids,
                embeddings=visual_embeddings,
                documents=visual_documents,
                metadatas=visual_metadatas,
            )
        return {"text": len(text_ids), "visual": len(visual_ids)}

    def remove_document(self, doc_id: str) -> dict:
        """Delete every vector belonging to the document, from BOTH collections."""
        removed: dict[str, int] = {}
        for name, col in (("text", self.text_col), ("visual", self.visual_col)):
            ids = col.get(where={"doc_id": doc_id})["ids"]
            if ids:
                col.delete(ids=ids)
            removed[name] = len(ids)
        return removed

    # ---- reads -------------------------------------------------------

    def counts(self) -> dict:
        return {"text": self.text_col.count(), "visual": self.visual_col.count()}

    def doc_ids(self) -> set[str]:
        ids: set[str] = set()
        for col in (self.text_col, self.visual_col):
            res = col.get()  # ids + metadatas (corpora are small)
            ids.update(m["doc_id"] for m in (res.get("metadatas") or []))
        return ids
