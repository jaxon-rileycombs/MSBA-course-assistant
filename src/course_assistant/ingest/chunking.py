"""Chunk document pages into retrieval units while preserving source details.

Every chunk carries the full provenance needed for citations and filtering:
document name, 1-based page/slide number, section (slide title / heading),
and the page's image paths (for visual evidence).

Chunk IDs are deterministic: ``<doc_id>:<page>:<index>``, so re-ingesting the
same corpus yields identical IDs (no duplicates, stable updates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    document: str  # filename
    page: int  # 1-based
    section: str  # slide title / heading ('' if none)
    text: str
    image_paths: list[str] = field(default_factory=list)


def chunk_document(
    doc: Document,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split a parsed document into source-preserving chunks (one page at a time)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: list[Chunk] = []
    for page in doc.pages:
        text = page.text.strip()
        if not text:
            continue
        pieces = splitter.split_text(text) if len(text) > chunk_size else [text]
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{page.number}:{i}",
                    doc_id=doc.doc_id,
                    document=doc.filename,
                    page=page.number,
                    section=page.section,
                    text=piece,
                    image_paths=list(page.image_paths),
                )
            )
    return chunks
