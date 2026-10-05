"""Data model for parsed course materials."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_FORMATS = {".pdf", ".pptx", ".docx", ".md", ".txt", ".html", ".htm"}


def file_sha256(path: str | Path) -> str:
    """Content hash used as the document identity (dedupe key)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Page:
    """One page, slide, or whole document (single-page text formats).

    ``number`` is 1-based. ``section`` is the slide title / nearest heading
    ('' when the format has no headings). ``image_paths`` lists files under
    the images root that belong to this page.
    """

    number: int
    text: str
    section: str = ""
    image_paths: list[str] = field(default_factory=list)


@dataclass
class Document:
    """A parsed course material with its source identity and pages."""

    doc_id: str  # sha256 of the source file bytes
    filename: str
    format: str  # lowercase extension with dot, e.g. ".pptx"
    source_path: str
    pages: list[Page]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def text_length(self) -> int:
        return sum(len(p.text) for p in self.pages)
