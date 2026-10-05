"""Structured answer schema: separate ``answer`` and ``sources`` fields,
validated against the evidence that was actually retrieved."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceRef:
    document: str  # filename
    page: int
    section: str = ""
    excerpt: str = ""           # verbatim quote from the evidence text
    image_path: str | None = None  # original page/slide image, if any

    def to_dict(self) -> dict:
        return {
            "document": self.document,
            "page": self.page,
            "section": self.section,
            "excerpt": self.excerpt,
            "image_path": self.image_path,
        }


@dataclass
class Answer:
    answer: str
    sources: list[SourceRef] = field(default_factory=list)
    missing_info: bool = False  # True when the materials cannot answer

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "missing_info": self.missing_info,
        }
