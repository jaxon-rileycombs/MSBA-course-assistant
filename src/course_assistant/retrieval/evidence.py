"""Evidence: one retrieved source unit, with everything citations need."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    """A retrieved chunk/page with full provenance.

    ``text`` carries the chunk text (or a page snippet for visual hits);
    ``image_path`` is set when the page has a visual the answer can show.
    """

    doc_id: str
    document: str  # filename
    page: int
    section: str
    text: str
    image_path: str | None = None
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        """Merge key: one Evidence per (document, page) after fusion."""
        return (self.doc_id, self.page)
