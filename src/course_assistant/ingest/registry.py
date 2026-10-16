"""Document registry — the app's source of truth for known materials.

Requirement 1: loading the same file twice must not duplicate it; removing a
document must remove its searchable content. The registry owns *what* is
known (keyed by doc_id = sha256 of the file bytes); the index layer owns the
*content* cascade (vector entries, images) when a doc is removed.

Persisted as JSON so the app's document list survives restarts.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .documents import Document


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class DocumentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    # ---- persistence -------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self._entries = data
                # Older registry files predate ingest_order. JSON preserves
                # insertion order, so migrate them deterministically in memory.
                for ingest_order, entry in enumerate(self._entries.values(), 1):
                    entry.setdefault("ingest_order", ingest_order)
        except (ValueError, OSError):
            self._entries = {}  # corrupt registry: start clean, don't crash

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._entries, indent=2))
        tmp.replace(self.path)  # atomic on POSIX

    # ---- operations --------------------------------------------------

    def add(self, document: Document) -> tuple[dict, bool]:
        """Register a parsed document. Returns (entry, is_new).

        Adding an already-known doc_id is a NO-OP returning the existing
        entry — this is the dedupe guarantee for repeated uploads.
        """
        with self._lock:
            existing = self._entries.get(document.doc_id)
            if existing is not None:
                return existing, False
            ingest_order = max(
                (int(e.get("ingest_order", 0)) for e in self._entries.values()),
                default=0,
            ) + 1
            entry = {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "format": document.format,
                "pages": document.total_pages,
                "ingest_time": _now(),
                "ingest_order": ingest_order,
            }
            self._entries[document.doc_id] = entry
            self._save()
            return entry, True

    def remove(self, doc_id: str) -> bool:
        """Remove the registry entry. Returns True if it existed."""
        with self._lock:
            if doc_id not in self._entries:
                return False
            del self._entries[doc_id]
            self._save()
            return True

    def get(self, doc_id: str) -> Optional[dict]:
        return self._entries.get(doc_id)

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._entries

    def list(self) -> list[dict]:
        """All entries, newest first with deterministic timestamp tie-breaking."""
        return sorted(
            self._entries.values(),
            key=lambda e: (e["ingest_time"], int(e.get("ingest_order", 0))),
            reverse=True,
        )

    def __len__(self) -> int:
        return len(self._entries)
