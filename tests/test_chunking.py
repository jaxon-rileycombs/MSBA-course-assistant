"""Tests for source-preserving chunking."""

import re
from pathlib import Path

from course_assistant.ingest.chunking import chunk_document
from course_assistant.ingest.documents import Document, Page


def make_doc(pages=(
    ("Short page text only.", "Section A", ["img1.png"]),
    ("A" * 4000, "Section B", []),
)):
    return Document(
        doc_id="d" * 64,
        filename="test.pptx",
        format=".pptx",
        source_path="test.pptx",
        pages=[Page(number=i + 1, text=t, section=s, image_paths=im) for i, (t, s, im) in enumerate(pages)],
    )


def test_short_page_single_chunk_with_metadata():
    chunks = chunk_document(make_doc())
    first = chunks[0]
    assert first.text == "Short page text only."
    assert first.page == 1
    assert first.section == "Section A"
    assert first.image_paths == ["img1.png"]
    assert first.chunk_id == "d" * 64 + ":1:0"


def test_long_page_splits_and_pieces_roundtrip():
    doc = make_doc()
    chunks = chunk_document(doc)
    page2 = [c for c in chunks if c.page == 2]
    assert len(page2) > 1
    joined = re.sub(r"\s+", "", "".join(c.text for c in page2))
    assert "A" * 4000 in joined
    assert all(c.document == "test.pptx" and c.doc_id == doc.doc_id for c in page2)
    assert all(c.section == "Section B" for c in page2)


def test_chunk_ids_deterministic():
    doc = make_doc()
    ids1 = [c.chunk_id for c in chunk_document(doc)]
    ids2 = [c.chunk_id for c in chunk_document(doc)]
    assert ids1 == ids2


def test_empty_page_skipped():
    doc = Document(
        doc_id="x" * 64, filename="a.md", format=".md", source_path="a.md",
        pages=[Page(number=1, text="   ")],
    )
    assert chunk_document(doc) == []


def test_pages_keep_their_own_images():
    chunks = chunk_document(make_doc())
    assert {c.page for c in chunks if c.image_paths} == {1}
