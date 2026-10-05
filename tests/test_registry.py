"""Tests for the document registry (dedupe, removal, persistence)."""

import json
from pathlib import Path

from course_assistant.ingest.documents import Document
from course_assistant.ingest.registry import DocumentRegistry


def make_doc(doc_id: str, name: str = "a.pptx") -> Document:
    return Document(
        doc_id=doc_id,
        filename=name,
        format=".pptx",
        source_path=name,
        pages=[],
    )


def test_add_new_and_repeat_upload_is_deduped(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    doc = make_doc("a" * 64)
    entry, is_new = reg.add(doc)
    assert is_new is True
    assert entry["filename"] == "a.pptx"
    assert entry["pages"] == 0

    entry2, is_new2 = reg.add(doc)  # same doc_id -> no-op
    assert is_new2 is False
    assert entry2 == entry
    assert len(reg) == 1


def test_remove_deletes_entry(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.add(make_doc("a" * 64))
    reg.add(make_doc("b" * 64))
    assert reg.remove("a" * 64) is True
    assert reg.remove("a" * 64) is False  # already gone
    remaining = reg.list()
    assert len(remaining) == 1
    assert remaining[0]["doc_id"] == "b" * 64


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    reg = DocumentRegistry(path)
    reg.add(make_doc("a" * 64, "syllabus.pdf"))
    reg2 = DocumentRegistry(path)
    assert reg2.contains("a" * 64)
    assert reg2.list()[0]["filename"] == "syllabus.pdf"


def test_corrupt_registry_starts_clean(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{not json!!")
    reg = DocumentRegistry(path)
    assert len(reg) == 0
    reg.add(make_doc("a" * 64))
    assert reg.contains("a" * 64)


def test_list_newest_first(tmp_path):
    reg = DocumentRegistry(tmp_path / "registry.json")
    reg.add(make_doc("1" * 64))
    reg.add(make_doc("2" * 64))
    order = [e["doc_id"] for e in reg.list()]
    assert order == ["2" * 64, "1" * 64]  # newest first
