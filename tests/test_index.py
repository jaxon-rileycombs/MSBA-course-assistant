"""Tests for the ChromaDB index layer (fake embedders — no keys needed)."""

import pytest

from course_assistant.ingest.chunking import chunk_document
from course_assistant.ingest.documents import Document, Page
from course_assistant.ingest.embeddings import HashEmbedder
from course_assistant.ingest.index import IndexStore


def make_doc(doc_id: str, filename: str, pages=(("Text A", "Sec A", ["img_a.png"]),)) -> Document:
    return Document(
        doc_id=doc_id,
        filename=filename,
        format=".pptx",
        source_path=filename,
        pages=[Page(number=i + 1, text=t, section=s, image_paths=im)
               for i, (t, s, im) in enumerate(pages)],
    )


@pytest.fixture
def store(tmp_path):
    emb = HashEmbedder()
    return IndexStore(tmp_path / "vec", emb.embed_texts, emb.embed_images)


def test_upsert_indexes_text_and_visual_separately(store):
    doc = make_doc("a" * 64, "deck.pptx")
    chunks = chunk_document(doc)
    counts = store.upsert_document(doc, chunks)
    assert counts == {"text": 1, "visual": 1}
    assert store.counts() == {"text": 1, "visual": 1}

    ids = store.text_col.get(where={"doc_id": "a" * 64})["ids"]
    assert len(ids) == 1
    meta = store.text_col.get(where={"doc_id": "a" * 64})["metadatas"][0]
    assert meta["document"] == "deck.pptx"
    assert meta["page"] == 1
    assert meta["section"] == "Sec A"
    vmeta = store.visual_col.get(where={"doc_id": "a" * 64})["metadatas"][0]
    assert vmeta["image"] == "img_a.png"


def test_unembeddable_image_is_skipped_not_fatal(tmp_path):
    from course_assistant.clients import ClassAPIError

    emb = HashEmbedder()
    reject_img = lambda paths: (_ for _ in ()).throw(
        ClassAPIError("Visual embeddings", 400, "over token budget")
    )
    ok_img = emb.embed_images
    call_count = {"n": 0}

    def flaky_embed(paths):
        call_count["n"] += 1
        if any("reject" in p for p in paths):
            return reject_img(paths)
        return ok_img(paths)

    store = IndexStore(tmp_path / "vec", emb.embed_texts, flaky_embed)
    doc = make_doc(
        "b" * 64,
        "mixed.pptx",
        pages=(
            ("Text A", "Sec A", ["good_a.png"]),
            ("Text B", "Sec B", ["reject_b.png"]),
            ("Text C", "Sec C", ["good_c.png"]),
        ),
    )
    chunks = chunk_document(doc)
    counts = store.upsert_document(doc, chunks)
    # text indexed for all pages; the rejected image simply has no vector
    assert counts["text"] == 3
    assert counts["visual"] == 2
    assert call_count["n"] == 3  # all three images were attempted


def test_upsert_is_idempotent(store):
    doc = make_doc("b" * 64, "deck.pptx")
    store.upsert_document(doc, chunk_document(doc))
    store.upsert_document(doc, chunk_document(doc))
    assert store.counts() == {"text": 1, "visual": 1}


def test_pages_without_images_skip_visual_index(store):
    doc = make_doc("c" * 64, "txt.md", pages=(("No image page", "", []),))
    counts = store.upsert_document(doc, chunk_document(doc))
    assert counts == {"text": 1, "visual": 0}


def test_remove_document_cleans_both_collections(store):
    store.upsert_document(make_doc("d1", "a.pptx"), chunk_document(make_doc("d1", "a.pptx")))
    store.upsert_document(make_doc("d2", "b.pptx"), chunk_document(make_doc("d2", "b.pptx")))
    removed = store.remove_document("d1")
    assert removed == {"text": 1, "visual": 1}
    assert store.counts() == {"text": 1, "visual": 1}  # d2 survives
    assert store.doc_ids() == {"d2"}


def test_remove_unknown_doc_is_safe(store):
    assert store.remove_document("zzz") == {"text": 0, "visual": 0}
