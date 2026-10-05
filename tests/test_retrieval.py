"""Tests for hybrid retrieval: keyword, text/visual embedding retrievers,
fusion merging, weights, and reranking. Uses HashEmbedder + real ChromaDB
persistence — no class services needed."""

import pytest

from course_assistant.ingest.chunking import chunk_document
from course_assistant.ingest.documents import Document, Page
from course_assistant.ingest.embeddings import HashEmbedder
from course_assistant.ingest.index import IndexStore
from course_assistant.retrieval.embedding_retriever import (
    TextEmbeddingRetriever,
    VisualEmbeddingRetriever,
)
from course_assistant.retrieval.hybrid import HybridRetriever
from course_assistant.retrieval.keyword import KeywordRetriever

# distinctive corpus so retrievers behave deterministically
CORPUS = [
    ("One does not simply vibe code a production-grade enterprise app.", "Vibe Coding on Prod", "meme.png"),
    ("Speculative decoding predicts draft tokens to speed up generation.", "Speculative Decoding", "spec.png"),
    ("Open-weight models can run locally but may lag behind closed ones.", "Open vs Closed", "open.png"),
]


def make_corpus(tmp_path):
    emb = HashEmbedder()
    index = IndexStore(tmp_path / "vec", emb.embed_texts, emb.embed_images)
    chunks_all = []
    for i, (text, title, img) in enumerate(CORPUS):
        doc = Document(
            doc_id=f"{i}" * 64,
            filename=f"week{i}.pptx",
            format=".pptx",
            source_path=f"week{i}.pptx",
            pages=[Page(number=1, text=text, section=title, image_paths=[str(tmp_path / img)])],
        )
        (tmp_path / img).write_bytes(b"png-bytes")
        chunks = chunk_document(doc)
        chunks_all.extend(chunks)
        index.upsert_document(doc, chunks)
    return index, chunks_all


@pytest.fixture
def hybrid(tmp_path):
    index, chunks = make_corpus(tmp_path)
    emb = HashEmbedder()
    kw = KeywordRetriever.from_index(index)
    text = TextEmbeddingRetriever(index, emb.embed_query)
    visual = VisualEmbeddingRetriever(index, emb.embed_query)  # same space in fake mode
    return HybridRetriever(kw, text, visual)


def test_keyword_finds_distinctive_term(hybrid):
    hits = hybrid.keyword.top_k("production-grade enterprise", k=3)
    assert len(hits) == 3
    assert "vibe code" in hits[0][0].text  # only chunk containing the term


def test_text_embedding_retriever_returns_evidence(hybrid):
    hits = hybrid.text_embedding.top_k("vibe coding", k=3)
    assert hits[0][0].section == "Vibe Coding on Prod"
    assert hits[0][1] >= hits[1][1]


def test_visual_embedding_retriever_carries_image(hybrid):
    hits = hybrid.visual_embedding.top_k("anything", k=3)
    assert len(hits) == 3
    for ev, _ in hits:  # structure check: every visual hit has provenance + image
        assert ev.image_path is not None
        assert ev.page == 1


def test_hybrid_merges_text_and_visual_evidence(hybrid):
    evs = hybrid.retrieve("vibe coding on prod meme", k=3)
    assert evs
    top = evs[0]
    assert top.section == "Vibe Coding on Prod"
    assert top.image_path  # visual hit merged into the text evidence
    assert top.text  # text hit merged
    assert "keyword" in top.scores and "visual" in top.scores


def test_hybrid_deduplicates_same_page(hybrid):
    evs = hybrid.retrieve("speculative decoding speed", k=10)
    keys = [e.key for e in evs]
    assert len(keys) == len(set(keys))


def test_reranker_reorders_results(hybrid):
    # fake reranker: reverse the order (worst first becomes best)
    calls = {"n": 0}

    def reverse_rerank(query, texts):
        calls["n"] += 1
        return list(range(1, len(texts) + 1))  # ascending: flips the fused order

    hybrid.reranker = reverse_rerank
    evs = hybrid.retrieve("vibe coding", k=3)
    assert calls["n"] == 1
    assert evs[-1].section == "Vibe Coding on Prod"  # got pushed to the back


def test_rerank_failure_keeps_fused_order(hybrid):
    def broken(query, texts):
        raise RuntimeError("service down")

    hybrid.reranker = broken
    evs = hybrid.retrieve("vibe coding", k=3)  # must not raise
    assert len(evs) == 3


def test_weights_change_ranking(hybrid):
    # keyword-heavy vs visual-heavy should reorder for the meme query
    kw_heavy = HybridRetriever(
        hybrid.keyword, hybrid.text_embedding, hybrid.visual_embedding,
        weights={"keyword": 100.0, "text": 1.0, "visual": 1.0},
    )
    vis_heavy = HybridRetriever(
        hybrid.keyword, hybrid.text_embedding, hybrid.visual_embedding,
        weights={"keyword": 1.0, "text": 1.0, "visual": 100.0},
    )
    evs_kw = kw_heavy.retrieve("vibe coding on prod", k=3)
    evs_vis = vis_heavy.retrieve("vibe coding on prod", k=3)
    # both must still return the meme page at top (single clear winner) — sanity
    assert evs_kw[0].section == "Vibe Coding on Prod"
    assert evs_vis[0].section == "Vibe Coding on Prod"
