"""End-to-end tests of the CourseAssistant service (offline mode — no keys,
no network): ingest -> retrieve -> answer -> sources; add/remove cascade;
per-material scope; quiz generate -> hide -> grade -> reveal."""

from pathlib import Path

import pytest

from course_assistant.qa.schema import Answer
from course_assistant.retrieval.evidence import Evidence
from course_assistant.service import ALL_SCOPE, CourseAssistant, ServiceDirs, _meme_candidate_score
from test_formats import make_pptx


@pytest.fixture
def app(tmp_path):
    deck = make_pptx(tmp_path / "Week 2.pptx", slides=2)
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    return svc, deck


def test_add_dedupe_and_list(app):
    svc, deck = app
    msgs = svc.add_documents([str(deck)])
    assert "2 pages" in msgs[0]
    assert svc.list_documents()[0]["filename"] == "Week 2.pptx"
    # repeat upload is a no-op
    again = svc.add_documents([str(deck)])
    assert "duplicate" in again[0]
    assert len(svc.list_documents()) == 1


def test_ask_returns_grounded_sources_with_image(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    answer = svc.ask("what does vibe coding mean?")
    assert "Week 2.pptx" in answer.answer
    assert answer.sources and answer.sources[0].document == "Week 2.pptx"
    assert answer.sources[0].image_path  # visual evidence attached
    md, images = svc.render_sources(answer)
    assert "Week 2.pptx" in md and images
    assert Path(images[0]).exists()


def test_remove_document_cleans_everything(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    doc_id = svc.list_documents()[0]["doc_id"]
    msg = svc.remove_document(doc_id)
    assert "Removed" in msg and "text vectors 2" in msg and "visual vectors 2" in msg
    assert svc.list_documents() == []
    answer = svc.ask("anything about vibe coding?")
    assert answer.missing_info and answer.sources == []


def test_remove_by_filename(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    svc.remove_document("Week 2.pptx")
    assert svc.list_documents() == []


def test_material_scope_restricts_evidence(app):
    svc, deck = app
    svc.add_documents([str(deck)])

    class _Capture:
        seen: list = []

    orig = svc.answerer.answer

    def spy(question, evidence):
        _Capture.seen = [e.document for e in evidence]
        return orig(question, evidence)

    svc.answerer.answer = spy
    svc.ask("vibe coding", scope="Week 2.pptx")
    assert _Capture.seen and set(_Capture.seen) == {"Week 2.pptx"}


def test_quiz_flow_hides_grades_reveals(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    quiz_id, view = svc.generate_quiz(scope="Week 2.pptx", n=2)
    assert len(view["questions"]) == 2
    assert "answer_index" not in view["questions"][0]  # solutions hidden

    # offline quizzes: correct answer is always index 0
    result = svc.grade_quiz(quiz_id, [0, 0])
    assert result["score"] == 2 and result["total"] == 2
    assert result["feedback"][0]["correct"] is True

    svc.grade_quiz(quiz_id, [1, 1])  # wrong answers
    assert result["score"] == 2  # unchanged: fixed key, no mutation

    revealed = svc.reveal_quiz(quiz_id)
    assert revealed["questions"][0]["answer_index"] == 0
    assert revealed["questions"][0]["explanation"]
    source = revealed["questions"][0]["sources"][0]
    assert source["excerpt"]
    assert source["image_path"] and Path(source["image_path"]).exists()


def test_meme_queries_prioritize_visual_meme_candidates(app, monkeypatch):
    svc, _ = app
    meme = Evidence(
        doc_id="meme",
        document="Week 2.pptx",
        page=33,
        section="Vibe Coding on Prod",
        text="Vibe Coding on Prod Notes: reddit.com/r/vibecoding/example",
        image_path="lord-of-the-rings.png",
    )
    ordinary = Evidence(
        doc_id="other",
        document="Week 3.pptx",
        page=2,
        section="Prompting",
        text="Prompt engineering overview",
    )
    monkeypatch.setattr(svc, "_meme_evidence", lambda: [meme])
    monkeypatch.setattr(svc._hybrid, "retrieve", lambda *args, **kwargs: [ordinary])
    seen = {}

    class CaptureAnswerer:
        def answer(self, question, evidence):
            seen["evidence"] = evidence
            return Answer(answer="Found it.")

    svc.answerer = CaptureAnswerer()
    svc.ask("Find the memes in the slides")

    assert seen["evidence"][0].page == 33
    assert len(seen["evidence"]) == 1  # unrelated retrieval cannot distract visual search
    assert _meme_candidate_score(meme.text, meme.section) > 0
    assert _meme_candidate_score("ordinary Reddit API documentation", "API") == 0


def test_topic_filter_limits_evidence(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    # a topic with no match anywhere -> explicit message, no silent mixing
    answer = svc.ask("what is this about?", topic="suchness-not-in-corpus")
    assert answer.missing_info
    assert "mentions the topic" in answer.answer

    # a matching topic -> only chunks mentioning it are used
    answer2 = svc.ask("what?", topic="embedded")  # 'embedded concepts' in the deck text
    assert "Slide 2 title" in answer2.answer or "embedded" in answer2.answer.lower()
    assert answer2.sources and all("Week 2.pptx" == s.document for s in answer2.sources)


def test_service_dirs_can_be_isolated_with_environment(monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_VECTOR_DIR", "vectorstore-online")
    monkeypatch.setenv("COURSE_ASSISTANT_REGISTRY_PATH", "data/registry-online.json")

    dirs = ServiceDirs()

    assert dirs.vector_dir == "vectorstore-online"
    assert dirs.registry_path == "data/registry-online.json"


def test_quiz_topic_filter_is_strict(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    # a topic nobody mentions -> explicit error, no silent whole-deck quiz
    with pytest.raises(RuntimeError, match="mentions the topic"):
        svc.generate_quiz(topic="suchness-not-in-corpus", n=2)
    # a matching topic -> quiz is generated (offline placeholder counts as one)
    quiz_id, view = svc.generate_quiz(topic="embedded", n=2)
    assert len(view["questions"]) == 2
    assert svc.get_quiz(quiz_id)["quiz_id"] == quiz_id


def test_quiz_on_empty_index_raises_friendly_runtime_error(tmp_path):
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    with pytest.raises(RuntimeError, match="No course materials are indexed yet"):
        svc.generate_quiz(n=3)


def test_ask_on_empty_index_acknowledges_missing_information(tmp_path):
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    answer = svc.ask("What is the final exam worth?")
    assert answer.missing_info
    assert answer.sources == []
    assert "could not" in answer.answer.lower() or "not" in answer.answer.lower()


def test_quiz_requested_count_is_honored_end_to_end(tmp_path):
    deck = make_pptx(tmp_path / "Week 2.pptx", slides=10)
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    svc.add_documents([str(deck)])
    quiz_id, view = svc.generate_quiz(scope="Week 2.pptx", n=3)
    assert len(view["questions"]) == 3  # the slider's value, exactly
    result = svc.grade_quiz(quiz_id, [0, 0, 0])
    assert result["total"] == 3


def test_quiz_sampling_spreads_across_the_whole_deck(tmp_path):
    deck = make_pptx(tmp_path / "Week 2.pptx", slides=12)
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    svc.add_documents([str(deck)])
    sampled = svc._scope_chunks("Week 2.pptx", 3)
    pages = sorted(e.page for e in sampled)
    assert len(pages) == 3  # one sample unit per slide
    assert pages[0] == 1 and pages[-1] >= 9  # covers early AND late slides


def test_scope_chunks_never_repeats_a_slide_when_avoidable(tmp_path):
    """Two chunks on the same page must not both be quizzed — a quiz should
    never ask twice about one slide while other slides go unused."""
    from course_assistant.ingest.embeddings import HashEmbedder
    from course_assistant.ingest.index import IndexStore
    from course_assistant.retrieval.evidence import Evidence

    deck = make_pptx(tmp_path / "Week 2.pptx", slides=2)
    svc = CourseAssistant(
        dirs=ServiceDirs(
            vector_dir=str(tmp_path / "vec"),
            images_root=str(tmp_path / "imgs"),
            registry_path=str(tmp_path / "reg.json"),
            quiz_dir=str(tmp_path / "quizzes"),
        ),
        force_offline=True,
    )
    svc.add_documents([str(deck)])

    emb = HashEmbedder()
    store = IndexStore(tmp_path / "vec2", emb.embed_texts, emb.embed_images)
    # page 1 twice (like a long slide split into two chunks) + page 2 once
    chunks = [
        Evidence(doc_id="x" * 64, document="Week 2.pptx", page=1, section="A",
                 text="First half of slide one content with enough words to matter here."),
        Evidence(doc_id="x" * 64, document="Week 2.pptx", page=1, section="A",
                 text="Second half of the same slide one content, still long enough."),
        Evidence(doc_id="x" * 64, document="Week 2.pptx", page=2, section="B",
                 text="Slide two content, deliberately longer than the title floor."),
    ]
    store.text_col.upsert(
        ids=["a", "b", "c"],
        embeddings=[emb.embed_texts([c.text])[0] for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"doc_id": c.doc_id, "document": c.document, "page": c.page,
             "section": c.section, "image": ""}
            for c in chunks
        ],
    )
    svc._index = store
    svc._refresh_keyword()

    sampled = svc._scope_chunks("Week 2.pptx", 2)
    assert len(sampled) == 2
    assert sorted(e.page for e in sampled) == [1, 2]  # one chunk per slide

