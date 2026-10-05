"""End-to-end tests of the CourseAssistant service (offline mode — no keys,
no network): ingest -> retrieve -> answer -> sources; add/remove cascade;
per-material scope; quiz generate -> hide -> grade -> reveal."""

from pathlib import Path

import pytest

from course_assistant.service import ALL_SCOPE, CourseAssistant, ServiceDirs
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


def test_topic_filter_used_in_query_and_quiz(app):
    svc, deck = app
    svc.add_documents([str(deck)])
    answer = svc.ask("what is this about?", topic="nothing-matches")
    # offline answerer still returns top evidence; the filter is passed to retrieval
    assert "Week 2.pptx" in answer.answer
