"""UI handler tests: every handler degrades to a friendly message instead of
raising — no raw 500s in the UI, whatever the service does."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → app.py

from app import (  # noqa: E402
    FUN_CSS,
    _friendly,
    do_add,
    do_ask,
    do_generate,
    do_remove,
    do_reveal,
    do_submit,
    scope_choices,
)
from course_assistant.clients import ClassAPIError


def test_quiz_options_css_targets_current_gradio_radio_markup():
    """Gradio 5 renders options inside div.wrap labels, without ARIA roles."""
    assert ".quiz-answer-options > div.wrap" in FUN_CSS
    assert "flex-direction: column" in FUN_CSS
    assert ".quiz-answer-options > div.wrap > label" in FUN_CSS
    assert "width: 100%" in FUN_CSS


class _BrokenService:
    """A service whose every call fails with a service-level error."""

    mode = "online (class services 9001-9004)"

    def list_documents(self):
        return []

    def ask(self, *a, **k):
        raise ClassAPIError("Vision LLM", 503, "upstream busy")

    def render_sources(self, answer):
        return "", []

    def generate_quiz(self, *a, **k):
        raise ClassAPIError("Vision LLM", 503, "upstream busy")

    def get_quiz(self, *a, **k):
        raise KeyError("no such quiz")

    def grade_quiz(self, *a, **k):
        raise ClassAPIError("Quiz", 500, "nope")

    def reveal_quiz(self, *a, **k):
        raise RuntimeError("boom")

    def add_documents(self, *a, **k):
        raise ClassAPIError("Text embeddings", 503, "busy")

    def remove_document(self, *a, **k):
        raise OSError("disk full")


def test_friendly_masks_service_errors():
    exc = ClassAPIError("Vision LLM", 503, "upstream busy")
    msg = _friendly(exc)
    assert "temporarily unavailable" in msg
    assert "upstream busy" not in msg  # key-free, no internals
    assert _friendly(RuntimeError("something broke")) == "something broke"
    assert _friendly(ValueError())


def test_do_ask_never_raises_and_returns_friendly_message():
    md, images = do_ask(_BrokenService(), "what is x?", "All materials", "")
    assert "Couldn't answer right now" in md
    assert images == []


def test_do_ask_empty_question_is_a_noop():
    md, images = do_ask(_BrokenService(), "   ", "All materials", "")
    assert md == "" and images == []


def test_ask_and_quiz_generation_report_progress():
    class Progress:
        def __init__(self):
            self.calls = []

        def __call__(self, value, **kwargs):
            self.calls.append((value, kwargs.get("desc")))

    class Service:
        def ask(self, *args, **kwargs):
            return SimpleNamespace(answer="Answer", missing_info=False, sources=[])

        def render_sources(self, answer):
            return "", []

        def generate_quiz(self, **kwargs):
            return "quiz-1", {"questions": [{"question": "Q?", "options": ["A", "B"]}]}

    ask_progress = Progress()
    quiz_progress = Progress()
    do_ask(Service(), "question", "All materials", "", progress=ask_progress)
    do_generate(Service(), "All materials", "", 2, "", progress=quiz_progress)

    assert ask_progress.calls[0][0] == 0
    assert ask_progress.calls[-1][0] == 1
    assert quiz_progress.calls[0][0] == 0
    assert quiz_progress.calls[-1][0] == 1


def test_do_generate_never_raises_and_resets_ui():
    out = do_generate(_BrokenService(), "All materials", "", 5, "state-1")
    assert out[0] == "state-1"  # state untouched
    assert out[-5] == "" and out[-4] == ""  # no quiz / feedback text
    assert out[-3] == []  # no source images
    assert out[-2]["visible"] is False and out[-1]["visible"] is False


def test_do_generate_shows_action_buttons_after_success():
    class _Svc:
        def generate_quiz(self, **kwargs):
            return "quiz-1", {
                "questions": [{"question": "Q?", "options": ["A", "B"]}]
            }

    out = do_generate(_Svc(), "All materials", "", 2, "")
    assert out[-2]["visible"] is True and out[-1]["visible"] is True


def test_do_submit_and_reveal_degrade_gracefully():
    assert "Couldn't grade right now" in do_submit(_BrokenService(), "q1", "A: x")[0]
    assert "Couldn't reveal" in do_reveal(_BrokenService(), "q1")[0]
    assert do_submit(_BrokenService(), "", "A: x")[0] == "Generate a quiz first."
    assert do_reveal(_BrokenService(), "")[0] == ""


def test_quiz_feedback_includes_excerpt_and_visual_evidence():
    source = {
        "document": "Week 2.pptx",
        "page": 33,
        "section": "Vibe Coding on Prod",
        "excerpt": "One does not simply vibe code a production-grade enterprise app.",
        "image_path": "slide33.png",
    }

    class _Svc:
        def get_quiz(self, qid):
            return {"questions": [{"options": ["right", "wrong"]}]}

        def grade_quiz(self, qid, answers):
            return {
                "score": 1,
                "total": 1,
                "feedback": [{"correct": True, "explanation": "Because.", "sources": [source]}],
            }

        def reveal_quiz(self, qid):
            return {
                "questions": [{
                    "question": "Question?",
                    "options": ["right", "wrong"],
                    "answer_index": 0,
                    "explanation": "Because.",
                    "sources": [source],
                }]
            }

    graded_md, graded_images = do_submit(_Svc(), "q1", "A: right")
    assert "Vibe Coding on Prod" in graded_md
    assert "One does not simply" in graded_md
    assert graded_images == ["slide33.png"]

    revealed_md, revealed_images = do_reveal(_Svc(), "q1")
    assert "Vibe Coding on Prod" in revealed_md
    assert "One does not simply" in revealed_md
    assert revealed_images == ["slide33.png"]


def test_materials_handlers_never_raise():
    status, *_ = do_add(_BrokenService(), ["deck.pptx"])
    assert "Could not index" in status
    status2, *_ = do_remove(_BrokenService(), "deck.pptx")
    assert "Could not remove" in status2
    status3, *_ = do_remove(_BrokenService(), None)
    assert "Pick a document" in status3


def test_add_refreshes_remove_dropdown_choices():
    class _Svc:
        def __init__(self):
            self.docs = []

        def add_documents(self, paths):
            self.docs.append({"filename": "Week 2.pptx"})
            return ["loaded"]

        def list_documents(self):
            return self.docs

    out = do_add(_Svc(), ["Week 2.pptx"])
    assert len(out) == 4
    assert out[-1]["choices"] == ["All materials", "Week 2.pptx"]


def test_remove_refreshes_remove_dropdown_choices():
    class _Svc:
        def __init__(self):
            self.docs = [{"filename": "Week 2.pptx"}]

        def remove_document(self, name):
            self.docs = []
            return f"Removed {name}"

        def list_documents(self):
            return self.docs

    out = do_remove(_Svc(), "Week 2.pptx")
    assert len(out) == 4
    assert out[-1]["choices"] == ["All materials"]


def test_scope_choices_lists_documents():
    class _Svc:
        def list_documents(self):
            return [{"filename": "Week 2.pptx"}, {"filename": "Syllabus.pdf"}]

    assert scope_choices(_Svc()) == ["All materials", "Week 2.pptx", "Syllabus.pdf"]
