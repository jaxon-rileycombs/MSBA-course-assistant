"""UI handler tests: every handler degrades to a friendly message instead of
raising — no raw 500s in the UI, whatever the service does."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → app.py

from app import (  # noqa: E402
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


def test_do_generate_never_raises_and_resets_ui():
    out = do_generate(_BrokenService(), "All materials", "", 5, "state-1")
    assert out[0] == ""  # stale quiz id cleared: nothing hidden to grade
    assert out[-2] == "" and out[-1] == ""  # no quiz / feedback text


def test_do_submit_and_reveal_degrade_gracefully():
    assert "Couldn't grade right now" in do_submit(_BrokenService(), "q1", "A: x")
    assert "Couldn't reveal" in do_reveal(_BrokenService(), "q1")
    assert do_submit(_BrokenService(), "", "A: x") == "Generate a quiz first."
    assert do_reveal(_BrokenService(), "") == ""


def test_materials_handlers_never_raise():
    status, *_ = do_add(_BrokenService(), ["deck.pptx"])
    assert "Could not index" in status
    status2, *_ = do_remove(_BrokenService(), "deck.pptx")
    assert "Could not remove" in status2
    status3, *_ = do_remove(_BrokenService(), None)
    assert "Pick a document" in status3


def test_scope_choices_lists_documents():
    class _Svc:
        def list_documents(self):
            return [{"filename": "Week 2.pptx"}, {"filename": "Syllabus.pdf"}]

    assert scope_choices(_Svc()) == ["All materials", "Week 2.pptx", "Syllabus.pdf"]
