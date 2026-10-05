"""Quiz engine tests with a fake LLM (no network, no keys)."""

import json
import re

import pytest

from course_assistant.clients import VisionLLMClient
from course_assistant.config import ServiceConfig
from course_assistant.quiz.generator import QuizGenerator, validate_questions
from course_assistant.quiz.schema import Question
from course_assistant.quiz.session import QuizSession
from course_assistant.retrieval.evidence import Evidence

_FAKE_SVC = ServiceConfig(name="fake", base_url="http://fake", model="fake-model", api_key="k")

EVIDENCE = [
    Evidence(doc_id="d1", document="Week 2.pptx", page=33, section="Vibe Coding on Prod",
             text="One does not simply vibe code a production-grade enterprise app."),
    Evidence(doc_id="d1", document="Week 2.pptx", page=34, section="Safety",
             text="Speculative decoding can double throughput in some settings."),
    Evidence(doc_id="d2", document="Syllabus.pdf", page=2, section="Grading",
             text="The final exam is worth 30 percent of the grade."),
]


class FakeLLM(VisionLLMClient):
    def __init__(self, responses):
        super().__init__(_FAKE_SVC, session=object())
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


GOOD_Q = {
    "question": "What is the meme about?",
    "options": ["Production vibes", "Speculative decoding", "Grading", "Nothing"],
    "answer_index": 0,
    "explanation": "The slide is called Vibe Coding on Prod.",
    "sources": [{"document": "Week 2.pptx", "page": 33}],
}


def test_validate_questions_filters_bad_ones():
    allowed = {(e.document, e.page) for e in EVIDENCE}
    good = [GOOD_Q]
    bad = [
        {"question": "", "options": ["a"], "answer_index": 0},                       # empty q
        {"question": "x", "options": ["a"], "answer_index": 0},                      # one option
        {"question": "x", "options": ["a", "b"], "answer_index": 5},                 # key out of range
        {"question": "x", "options": ["a", "b"], "answer_index": 0,
         "sources": [{"document": "Ghost.pptx", "page": 9}]},                         # unretrieved source
    ]
    out = validate_questions(good + bad, allowed)
    assert len(out) == 2  # good one + the well-formed question whose ghost source got filtered
    assert out[0].sources[0]["page"] == 33
    assert out[1].sources == []


def test_generator_parses_structured_quiz():
    payload = {"quiz": [GOOD_Q, {**GOOD_Q, "question": "Second?", "sources": [{"document": "Syllabus.pdf", "page": 2}]}]}
    llm = FakeLLM([json.dumps(payload)])
    qs = QuizGenerator(llm, n=2).generate(EVIDENCE)
    assert llm.calls == 1 and len(qs) == 2
    assert qs[0].answer_index == 0 and qs[1].sources[0]["page"] == 2


def test_generator_raises_when_not_enough_valid():
    llm = FakeLLM(['{"quiz": []}'])
    with pytest.raises(RuntimeError):
        QuizGenerator(llm).generate(EVIDENCE)


def test_session_hides_solutions_until_reveal(tmp_path):
    questions = [Question("Q?", ["a", "b"], 0, "because", [{"document": "Week 2.pptx", "page": 33}])]
    s = QuizSession(tmp_path)
    qid = s.create("Quiz 1", questions)

    view = s.get(qid)
    assert "answer_index" not in view["questions"][0]
    assert "explanation" not in view["questions"][0]

    r = s.submit(qid, [0])
    assert r.score == 1 and r.total == 1 and r.correct == [True]
    assert r.feedback[0]["explanation"] == "because"

    revealed = s.reveal(qid)
    assert revealed["questions"][0]["answer_index"] == 0


def test_session_score_matches_key_exactly(tmp_path):
    questions = [
        Question("Q1?", ["a", "b", "c", "d"], 2, "e", [{"document": "Syllabus.pdf", "page": 2}]),
        Question("Q2?", ["a", "b", "c", "d"], 1, "e", [{"document": "Syllabus.pdf", "page": 2}]),
        Question("Q3?", ["a", "b", "c", "d"], 0, "e", [{"document": "Syllabus.pdf", "page": 2}]),
    ]
    s = QuizSession(tmp_path)
    qid = s.create("T", questions)
    r = s.submit(qid, [2, 1, 0])
    assert (r.score, r.total, r.correct) == (3, 3, [True, True, True])
    r2 = s.submit(qid, [0, 2, 1])  # every answer wrong on purpose
    assert (r2.score, r2.correct) == (0, [False, False, False])


def test_session_persists_and_reloads(tmp_path):
    s = QuizSession(tmp_path)
    qid = s.create("P", [Question("Q?", ["a", "b"], 1, "e", [{"document": "Week 2.pptx", "page": 33}])])
    s2 = QuizSession(tmp_path)
    assert s2.get(qid)["title"] == "P"
    assert s2.submit(qid, [1]).score == 1
    assert [q["quiz_id"] for q in s2.list_all()] == [qid]


def test_session_validates_answer_count(tmp_path):
    s = QuizSession(tmp_path)
    qid = s.create("V", [Question("Q?", ["a", "b"], 0, "e", [])])
    with pytest.raises(ValueError):
        s.submit(qid, [])
