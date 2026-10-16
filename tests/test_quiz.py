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
             text="One does not simply vibe code a production-grade enterprise app.",
             image_path="slide33.png"),
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
        return self.responses.pop(0) if self.responses else "{}"


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
    out = validate_questions(good + bad, EVIDENCE)
    assert len(out) == 1  # unsupported citations invalidate the question
    assert out[0].sources[0] == {
        "document": "Week 2.pptx",
        "page": 33,
        "section": "Vibe Coding on Prod",
        "excerpt": "One does not simply vibe code a production-grade enterprise app.",
        "image_path": "slide33.png",
    }


def test_generator_parses_structured_quiz():
    second = {
        "question": "What share of the grade is the final exam?",
        "options": ["30 percent", "40 percent", "Half", "10 percent"],
        "answer_index": 0,
        "explanation": "The syllabus says the final exam is worth 30 percent of the grade.",
        "sources": [{"document": "Syllabus.pdf", "page": 2}],
    }
    payload = {"quiz": [GOOD_Q, second]}
    llm = FakeLLM([json.dumps(payload)])
    qs = QuizGenerator(llm, n=2).generate([EVIDENCE[0], EVIDENCE[-1]])
    assert llm.calls == 1 and len(qs) == 2
    # grounded: the correct option is present in the cited evidence
    assert qs[0].sources[0]["page"] == 33
    assert "30 percent" in EVIDENCE[2].text
    assert qs[0].options[qs[0].answer_index] == "Production vibes"


def test_generator_raises_when_not_enough_valid():
    llm = FakeLLM(['{"quiz": []}'])
    with pytest.raises(RuntimeError):
        QuizGenerator(llm).generate(EVIDENCE)


def test_generator_honors_requested_count():
    q2 = {"question": "What technique can roughly double throughput?",
          "options": ["Speculative decoding", "Vibe coding", "Prompt tuning", "Reranking"],
          "answer_index": 0,
          "explanation": "The materials say speculative decoding can double throughput in some settings.",
          "sources": [{"document": "Week 2.pptx", "page": 34}]}
    q3 = {"question": "What share of the grade is the final exam?",
          "options": ["30 percent", "40 percent", "Half", "10 percent"],
          "answer_index": 0,
          "explanation": "The syllabus says the final exam is worth 30 percent of the grade.",
          "sources": [{"document": "Syllabus.pdf", "page": 2}]}
    payload = {"quiz": [GOOD_Q, q2, q3]}
    llm = FakeLLM([json.dumps(payload)])
    qs = QuizGenerator(llm, n=5).generate(EVIDENCE, n=3)
    assert llm.calls == 1 and len(qs) == 3  # slider value wins, never more
    assert sorted(q.sources[0]["page"] for q in qs) == [2, 33, 34]  # one slide each


def test_generator_retries_once_when_short():
    q2 = {"question": "What technique can roughly double throughput?",
          "options": ["Speculative decoding", "Vibe coding", "Prompt tuning", "Reranking"],
          "answer_index": 0,
          "explanation": "The materials say speculative decoding can double throughput.",
          "sources": [{"document": "Week 2.pptx", "page": 34}]}
    q3 = {"question": "What share of the grade is the final exam?",
          "options": ["30 percent", "40 percent", "Half", "10 percent"],
          "answer_index": 0,
          "explanation": "The syllabus says the final exam is worth 30 percent of the grade.",
          "sources": [{"document": "Syllabus.pdf", "page": 2}]}
    short = json.dumps({"quiz": [GOOD_Q]})
    full = json.dumps({"quiz": [GOOD_Q, q2, q3]})
    llm = FakeLLM([short, full])
    qs = QuizGenerator(llm, n=3).generate(EVIDENCE)
    assert llm.calls == 2 and len(qs) == 3  # second attempt filled the count


def test_generator_exact_count_never_more():
    enough_evidence = EVIDENCE + [
        Evidence(doc_id="d1", document="Week 2.pptx", page=35, section="More",
                 text="Extra evidence entry for a longer deck."),
        Evidence(doc_id="d1", document="Week 2.pptx", page=36, section="More",
                 text="Another evidence entry, still on the same deck."),
        Evidence(doc_id="d1", document="Week 2.pptx", page=37, section="More",
                 text="A third extra entry so there is more evidence than questions."),
    ]
    q2 = {"question": "What technique can roughly double throughput?",
          "options": ["Speculative decoding", "Vibe coding", "Prompt tuning", "Reranking"],
          "answer_index": 0,
          "explanation": "The materials say speculative decoding can double throughput.",
          "sources": [{"document": "Week 2.pptx", "page": 34}]}
    q3 = {"question": "What share of the grade is the final exam?",
          "options": ["30 percent", "40 percent", "Half", "10 percent"],
          "answer_index": 0,
          "explanation": "The syllabus says the final exam is worth 30 percent of the grade.",
          "sources": [{"document": "Syllabus.pdf", "page": 2}]}
    q4 = {"question": "What does the extra passage describe?",
          "options": ["Extra evidence entry", "An exam policy", "A random topic", "Nothing at all"],
          "answer_index": 0,
          "explanation": "It is an extra evidence entry for a longer deck.",
          "sources": [{"document": "Week 2.pptx", "page": 35}]}
    q5 = {**GOOD_Q, "question": "A fifth question that repeats the first slide."}
    payload = {"quiz": [GOOD_Q, q2, q3, q4, q5]}
    llm = FakeLLM([json.dumps(payload)])
    qs = QuizGenerator(llm, n=5).generate(enough_evidence, n=4)
    assert len(qs) == 4 and llm.calls == 1  # model sent 5, one repeated a slide — keep 4
    assert len({(q.sources[0]["document"], q.sources[0]["page"]) for q in qs}) == 4


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


def test_session_rejects_unsafe_quiz_ids(tmp_path):
    session = QuizSession(tmp_path / "quizzes")
    questions = [Question("Q?", ["a", "b"], 0, "e", [])]
    with pytest.raises(ValueError):
        session.create("unsafe", questions, quiz_id="../outside")
    with pytest.raises(ValueError):
        session.get("../outside")
    assert not (tmp_path / "outside.json").exists()


def test_session_validates_answer_count(tmp_path):
    s = QuizSession(tmp_path)
    qid = s.create("V", [Question("Q?", ["a", "b"], 0, "e", [])])
    with pytest.raises(ValueError):
        s.submit(qid, [])


# ---- strict-content rules (user: answers never given away, questions
# ---- strictly about the material, no deck/week references) -------------

def _validate(payload: dict, evidence=EVIDENCE) -> list[Question]:
    return validate_questions(payload["quiz"], evidence)


def test_options_prefix_labels_are_stripped():
    q = {**GOOD_Q, "options": ["A. Production vibes", "B. Speculative decoding",
                                "C: Grading", "D) Nothing"]}
    out = _validate({"quiz": [q]})
    assert len(out) == 1
    assert set(out[0].options) == {"Production vibes", "Speculative decoding", "Grading", "Nothing"}
    assert not any(re.match(r"^[A-D][.):]", o) for o in out[0].options)  # no prefix residue
    assert out[0].options[out[0].answer_index] == "Production vibes"


def test_deck_and_week_references_are_rejected():
    for bad in ("According to the slides, what is the meme about?",
                "What did we cover in week 2 about memory?",
                "Which slide explains speculative decoding?",
                "Per the course material, what is the meme about?"):
        out = _validate({"quiz": [{**GOOD_Q, "question": bad}]})
        assert out == [], f"should reject: {bad!r}"


def test_question_that_gives_answer_away_is_rejected():
    # question contains two+ significant words of the key -> leak
    leaky = {**GOOD_Q, "question": "Which option describes production vibes?"}
    assert _validate({"quiz": [leaky]}) == []


def test_duplicate_questions_are_deduped():
    payload = {"quiz": [GOOD_Q, {"question": "What is the   meme about?",
                                  "options": ["z", "y", "x", "w"], "answer_index": 1,
                                  "sources": [{"document": "Week 2.pptx", "page": 33}]}]}
    out = _validate(payload)
    assert len(out) == 1  # normalized text matches -> deduped


def test_ungrounded_question_is_rejected():
    # correct option appears nowhere in the cited page's text
    q = {
        "question": "Which color is the sky?",
        "options": ["Blue", "Cyan", "Navy", "Azure"],
        "answer_index": 0,
        "explanation": "It just is.",
        "sources": [{"document": "Week 2.pptx", "page": 33}],
    }
    assert _validate({"quiz": [q]}) == []  # 'blue' not in the vibe text


def test_key_position_is_shuffled_deterministically():
    q = {**GOOD_Q, "question": "A unique question seed for the shuffle."}
    out = _validate({"quiz": [q]})
    assert len(out) == 1
    question = out[0]
    import random

    positions = list(range(4))
    random.Random(question.question.strip()).shuffle(positions)
    assert question.answer_index == positions.index(0)
    assert question.options[question.answer_index] == "Production vibes"


def test_no_two_questions_from_the_same_slide():
    same_page = {**GOOD_Q, "question": "A different question on the very same page.",
                 "options": ["Production vibes", "Something else", "Another", "Fourth"]}
    payload = {"quiz": [GOOD_Q, same_page]}
    out = _validate(payload)
    assert len(out) == 1  # second question cites the already-used slide -> dropped
