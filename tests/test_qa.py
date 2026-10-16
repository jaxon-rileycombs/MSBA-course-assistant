"""QA pipeline tests against a fake vision LLM (no network, no keys)."""

import pytest

from course_assistant.clients import VisionLLMClient
from course_assistant.config import ServiceConfig
from course_assistant.qa.answer import AnswerError, Answerer, OfflineAnswerer
from course_assistant.qa.schema import SourceRef
from course_assistant.qa.validate import parse_json_payload, validate_answer
from course_assistant.retrieval.evidence import Evidence

_FAKE_SVC = ServiceConfig(name="fake", base_url="http://fake", model="fake-model", api_key="k")

EV_BASE = [
    Evidence(
        doc_id="d1", document="Week 2.pptx", page=33, section="Vibe Coding on Prod",
        text="One does not simply vibe code a production-grade enterprise app.",
        image_path="img.png",
    ),
    Evidence(
        doc_id="d2", document="Syllabus.pdf", page=2, section="Grading",
        text="The final exam is worth 30% of the grade.",
    ),
]


class FakeLLM(VisionLLMClient):
    def __init__(self, responses, on_messages=None):
        super().__init__(_FAKE_SVC, session=object())
        self.responses = list(responses)
        self.calls = 0
        self.on_messages = on_messages

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.on_messages:
            self.on_messages(messages, kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra chat call")
        return self.responses.pop(0)


def test_offline_answerer_uses_top_evidence():
    a = OfflineAnswerer().answer("what does the meme say?", EV_BASE)
    assert "Week 2.pptx" in a.answer and "slide 33" in a.answer
    assert len(a.sources) == 1
    assert a.sources[0].image_path == "img.png"
    assert not a.missing_info


def test_offline_answerer_missing_info_on_no_evidence():
    a = OfflineAnswerer().answer("what is the meaning of life?", [])
    assert a.missing_info and a.sources == []


def test_answerer_parses_valid_structured_response(tmp_path):
    (tmp_path / "img.png").write_bytes(b"png")
    evs = [EV_BASE[0]]
    evs[0].image_path = str(tmp_path / "img.png")  # evidence really has the slide image
    captured = {}

    def spy(messages, kwargs):
        captured.update(kwargs)
        content = messages[0]["content"]
        assert isinstance(content, list)  # image attached
        assert any(p["type"] == "image_url" for p in content)

    llm = FakeLLM([
        '{"answer": "You need a plan, not vibes.", '
        '"sources": [{"document": "Week 2.pptx", "page": 33, "section": "Vibe Coding on Prod", "excerpt": "vibe code a production-grade"}]}'
    ], on_messages=spy)

    a = Answerer(llm).answer("what does the meme say?", evs)
    assert a.answer == "You need a plan, not vibes."
    assert a.sources[0].page == 33
    assert a.sources[0].image_path == str(tmp_path / "img.png")
    assert captured["json_mode"] is True


def test_answerer_no_image_sends_text_only():
    evs = [EV_BASE[1]]

    def spy(messages, kwargs):
        assert isinstance(messages[0]["content"], str)  # no image -> plain text

    llm = FakeLLM(
        ['{"answer": "30%", "sources": [{"document": "Syllabus.pdf", "page": 2, "section": "Grading", "excerpt": "final exam is worth 30%"}]}'],
        on_messages=spy,
    )
    result = Answerer(llm).answer("how much is the final?", evs)
    assert result.answer == "30%"


def test_answerer_retries_once_then_succeeds():
    llm = FakeLLM([
        "not json at all",
        '{"answer": "It is a joke about deployment.", "sources": [{"document": "Week 2.pptx", "page": 33, "section": "", "excerpt": "vibe code"}]}',
    ])
    a = Answerer(llm).answer("the meme?", EV_BASE)
    assert llm.calls == 2
    assert "joke" in a.answer


def test_answerer_drops_unretrieved_citations():
    # cites a page that was never retrieved -> dropped, answer kept
    llm = FakeLLM([
        '{"answer": "Vibes.", "sources": [{"document": "Week 2.pptx", "page": 33, "section": "s", "excerpt": "e"}, {"document": "Ghost.pptx", "page": 99, "section": "s", "excerpt": "e"}]}'
    ])
    a = Answerer(llm).answer("q", EV_BASE)
    assert llm.calls == 1
    assert len(a.sources) == 1 and a.sources[0].page == 33


def test_answerer_raises_after_two_bad_responses():
    llm = FakeLLM(["garbage", "still garbage"])
    with pytest.raises(AnswerError):
        Answerer(llm).answer("q", EV_BASE)


def test_answerer_missing_info_without_calls():
    llm = FakeLLM([])
    a = Answerer(llm).answer("what is nothing?", [])
    assert a.missing_info and llm.calls == 0


def test_answerer_flags_missing_info_refusal():
    llm = FakeLLM(['{"answer": "The materials do not cover this.", "sources": []}'])
    a = Answerer(llm).answer("atomic clocks?", EV_BASE)
    assert a.missing_info and a.sources == []


def test_validate_answer_rules():
    ok, issues = validate_answer({'answer': 'x', 'sources': [
        {'document': 'Week 2.pptx', 'page': 33, 'excerpt': 'e'}]}, EV_BASE)
    assert ok is not None and not issues
    none, issues = validate_answer({'sources': []}, EV_BASE)
    assert none is None and issues
    ans, issues = validate_answer({'answer': 'x', 'sources': [
        {'document': 'Syllabus.pdf', 'page': 33}]}, EV_BASE)
    assert ans is not None and any("not in retrieved" in i for i in issues)


def test_answerer_no_sources_flagged_missing():
    llm = FakeLLM(['{"answer": "I could not find that anywhere.", "sources": []}'])
    a = Answerer(llm).answer("hard question?", EV_BASE)
    assert a.missing_info and a.sources == []


def test_answerer_sends_downscaled_images_for_slide_heavy_evidence(tmp_path):
    import base64 as _b64

    PNG_1PX = _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    img_a.write_bytes(PNG_1PX)
    img_b.write_bytes(PNG_1PX)
    evs = [
        Evidence(doc_id="d1", document="Week 2.pptx", page=33, section="Vibe",
                 text="One does not simply vibe code a production-grade enterprise app.",
                 image_path=str(img_a)),
        Evidence(doc_id="d1", document="Week 2.pptx", page=34, section="Speed",
                 text="Speculative decoding can double throughput in some settings.",
                 image_path=str(img_b)),
    ]
    seen = {}

    def spy(messages, kwargs):
        seen["content"] = messages[0]["content"]

    llm = FakeLLM([
        '{"answer": "Both pages.", "sources": [{"document": "Week 2.pptx", "page": 33, "section": "", "excerpt": "vibe code"}]}'
    ], on_messages=spy)
    Answerer(llm).answer("what do the slides say?", evs)

    content = seen["content"]
    assert isinstance(content, list) and len(content) == 3  # text + 2 images
    assert any(p["type"] == "image_url" for p in content[1:])
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in content[1:])
    assert "EVIDENCE IMAGES: 2 images" in content[0]["text"]
    assert "attached image 1" in content[0]["text"]
    assert "visual questions" in content[0]["text"].lower()


def test_parse_json_payload_lenient():
    assert parse_json_payload('```json\n{"answer": "a"}\n```') == {"answer": "a"}
    assert parse_json_payload('prefix {"answer": "a"} suffix') == {"answer": "a"}
    assert parse_json_payload("not json") is None
    assert parse_json_payload('["just", "a", "list"]') is None
