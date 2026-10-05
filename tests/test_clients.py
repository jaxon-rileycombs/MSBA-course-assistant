"""Tests for the class-service HTTP clients (mocked transport, no network)."""

from typing import Optional

import pytest

from course_assistant.clients import (
    ClassAPIError,
    EmbeddingsClient,
    ParseClient,
    RerankClient,
    VisionLLMClient,
    image_part,
    text_part,
)
from course_assistant.config import Config

ENV_OK = {
    "CLASS_VISION_LLM_BASE_URL": "http://dobolyi.com:9001/v1",
    "CLASS_VISION_LLM_API_KEY": "test-key-vision",
    "CLASS_VISION_LLM_MODEL": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "CLASS_TEXT_EMBED_BASE_URL": "http://dobolyi.com:9002",
    "CLASS_TEXT_EMBED_API_KEY": "test-key-text",
    "CLASS_TEXT_EMBED_MODEL": "nvidia/Nemotron-3-Embed-1B-BF16",
    "CLASS_VISUAL_EMBED_BASE_URL": "http://dobolyi.com:9003/v1",
    "CLASS_VISUAL_EMBED_API_KEY": "test-key-visual",
    "CLASS_VISUAL_EMBED_MODEL": "Qwen/Qwen3-VL-Embedding-2B",
    "CLASS_RERANK_BASE_URL": "http://dobolyi.com:9004",
    "CLASS_RERANK_API_KEY": "test-key-rerank",
    "CLASS_RERANK_MODEL": "Qwen/Qwen3-VL-Reranker-2B",
    "CLASS_PARSE_BASE_URL": "http://dobolyi.com:9005/v1",
    "CLASS_PARSE_API_KEY": "test-key-parse",
    "CLASS_PARSE_MODEL": "dots.mocr",
}

cfg = Config.from_env(ENV_OK)


class FakeResponse:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records calls; returns queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers, timeout))
        return self.responses.pop(0)


def chat_response(content: str) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_urls_match_documented_role_map():
    s = FakeSession([chat_response("hi")])
    VisionLLMClient(cfg.vision_llm, session=s).chat([{"role": "user", "content": "hi"}])
    assert s.calls[0][0] == "http://dobolyi.com:9001/v1/chat/completions"

    s = FakeSession([FakeResponse(200, {"data": [{"embedding": [0.1, 0.2]}]})])
    EmbeddingsClient(cfg.text_embed, session=s).embed("x")
    assert s.calls[0][0] == "http://dobolyi.com:9002/v1/embeddings"

    s = FakeSession([FakeResponse(200, {"data": [{"embedding": [0.1]}]})])
    EmbeddingsClient(cfg.visual_embed, session=s, path="/embeddings").embed("x")
    assert s.calls[0][0] == "http://dobolyi.com:9003/v1/embeddings"

    s = FakeSession([FakeResponse(200, {"results": [{"index": 0, "relevance_score": 0.9}]})])
    RerankClient(cfg.rerank, session=s).rerank("q", ["d"])
    assert s.calls[0][0] == "http://dobolyi.com:9004/v1/rerank"

    s = FakeSession([chat_response("md")])
    ParseClient(cfg.parse, session=s).parse([text_part("p")])
    assert s.calls[0][0] == "http://dobolyi.com:9005/v1/chat/completions"


def test_auth_header_and_payload():
    s = FakeSession([chat_response("answer")])
    VisionLLMClient(cfg.vision_llm, session=s).chat([{"role": "user", "content": "q"}], temperature=0.2, max_tokens=64)
    url, payload, headers, _ = s.calls[0]
    assert headers["Authorization"] == "Bearer test-key-vision"
    assert payload["model"] == "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64


def test_no_auth_header_when_key_empty():
    env = dict(ENV_OK)
    env["CLASS_VISION_LLM_API_KEY"] = ""
    c = VisionLLMClient(Config.from_env(env).vision_llm, session=FakeSession([chat_response("x")]))
    c.chat([{"role": "user", "content": "q"}])
    # second call from a fresh session; inspect headers via new call
    s = FakeSession([chat_response("x")])
    c2 = VisionLLMClient(Config.from_env(env).vision_llm, session=s)
    c2.chat([{"role": "user", "content": "q"}])
    assert "Authorization" not in s.calls[0][2]


def test_multimodal_message_parts():
    part = image_part("QUJD", mime="image/png")
    assert part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    assert text_part("hello") == {"type": "text", "text": "hello"}


def test_embeddings_return_row_major_vectors():
    payload = {"data": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]}
    s = FakeSession([FakeResponse(200, payload)])
    out = EmbeddingsClient(cfg.text_embed, session=s).embed(["a", "b"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]
    assert s.calls[0][1]["input"] == ["a", "b"]


def test_embeddings_empty_input_short_circuit():
    s = FakeSession([])
    assert EmbeddingsClient(cfg.text_embed, session=s).embed([]) == []
    assert s.calls == []


def test_rerank_parses_and_preserves_order():
    payload = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
    s = FakeSession([FakeResponse(200, payload)])
    out = RerankClient(cfg.rerank, session=s).rerank("q", ["a", "b"], top_n=2)
    assert out == [(1, 0.9), (0, 0.1)]
    assert s.calls[0][1]["query"] == "q"
    assert s.calls[0][1]["documents"] == ["a", "b"]
    assert s.calls[0][1]["top_n"] == 2


def test_rerank_empty_documents_short_circuit():
    s = FakeSession([])
    assert RerankClient(cfg.rerank, session=s).rerank("q", []) == []
    assert s.calls == []


def test_json_mode_sets_response_format():
    s = FakeSession([chat_response('{"k": 1}')])
    VisionLLMClient(cfg.vision_llm, session=s).chat([{"role": "user", "content": "q"}], json_mode=True)
    assert s.calls[0][1]["response_format"] == {"type": "json_object"}


def test_http_error_raises_with_status_and_sanitized_message():
    body = '{"error": "invalid api key test-key-vision rejected"}'
    s = FakeSession([FakeResponse(401, text=body)])
    with pytest.raises(ClassAPIError) as ei:
        VisionLLMClient(cfg.vision_llm, session=s).chat([{"role": "user", "content": "q"}])
    msg = str(ei.value)
    assert ei.value.status == 401
    assert "test-key-vision" not in msg  # key stripped from the error
    assert "***" in msg


def test_retry_then_success():
    s = FakeSession([FakeResponse(503, text="overloaded"), chat_response("ok")])
    c = VisionLLMClient(cfg.vision_llm, session=s, retries=2)
    assert c.chat([{"role": "user", "content": "q"}]) == "ok"
    assert len(s.calls) == 2


def test_retries_exhausted_raises_last_status():
    s = FakeSession([FakeResponse(503, text="a"), FakeResponse(503, text="b"), FakeResponse(503, text="c")])
    c = VisionLLMClient(cfg.vision_llm, session=s, retries=3)
    with pytest.raises(ClassAPIError) as ei:
        c.chat([{"role": "user", "content": "q"}])
    assert ei.value.status == 503
    assert len(s.calls) == 3


def test_non_json_success_body_raises():
    s = FakeSession([FakeResponse(200, text="<html>")])
    with pytest.raises(ClassAPIError, match="invalid JSON"):
        EmbeddingsClient(cfg.text_embed, session=s).embed("x")


def test_bad_response_shape_raises_clean_error():
    s = FakeSession([FakeResponse(200, {"unexpected": True})])
    with pytest.raises(ClassAPIError):
        VisionLLMClient(cfg.vision_llm, session=s).chat([{"role": "user", "content": "q"}])


def test_repr_never_contains_key():
    for svc in cfg.services:
        assert svc.api_key not in repr(svc)
    assert "test-key-vision" not in repr(VisionLLMClient(cfg.vision_llm))
