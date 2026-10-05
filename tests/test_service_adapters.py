"""Adapter + client tests with fake transports (no network, no keys).

The fakes override ``_post`` (HTTP layer), so the REAL request-building
code runs: data-URL payloads, batching, index mapping.
"""

import base64
from pathlib import Path

from course_assistant.clients import EmbeddingsClient, RerankClient
from course_assistant.config import ServiceConfig
from course_assistant.retrieval.embeddings_service import (
    TextEmbedderAdapter,
    VisualEmbedderAdapter,
)
from course_assistant.retrieval.rerank import ServiceReranker

DIM = 4

_FAKE_SVC = ServiceConfig(name="fake", base_url="http://fake", model="fake-model", api_key="k")


class FakeEmbeddingsClient(EmbeddingsClient):
    def __init__(self):
        super().__init__(_FAKE_SVC, session=object())
        self.payloads = []

    def _post(self, payload, *args, **kwargs):
        self.payloads.append(payload)
        n = len(payload.get("input", []))
        return {"data": [{"embedding": [0.1] * DIM} for _ in range(n)]}


class FakeRerankClient(RerankClient):
    def __init__(self):
        super().__init__(_FAKE_SVC, session=object())
        self._base = 0

    def _post(self, payload, *args, **kwargs):
        docs = payload["documents"]
        n = len(docs)
        results = [
            {"index": i, "relevance_score": 1.0 / (self._base + i + 1)} for i in range(n)
        ]
        self._base += n  # global position, as a real service would score
        return {"results": results}


def test_embed_images_builds_data_url_payload(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG fake")
    client = FakeEmbeddingsClient()

    out = client.embed_images([str(img)])
    assert len(out) == 1 and len(out[0]) == DIM
    url = client.payloads[-1]["input"][0]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == b"\x89PNG fake"
    # one data-URL string per image, completion-style input
    assert len(client.payloads[-1]["input"]) == 1


def test_text_embedder_adapter_batches():
    client = FakeEmbeddingsClient()
    adapter = TextEmbedderAdapter(client, batch_size=2)
    vectors = adapter.embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5 and all(len(v) == DIM for v in vectors)
    batch_sizes = [len(p["input"]) for p in client.payloads]
    assert batch_sizes == [2, 2, 1]


def test_url_never_duplicates_v1_segment():
    # Base URLs in .env may already end in /v1 while the client path also
    # carries /v1 (9003 visual embeddings). http://host:9003/v1 + /v1/embeddings
    # → 404; the dedupe makes it http://host:9003/v1/embeddings.
    from course_assistant.clients import EmbeddingsClient, RerankClient
    from course_assistant.config import ServiceConfig

    def cfg(base):
        return ServiceConfig(name="x", base_url=base, api_key="k", model="m")

    cases = [
        (EmbeddingsClient(cfg("http://h:9003/v1")), "http://h:9003/v1/embeddings"),
        (EmbeddingsClient(cfg("http://h:9002")), "http://h:9002/v1/embeddings"),
        (RerankClient(cfg("http://h:9004")), "http://h:9004/v1/rerank"),
        (RerankClient(cfg("http://h:9004/v1")), "http://h:9004/v1/rerank"),
    ]
    for client, expected in cases:
        assert client.url == expected, (client.url, expected)
        assert "/v1/v1" not in client.url


def test_visual_embedder_adapter_modes(tmp_path):
    client = FakeEmbeddingsClient()
    adapter = VisualEmbedderAdapter(client)
    paths = []
    for i in range(5):
        f = tmp_path / f"p{i}.png"
        f.write_bytes(b"png-bytes")
        paths.append(str(f))
    vectors = adapter.embed_images(paths)
    assert len(vectors) == 5 and all(v == [0.1] * DIM for v in vectors)
    # image payloads are data-URL string lists, batched at the service cap
    inputs = [p["input"] for p in client.payloads]
    assert [len(i) for i in inputs] == [4, 1]
    assert all(isinstance(part, str) and part.startswith("data:image/") for part in inputs[0])

    client.payloads.clear()
    assert adapter.embed_query("question?") == [0.1] * DIM
    assert client.payloads[-1]["input"] == ["question?"]


def test_visual_batches_respect_input_size_cap(tmp_path):
    client = FakeEmbeddingsClient()
    adapter = VisualEmbedderAdapter(client)
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 300_000)
    small = tmp_path / "small.png"
    small.write_bytes(b"x" * 10_000)
    files = [str(big), str(small), str(small), str(small), str(big), str(small)]
    batches = adapter._batches(files)
    assert all(len(b) <= 4 for b in batches), [len(b) for b in batches]
    for b in batches:
        chars = sum(Path(p).stat().st_size for p in b) * 4 / 3
        assert chars <= 700_000, chars
    # 6 files at count-4 -> at least 2 batches; every path goes out once
    assert set(p for b in batches for p in b) == set(files)


def test_service_reranker_maps_scores_to_original_order():
    r = ServiceReranker(FakeRerankClient(), batch_size=2)
    scores = r("q", ["a", "b", "c", "d", "e"])
    assert len(scores) == 5
    assert scores[0] > scores[1] > scores[2] > scores[3] > scores[4]


def test_service_reranker_empty_input():
    assert ServiceReranker(FakeRerankClient())("q", []) == []
