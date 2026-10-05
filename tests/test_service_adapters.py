"""Adapter + client tests with fake transports (no network, no keys).

The fakes override ``_post`` (HTTP layer), so the REAL request-building
code runs: data-URL payloads, batching, index mapping.
"""

import base64

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
    part = client.payloads[-1]["input"][0]
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(url.split(",", 1)[1])
    assert decoded == b"\x89PNG fake"


def test_text_embedder_adapter_batches():
    client = FakeEmbeddingsClient()
    adapter = TextEmbedderAdapter(client, batch_size=2)
    vectors = adapter.embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5 and all(len(v) == DIM for v in vectors)
    batch_sizes = [len(p["input"]) for p in client.payloads]
    assert batch_sizes == [2, 2, 1]


def test_visual_embedder_adapter_modes(tmp_path):
    client = FakeEmbeddingsClient()
    adapter = VisualEmbedderAdapter(client, batch_size=2)
    paths = []
    for i in range(5):
        f = tmp_path / f"p{i}.png"
        f.write_bytes(b"png-bytes")
        paths.append(str(f))
    vectors = adapter.embed_images(paths)
    assert len(vectors) == 5 and all(v == [0.1] * DIM for v in vectors)
    # image payloads are content-part lists, batched by 2
    inputs = [p["input"] for p in client.payloads]
    assert [len(i) for i in inputs] == [2, 2, 1]
    assert all(isinstance(part, dict) for part in inputs[0])

    client.payloads.clear()
    assert adapter.embed_query("question?") == [0.1] * DIM
    assert client.payloads[-1]["input"] == ["question?"]


def test_service_reranker_maps_scores_to_original_order():
    r = ServiceReranker(FakeRerankClient(), batch_size=2)
    scores = r("q", ["a", "b", "c", "d", "e"])
    assert len(scores) == 5
    assert scores[0] > scores[1] > scores[2] > scores[3] > scores[4]


def test_service_reranker_empty_input():
    assert ServiceReranker(FakeRerankClient())("q", []) == []
