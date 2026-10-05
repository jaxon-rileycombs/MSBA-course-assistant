"""HTTP clients for the class services (dobolyi.com).

Role map (from the team's class-endpoints.txt, verified against live routes):
    9001  Vision LLM        base http://dobolyi.com:9001/v1 -> /chat/completions (OpenAI-compatible)
    9002  Text embeddings   base http://dobolyi.com:9002    -> /v1/embeddings
    9003  Visual embeddings base http://dobolyi.com:9003/v1 -> /embeddings
    9004  Multimodal rerank base http://dobolyi.com:9004    -> /v1/rerank
    9005  Document parsing  base http://dobolyi.com:9005/v1 -> /chat/completions (dots.mocr)

All services require an API key (unauthenticated calls return 401) sent as an
``Authorization: Bearer`` header.

Security: API keys NEVER appear in reprs, errors, or logs. Error messages are
sanitized so that any occurrence of the key is replaced with ``***``.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests

from .config import ServiceConfig


class ClassAPIError(RuntimeError):
    """A class-service call failed (network, HTTP status, or bad response shape)."""

    def __init__(self, service: str, status: Optional[int], message: str):
        super().__init__(f"[{service}] HTTP {status}: {message}")
        self.service = service
        self.status = status


def text_part(text: str) -> dict:
    """OpenAI-style text content part for multimodal chat messages."""
    return {"type": "text", "text": text}


def image_part(base64: str, mime: str = "image/png") -> dict:
    """OpenAI-style image content part (base64 data URL) for chat messages."""
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64}"}}


class _BaseClient:
    """Shared HTTP plumbing: auth header, retries, error sanitization."""

    path = ""  # endpoint path relative to the service base URL

    def __init__(
        self,
        svc: ServiceConfig,
        session: Optional[Any] = None,
        *,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self._svc = svc
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._retries = max(1, retries)

    @property
    def url(self) -> str:
        return self._svc.base_url + self.path

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._svc.api_key:
            headers["Authorization"] = f"Bearer {self._svc.api_key}"
        return headers

    def _post(self, payload: dict) -> dict:
        last: Optional[Any] = None
        for attempt in range(self._retries):
            resp = self._session.post(
                self.url, json=payload, headers=self._headers(), timeout=self._timeout
            )
            if resp.status_code in (408, 429) or resp.status_code >= 500:
                last = resp
                time.sleep(0.5 * (2**attempt))
                continue
            if resp.status_code >= 400:
                raise ClassAPIError(self._svc.name, resp.status_code, self._sanitize(resp.text or ""))
            try:
                return resp.json()
            except ValueError as exc:  # not JSON
                raise ClassAPIError(
                    self._svc.name, resp.status_code, f"invalid JSON response: {exc}"
                ) from exc
        raise ClassAPIError(
            self._svc.name, last.status_code if last is not None else None,
            "service unreachable after retries",
        )

    def _sanitize(self, text: str) -> str:
        if self._svc.api_key:
            text = text.replace(self._svc.api_key, "***")
        return text[:500]

    def __repr__(self) -> str:  # delegates to ServiceConfig's redacted repr
        return f"{type(self).__name__}({self._svc!r})"


class VisionLLMClient(_BaseClient):
    """Chat completions against the vision-capable LLM (9001, Qwen3.6-35B VLM)."""

    path = "/chat/completions"

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """Single-turn chat; ``messages`` may mix text_part/image_part content."""
        payload: dict[str, Any] = {
            "model": self._svc.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassAPIError(self._svc.name, None, f"unexpected chat response shape: {exc}") from exc


class EmbeddingsClient(_BaseClient):
    """Text or visual embeddings (9002 Nemotron / 9003 Qwen3-VL-Embedding)."""

    def __init__(self, svc: ServiceConfig, session: Optional[Any] = None, *, path: str = "/v1/embeddings", **kwargs):
        super().__init__(svc, session, **kwargs)
        self.path = path

    def embed(self, inputs: str | list[str]) -> list[list[float]]:
        """Embed one or more text inputs; returns (n, dim) row-major vectors."""
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            return []
        data = self._post({"model": self._svc.model, "input": inputs})
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise ClassAPIError(self._svc.name, None, f"unexpected embedding response shape: {exc}") from exc


class RerankClient(_BaseClient):
    """Multimodal reranking (9004, Qwen3-VL-Reranker-2B)."""

    path = "/v1/rerank"

    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[tuple[int, float]]:
        """Rank ``documents`` against ``query``; returns [(index, score), ...] in score order."""
        if not documents:
            return []
        payload: dict[str, Any] = {"model": self._svc.model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        data = self._post(payload)
        out: list[tuple[int, float]] = []
        for item in data.get("results", []):
            idx = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            if idx is not None and score is not None:
                out.append((int(idx), float(score)))
        if not out and documents:
            raise ClassAPIError(self._svc.name, None, "unexpected rerank response: no results")
        return out


class ParseClient(_BaseClient):
    """Document parsing (9005, dots.mocr) via multimodal chat completion.

    NOTE: the exact dots.mocr request format is validated in issue 2 against
    http://dobolyi.com:9005/docs and the Hugging Face page; this is the
    documented vLLM chat surface, and may need a service-specific tweak.
    """

    path = "/chat/completions"

    SYSTEM_PROMPT = (
        "Extract the complete content of the provided document as markdown: all text "
        "verbatim, tables, headings, layout order, and slide notes. Do not summarize "
        "and do not invent content that is not in the document."
    )

    def parse(self, parts: list[dict], *, max_tokens: int = 8192) -> str:
        """Parse a document given as chat content parts (page images + text)."""
        payload: dict[str, Any] = {
            "model": self._svc.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": parts},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassAPIError(self._svc.name, None, f"unexpected parse response shape: {exc}") from exc
