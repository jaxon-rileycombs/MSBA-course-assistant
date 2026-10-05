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
from pathlib import Path
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


def chat_image_part(image_path: str, max_side: int = 512) -> dict | None:
    """Resized image part for chat.

    Slide renders are multi-MB at full size; a raw base64 data URL can push
    the gateway's request-size limit. Downscale to ``max_side`` pixels and
    re-encode as JPEG. Returns None when the file can't be read/resized —
    callers then fall back to the raw bytes or skip the image.
    """
    try:
        import base64 as _b64
        import io as _io

        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                im = im.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
                )
            buf = _io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=82)
        return image_part(_b64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg")
    except Exception:
        return None


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
        base = self._svc.base_url.rstrip("/")
        # Some services document BASE_URL with a trailing /v1 and ALSO expect
        # the path to carry /v1 (e.g. http://host:9003/v1 + /v1/embeddings).
        # Deduplicate so we never post to /v1/v1/... (404).
        if base.endswith("/v1") and self.path.startswith("/v1"):
            base = base[:-3]
        return base + self.path

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._svc.api_key:
            headers["Authorization"] = f"Bearer {self._svc.api_key}"
        return headers

    def _post(self, payload: dict) -> dict:
        last: Optional[Any] = None
        for attempt in range(self._retries):
            try:
                resp = self._session.post(
                    self.url, json=payload, headers=self._headers(), timeout=self._timeout
                )
            except requests.RequestException as exc:  # network-level failure
                last = exc
                time.sleep(0.5 * (2**attempt))
                continue
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
        status = getattr(last, "status_code", None) if last is not None else None
        raise ClassAPIError(
            self._svc.name, status,
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
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Single-turn chat; ``messages`` may mix text_part/image_part content.

        The served model is a reasoning model (Qwen3.6-35B): thinking goes to
        ``message.reasoning`` and small budgets leave ``content`` null. We
        disable thinking via the vLLM chat-template knob so answers arrive
        deterministic and fast (verified live against 9001).
        """
        payload: dict[str, Any] = {
            "model": self._svc.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassAPIError(self._svc.name, None, f"unexpected chat response shape: {exc}") from exc
        if not content:  # null/empty even after retries — never fabricate
            raise ClassAPIError(
                self._svc.name, None,
                "model returned an empty completion (no reasoning visible)",
            )
        return content


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
        return self._vectors(data)

    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        """Embed image files (9003 Qwen3-VL-Embedding) via base64 data URLs.

        Qwen3-VL-Embedding mixes text and images in one space, so queries use
        :meth:`embed` and page/slide images use this. The completion-style
        ``input`` of base64 data URLs is the documented vLLM multimodal
        format (confirmed against the live service's /docs).
        """
        import base64
        import mimetypes
        from pathlib import Path

        if not image_paths:
            return []
        content = []
        for p in image_paths:
            b64 = base64.b64encode(self._read_embeddable(p)).decode("ascii")
            mime = mimetypes.guess_type(p)[0] or "image/png"
            content.append(f"data:{mime};base64,{b64}")
        data = self._post({"model": self._svc.model, "input": content})
        return self._vectors(data)

    @staticmethod
    def _read_embeddable(path: str) -> bytes:
        """Return image bytes small enough for the embedding API.

        The service caps input at 8192 tokens, and the bound is effectively
        IMAGE PATCHES: the vision encoder re-tiles detailed images, so a
        downscaled photo can fail at ANY resolution (measured: 64px photos
        still fail — 78/250 images fail at 256px, 35 at 128px). We therefore
        downscale any image whose longest side exceeds 128px to 128px via
        PyMuPDF — a good retrieval-quality/size trade-off (86% of the
        corpus embeds at 128px) — and callers skip images the service still
        rejects. Originals are never modified and stay full-resolution for
        display.
        """
        path_obj = Path(path)
        raw = path_obj.read_bytes()
        px = EmbeddingsClient._longest_side_px(path_obj)
        known_small = px > 0 and px <= 128
        too_byte_heavy = len(raw) > 600_000  # base64 would exceed the 1M-char bound
        if known_small and not too_byte_heavy:
            return raw
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            page = doc[0]
            w, h = page.rect.width, page.rect.height
            scale = min(1.0, 128.0 / max(w, h))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            out = pix.tobytes("png")
            doc.close()
            return out
        except Exception:
            return raw  # fall back to full size; better one oversize attempt

    @staticmethod
    def _longest_side_px(path_obj: Path) -> int:
        """Read width/height from the PNG/JPEG/GIF header without decoding
        (0 = unknown)."""
        try:
            import struct

            head = path_obj.read_bytes()[:4096]
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return max(w, h)
            if head[:3] == b"\xff\xd8\xff":  # JPEG: scan for SOF0/SOF2
                pos, n = 2, len(head)
                while pos + 9 < n:
                    if head[pos] != 0xFF:
                        pos += 1
                        continue
                    marker = head[pos + 1]
                    if marker in (0xC0, 0xC2):
                        h, w = struct.unpack(">HH", head[pos + 5 : pos + 9])
                        return max(w, h)
                    pos += 2
            if head[:6] in (b"GIF87a", b"GIF89a"):
                w, h = struct.unpack("<HH", head[6:10])
                return max(w, h)
        except Exception:
            pass
        return 0  # unknown -> caller routes through the downscale path

    @staticmethod
    def _vectors(data: dict) -> list[list[float]]:
        try:
            return [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise ClassAPIError(
                data.get("model", "embedding"), None, f"unexpected embedding response shape: {exc}"
            ) from exc


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
