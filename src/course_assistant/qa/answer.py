"""Grounded answering.

- :class:`Answerer` — calls the class vision LLM (9001) with the retrieved
  evidence (attaching the top slide image for visual questions), asks for a
  structured JSON answer, validates format + source support, and retries
  once with a corrective instruction before giving up.
- :class:`OfflineAnswerer` — deterministic answers from the top evidence;
  lets the whole app (and demo screenshots) run before the team's API keys
  arrive. Never claims to be the real model.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..clients import chat_image_part, image_part, text_part
from ..retrieval.evidence import Evidence
from .prompt import NOT_FOUND_TEXT, build_prompt
from .schema import Answer, SourceRef
from .validate import parse_json_payload, validate_answer


class AnswerError(RuntimeError):
    """Raised (and surfaced) when the LLM cannot produce a valid answer."""


class Answerer:
    def __init__(self, client, *, max_evidence: int = 16, max_images: int = 3,
                 max_tokens: int = 1024):
        self.client = client
        self.max_evidence = max_evidence
        self.max_images = max_images
        self.max_tokens = max_tokens

    def answer(self, question: str, evidence: list[Evidence]) -> Answer:
        evs = evidence[: self.max_evidence]
        if not evs:
            return Answer(answer=NOT_FOUND_TEXT, sources=[], missing_info=True)
        seen: set[str] = set()
        images: list[str] = []
        for e in evs:
            if e.image_path and e.image_path not in seen:
                seen.add(e.image_path)
                images.append(e.image_path)
            if len(images) >= self.max_images:
                break
        return self._attempt(question, evs, images, repair=False)

    def _attempt(self, question: str, evs: list[Evidence], images: list[str],
                 repair: bool) -> Answer:
        messages = self._messages(question, evs, images, repair)
        raw = self.client.chat(messages=messages, temperature=0.0,
                               max_tokens=self.max_tokens, json_mode=True)
        payload = parse_json_payload(raw)
        answer, issues = validate_answer(payload, evs)
        if answer is not None:
            return answer
        if not repair:
            fix = (
                "Your previous response was invalid: "
                + ("; ".join(issues) if issues else "it was not valid JSON")
                + ". Reply with ONLY the required JSON object now."
            )
            try:
                return self._attempt(question, evs, images, repair=True)
            except AnswerError:
                raise AnswerError(
                    "The model could not produce a valid, sourced answer. "
                    "This is surfaced instead of showing an unvalidated response.",
                ) from None
        raise AnswerError("model failed structured output twice")

    def _messages(self, question: str, evs: list[Evidence], images: list[str],
                  repair: bool) -> list[dict]:
        prompt = build_prompt(question, evs, n_images=len(images))
        if repair:
            prompt = "Repair your JSON per the instruction. " + prompt
        content: list[dict] | str
        if images:
            content = [text_part(prompt)]
            for path in images:
                part = chat_image_part(path)  # downscaled; None if unreadable
                if part is None:
                    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                    part = image_part(b64, mimetypes.guess_type(path)[0] or "image/png")
                content.append(part)
        else:
            content = prompt
        return [{"role": "user", "content": content}]


class OfflineAnswerer:
    """Deterministic stand-in (demo/tests only) — does NOT call any model."""

    def answer(self, question: str, evidence: list[Evidence]) -> Answer:
        evs = evidence or []
        if not evs:
            return Answer(answer=NOT_FOUND_TEXT, sources=[], missing_info=True)
        top = evs[0]
        excerpt = " ".join(top.text.split())[:600]
        where = f"{top.document}, slide {top.page}" + (f" ({top.section})" if top.section else "")
        answer_text = f"Based on {where}: {excerpt}"
        return Answer(
            answer=answer_text,
            sources=[SourceRef(top.document, top.page, top.section, excerpt, top.image_path)],
        )
