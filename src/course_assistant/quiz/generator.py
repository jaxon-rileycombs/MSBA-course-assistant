"""Quiz generation, grounded in the selected course material.

- :class:`QuizGenerator` — asks the vision LLM (9001) to write MCQs from
  the supplied evidence chunks; validates the structured JSON (sources must
  be among the retrieved evidence, options well-formed), retrying once.
  The answer key is fixed at generation and stored by :class:`QuizSession`.
- :class:`OfflineQuizGenerator` — deterministic placeholder questions from
  the top evidence chunk so the UI/e2e/tests run before the keys arrive.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict

from ..qa.validate import parse_json_payload
from ..retrieval.evidence import Evidence
from .schema import Question


def _extract_questions(payload: dict | None) -> list[dict] | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("quiz")
    return raw if isinstance(raw, list) else (payload.get("questions") if isinstance(payload.get("questions"), list) else None)


def validate_questions(raw: list, allowed: set[tuple[str, int]]) -> list[Question]:
    """Keep only well-formed questions whose sources were actually supplied."""
    questions: list[Question] = []
    for q in raw:
        if not isinstance(q, dict):
            continue
        text = q.get("question")
        options = q.get("options")
        key = q.get("answer_index")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, str) and o for o in options):
            continue
        if not isinstance(key, int) or not 0 <= key < len(options):
            continue
        sources = []
        for s in q.get("sources", []) or []:
            if isinstance(s, dict) and isinstance(s.get("document"), str) and isinstance(s.get("page"), int):
                if (s["document"], s["page"]) in allowed:
                    sources.append({"document": s["document"], "page": s["page"]})
        questions.append(
            Question(
                question=text.strip(),
                options=[o.strip() for o in options[:4]],
                answer_index=key,
                explanation=str(q.get("explanation", "")).strip(),
                sources=sources,
            )
        )
    return questions


QUIZ_PROMPT = """You are a course assistant for MBAX 6418. Using ONLY the EVIDENCE below, write {n} multiple-choice quiz questions that ARE answerable from this material, one per evidence entry.

EVIDENCE (each: [document | slide | section] text):
{evidence}

Reply with ONE JSON object, no extra text:
{{"quiz": [{{"question": "...", "options": ["A", "B", "C", "D"], "answer_index": 0, "explanation": "...", "sources": [{{"document": "<filename>", "page": <int>}}]}}]}}

Rules:
- Exactly {n} questions; four options each; one clearly correct answer.
- answer_index is the position of the correct option (0-3).
- explanation must say WHY and reference the material.
- sources must only use documents/pages from EVIDENCE."""


class QuizGenerator:
    def __init__(self, client, *, n: int = 5, max_tokens: int = 2048):
        self.client = client
        self.n = n
        self.max_tokens = max_tokens

    def generate(self, chunks: list[Evidence], n: int | None = None) -> list[Question]:
        """``chunks`` = evidence sampled across the selected material.

        Returns EXACTLY the requested number of questions when the model
        cooperates (never more): one call at temperature 0, then a second
        attempt at 0.2 if the first came back short, then a clear error
        rather than an under-filled quiz.
        """
        want = min(n or self.n, len(chunks))
        if not chunks or want < 1:
            return []
        allowed = {(e.document, e.page) for e in chunks}
        evidence_txt = "\n".join(
            f"{i}. [{e.document} | slide {e.page}" + (f" | {e.section}" if e.section else "") + f"] {e.text[:500]}"
            for i, e in enumerate(chunks[:want], 1)
        )
        questions: list[Question] = []
        for attempt in range(2):
            prompt = QUIZ_PROMPT.format(n=want, evidence=evidence_txt)
            raw = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 if attempt == 0 else 0.2,
                max_tokens=self.max_tokens,
                json_mode=True,
            )
            extracted = _extract_questions(parse_json_payload(raw))
            questions = validate_questions(extracted if extracted is not None else [], allowed)
            if len(questions) >= want:
                break
        if len(questions) < min(2, want):
            raise RuntimeError(
                f"Could not generate {want} valid grounded questions from this material — please try again."
            )
        return questions[:want]  # exact count, never more


class OfflineQuizGenerator:
    """Deterministic placeholder quiz (demo/tests only) — no model."""

    def generate(self, chunks: list[Evidence], n: int | None = None) -> list[Question]:
        want = min(n or 5, len(chunks))
        questions = []
        for e in chunks[:want]:
            options = [e.section or e.document, "The syllabus", "A random topic", "None of the above"]
            questions.append(
                Question(
                    question=f"Per the materials, what is the topic shown on {e.document}, slide {e.page}?",
                    options=options,
                    answer_index=0,
                    explanation=f"The material for this slide is titled “{e.section}”. "
                                f"Full text: {e.text[:300]}",
                    sources=[{"document": e.document, "page": e.page}],
                )
            )
        return questions
