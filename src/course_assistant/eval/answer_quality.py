"""Answer-quality measurements shared by the repeatable RAG evaluation."""

from __future__ import annotations

import time
from typing import Protocol

from ..qa.schema import Answer
from ..retrieval.evidence import Evidence


class AnswererLike(Protocol):
    def answer(self, question: str, evidence: list[Evidence]) -> Answer: ...


def _expected_source_supported(spec: dict, answer: Answer) -> bool:
    """Check whether citations point to the expected course evidence.

    This measures citation targeting, not semantic entailment. Human reviewers
    record answer correctness separately in the evaluation artifact.
    """
    expected_doc = spec.get("doc")
    expected_page = spec.get("page")
    if expected_doc is None:
        return answer.missing_info and not answer.sources
    for source in answer.sources:
        if expected_doc.lower() not in source.document.lower():
            continue
        if expected_page is None or source.page == expected_page:
            return True
    return False


def evaluate_answer(
    spec: dict,
    evidence: list[Evidence],
    answerer: AnswererLike,
) -> dict:
    """Generate one answer and record support, timing, and review fields."""
    started = time.perf_counter()
    answer = answerer.answer(spec["question"], evidence)
    elapsed = round(time.perf_counter() - started, 4)
    return {
        "answer": answer.answer,
        "missing_info": answer.missing_info,
        "source_support": _expected_source_supported(spec, answer),
        "answer_correct": None,
        "answer_seconds": elapsed,
        "cited_documents": [source.document for source in answer.sources],
        "cited_pages": [source.page for source in answer.sources],
    }
