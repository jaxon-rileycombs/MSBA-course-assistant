"""Validation of the model's structured answer.

Rules (assignment: "validate the format and check that sources support the
answer"):
1. response must parse as JSON with a non-empty string ``answer``
2. every cited source must exist in the evidence actually retrieved
   (a citation to anything else is dropped and recorded as an issue)
3. an answer that ends up citing nothing is flagged ``missing_info`` —
   a grounded answer must cite, so no sources means the materials were
   not used (or not found)
"""

from __future__ import annotations

import json
import re

from ..retrieval.evidence import Evidence
from .schema import Answer, SourceRef


def parse_json_payload(text: str) -> dict | None:
    """Lenient JSON extraction: accepts raw JSON or fenced/code-wrapped text."""
    if not text or not text.strip():
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def validate_answer(payload: dict | None, evidence: list[Evidence]) -> tuple[Answer | None, list[str]]:
    """Return (Answer, issues). Answer is None when the payload is unusable."""
    issues: list[str] = []
    if not isinstance(payload, dict):
        return None, ["response was not valid JSON"]
    answer_text = payload.get("answer")
    if not isinstance(answer_text, str) or not answer_text.strip():
        return None, ["missing non-empty 'answer' field"]

    sources_raw = payload.get("sources", [])
    if not isinstance(sources_raw, list):
        issues.append("'sources' was not a list")
        sources_raw = []

    allowed = {(e.document, e.page) for e in evidence}
    evidence_by_key = {(e.document, e.page): e for e in evidence}
    refs: list[SourceRef] = []
    for s in sources_raw:
        if not isinstance(s, dict):
            issues.append("dropped a malformed source entry")
            continue
        doc, page = s.get("document"), s.get("page")
        if not isinstance(doc, str) or not isinstance(page, int):
            issues.append(f"dropped source with invalid fields: {s!r}")
            continue
        if (doc, page) not in allowed:
            issues.append(f"dropped citation ({doc}, page {page}) not in retrieved evidence")
            continue
        refs.append(
            SourceRef(
                document=doc,
                page=page,
                section=str(s.get("section", "")),
                excerpt=str(s.get("excerpt", "")),
                image_path=evidence_by_key[(doc, page)].image_path,
            )
        )

    missing = not refs
    if not refs:
        issues.append("answer cites no sources — flagged as missing information")

    return Answer(answer=answer_text.strip(), sources=refs, missing_info=missing), issues
