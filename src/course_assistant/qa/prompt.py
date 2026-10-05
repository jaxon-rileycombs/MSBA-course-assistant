"""Prompt template for grounded answering.

The model is explicitly told: use ONLY the evidence, quote verbatim in
``excerpt``, never invent citations, and say so when the materials cannot
answer. Combined with :func:`validate_answer` this is the never-invent
guardrail: any cited source that was not retrieved is dropped.
"""

from __future__ import annotations

from ..retrieval.evidence import Evidence

NOT_FOUND_TEXT = (
    "I could not find this in the course materials. The slides and syllabus "
    "I have access to do not cover this — please check the question or the "
    "selected material."
)


def build_prompt(question: str, evidence: list[Evidence], n_images: int = 0) -> str:
    lines = [
        "You are a course assistant for MBAX 6418. Answer the QUESTION using ONLY the EVIDENCE below.",
        "",
        "QUESTION:",
        question.strip(),
        "",
        "EVIDENCE (each entry: [document | page/slide | section]):",
    ]
    for i, ev in enumerate(evidence, 1):
        excerpt = " ".join(ev.text.split())
        if len(excerpt) > 800:
            excerpt = excerpt[:800] + " …"
        lines.append(f"{i}. [{ev.document} | slide {ev.page}" + (f" | {ev.section}" if ev.section else "") + f"] {excerpt}")
    if n_images:
        lines.append(
            "EVIDENCE IMAGES: " + ("an image" if n_images == 1 else f"{n_images} images")
            + " of the cited slides is attached after this text — read them; "
            "they may contain content the text extraction missed."
        )
    lines += [
        "",
        "INSTRUCTIONS:",
        "- If the EVIDENCE does not contain the answer, reply with sources: [] and say the materials do not cover this.",
        "- Reply with ONE JSON object, no extra text:",
        '  {"answer": "<concise answer>", "sources": [{"document": "<filename>", "page": <int>, "section": "<section>", "excerpt": "<verbatim quote from evidence>"}]}',
        "- The excerpt MUST be copied verbatim from the EVIDENCE text.",
        "- Cite ONLY documents and pages listed in EVIDENCE. Never invent sources.",
    ]
    return "\n".join(lines)
