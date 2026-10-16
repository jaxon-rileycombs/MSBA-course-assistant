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
import random
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


def _sig_tokens(text: str) -> set[str]:
    """Significant tokens (>=4 alnum chars) — used for leak/grounding checks."""
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


BANNED_QUESTION = re.compile(
    r"\bweek\s*\d|\bslides?\b|\bdeck\b|according to|course material", re.I
)


def _strip_option_prefix(o: str) -> str:
    """Drop a leading 'A.' / 'B)' label the model sometimes adds."""
    return re.sub(r"^\s*[A-D][.):]\s*", "", o).strip()


def _question_leaks_answer(qtext: str, key: str) -> bool:
    return len(_sig_tokens(qtext) & _sig_tokens(key)) >= 2


def _grounded_in(key: str, explanation: str, cited_text: str) -> bool:
    """The question is tied to the cited material when the correct option OR
    the explanation overlaps it: paraphrased answers are fine as long as the
    explanation restates the material (>=2 significant shared tokens or a
    6-token verbatim phrase)."""
    if not cited_text:
        return True  # no verifiable source -> keep (validated elsewhere)
    cited_toks = _sig_tokens(cited_text)
    if _sig_tokens(key) & cited_toks:
        return True  # verbatim answer: strongest grounding
    expl_toks = _sig_tokens(explanation)
    if len(expl_toks & cited_toks) >= 2:
        return True  # explanation restates the material
    expl = re.findall(r"[a-z0-9]{3,}", explanation.lower())
    cited = " ".join(re.findall(r"[a-z0-9]{3,}", cited_text.lower()))
    return any(" ".join(expl[i : i + 6]) in cited for i in range(len(expl) - 5))


def _shuffle_key(options: list[str], key: int, seed_text: str) -> tuple[list[str], int]:
    """Randomize the correct option's position (deterministic per question
    text) so the key never sits in a predictable slot."""
    positions = list(range(len(options)))
    random.Random(seed_text.strip()).shuffle(positions)
    new_key = positions.index(key)
    return [options[i] for i in positions], new_key


def validate_questions(raw: list, evidence) -> list[Question]:
    """Keep only well-formed questions that are strictly grounded.

    ``evidence`` is the retrieved :class:`Evidence` list (for grounding
    checks); a plain set of (document, page) is also accepted (grounding
    checks are skipped). Filtered out: questions whose text references the
    deck/week/slide, questions that give the answer away, duplicates,
    weakly-grounded questions, and anything with an unretrieved source.
    The remaining questions get their key position shuffled.
    """
    evidence_list = [e for e in evidence] if not isinstance(evidence, set) else []
    allowed = (
        {(e.document, e.page) for e in evidence_list}
        if evidence_list
        else set(evidence or {})
    )
    evidence_by_page = {(e.document, e.page): e for e in evidence_list}
    text_by_page = {key: item.text for key, item in evidence_by_page.items()}
    seen: set[str] = set()
    used_pages: set[tuple[str, int]] = set()  # one question per slide
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
        text = text.strip()
        options = [_strip_option_prefix(o) for o in options[:4]]

        # stand-alone question: never reference the deck/week/slide
        if BANNED_QUESTION.search(text):
            continue
        # no duplicates (same question asked twice)
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        if norm in seen:
            continue
        # the question must not give the answer away
        if _question_leaks_answer(text, options[key]):
            continue
        # sources must be among the supplied evidence and carry the excerpt,
        # section, and original image needed for visible quiz feedback.
        sources = []
        for s in q.get("sources", []) or []:
            if isinstance(s, dict) and isinstance(s.get("document"), str) and isinstance(s.get("page"), int):
                source_key = (s["document"], s["page"])
                if source_key in allowed:
                    item = evidence_by_page.get(source_key)
                    sources.append({
                        "document": s["document"],
                        "page": s["page"],
                        "section": item.section if item else "",
                        "excerpt": item.text[:700] if item else "",
                        "image_path": item.image_path if item else None,
                    })
        if evidence_list and not sources:
            continue
        # strictly tied to the cited material
        if sources and not _grounded_in(
            options[key], str(q.get("explanation", "")), text_by_page.get((sources[0]["document"], sources[0]["page"]), "")
        ):
            continue
        # a quiz covers the deck: no two questions from the same slide
        if sources and (sources[0]["document"], sources[0]["page"]) in used_pages:
            continue

        seen.add(norm)
        if sources:
            used_pages.add((sources[0]["document"], sources[0]["page"]))
        options, key = _shuffle_key(options, key, text)
        questions.append(
            Question(
                question=text,
                options=options,
                answer_index=key,
                explanation=str(q.get("explanation", "")).strip(),
                sources=sources,
            )
        )
    return questions


QUIZ_PROMPT = """You are an exam writer for MBAX 6418. Write {n} multiple-choice quiz questions, each answerable from EXACTLY ONE entry of the EVIDENCE below. Use a different entry per question when possible.

EVIDENCE (each entry: [N] material text):
{evidence}

Reply with ONE JSON object, no extra text:
{{"quiz": [{{"question": "...", "options": ["opt1", "opt2", "opt3", "opt4"], "answer_index": 0, "explanation": "...", "sources": [{{"document": "<filename>", "page": <int>}}]}}]}}

Hard rules:
- Ask the CONCEPT directly. Never mention week numbers, deck names, slide numbers, sections, or use openers like "According to the course materials".
- The question text must NEVER contain or hint at the correct option. Write the question before the options.
- Options: exactly four, plain text (no "A." / "1)" prefixes), similar length and style, all plausible. The correct one must NOT be the longest, most hedged, or only specific option.
- Vary the position of the correct option (answer_index 0-3); do not always use the same slot.
- Each question must be answerable from the cited EVIDENCE entry alone — never from outside knowledge.
- explanation: 1-2 sentences, citing what the material says.
- sources: only documents/pages present in EVIDENCE."""


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
        questions: list[Question] = []
        for attempt in range(4):
            # rotate a fresh window of slides per attempt so a bad draw
            # doesn't get retried against the same evidence
            start = 0
            if len(chunks) > want:
                start = (attempt * want) % (len(chunks) - want + 1)
            window = chunks[start : start + want]
            evidence_txt = "\n".join(
                f"{i}. [{e.document} | slide {e.page}" + (f" | {e.section}" if e.section else "") + f"] {e.text[:700]}"
                for i, e in enumerate(window, 1)
            )
            prompt = QUIZ_PROMPT.format(n=want, evidence=evidence_txt)
            raw = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=(0.0, 0.2, 0.4, 0.6)[attempt],
                max_tokens=self.max_tokens,
                json_mode=True,
            )
            extracted = _extract_questions(parse_json_payload(raw))
            questions = validate_questions(extracted if extracted is not None else [], window)
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
            questions.append(
                Question(
                    question="Which topic does the material cover?",
                    options=[e.section or "The topic shown", "The syllabus",
                             "A random topic", "None of the above"],
                    answer_index=0,
                    explanation=f"The cited material is “{e.section}” from {e.document}, slide {e.page}. "
                                f"Full text: {e.text[:300]}",
                    sources=[{
                        "document": e.document,
                        "page": e.page,
                        "section": e.section,
                        "excerpt": e.text[:700],
                        "image_path": e.image_path,
                    }],
                )
            )
        return questions
