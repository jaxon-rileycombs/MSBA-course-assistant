"""Quiz data model. The answer key is FIXED at generation time and never
regenerated; sessions only grade against it."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Question:
    question: str
    options: list[str]  # usually 4
    answer_index: int   # correct option index (the key)
    explanation: str = ""
    sources: list[dict] = field(default_factory=list)  # [{"document", "page"}, ...]

    def without_answer(self) -> dict:
        """Student-facing view: no key, no explanation."""
        return {
            "question": self.question,
            "options": self.options,
        }


@dataclass
class Quiz:
    quiz_id: str
    title: str
    questions: list[Question]
    created_at: str = ""
