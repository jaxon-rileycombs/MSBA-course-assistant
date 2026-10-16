"""Quiz sessions: persist, hide solutions, grade against the FIXED key.

Requirement: solutions stay hidden until the student answers or requests
them, and scores must match the answer key. The key is stored at generation
time and never regenerated; ``get()`` omits it; ``submit()`` grades; only
``reveal()`` returns explanations with their sources.

All lookups take an explicit ``quiz_id`` (the store keeps many quizzes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .schema import Question


@dataclass
class Result:
    score: int
    total: int
    correct: list[bool]
    feedback: list[dict]  # per question: explanation + sources (after grading)


class QuizSession:
    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, quiz_id: str) -> Path:
        if not isinstance(quiz_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", quiz_id):
            raise ValueError("invalid quiz id")
        return self.store_dir / f"{quiz_id}.json"

    def create(self, title: str, questions: list[Question], quiz_id: str = "") -> str:
        if not questions:
            raise ValueError("cannot create an empty quiz")
        quiz_id = quiz_id or uuid4().hex[:12]
        doc = {
            "quiz_id": quiz_id,
            "title": title,
            "questions": [_q_to_dict(q) for q in questions],  # includes the fixed key
            "revealed": False,
        }
        self._path(quiz_id).write_text(json.dumps(doc, indent=2))
        return quiz_id

    def get(self, quiz_id: str, include_solutions: bool = False) -> dict:
        """Student view hides answer_index/explanation unless revealed."""
        path = self._path(quiz_id)
        if not path.exists():
            raise KeyError(f"quiz {quiz_id} not found")
        doc = json.loads(path.read_text())
        questions = []
        for q in doc["questions"]:
            qq = {k: q[k] for k in ("question", "options")}
            if include_solutions:
                qq["answer_index"] = q["answer_index"]
                qq["explanation"] = q["explanation"]
                qq["sources"] = q["sources"]
            questions.append(qq)
        return {"quiz_id": doc["quiz_id"], "title": doc["title"], "questions": questions}

    def submit(self, quiz_id: str, answers: list[int]) -> Result:
        path = self._path(quiz_id)
        if not path.exists():
            raise KeyError(f"quiz {quiz_id} not found")
        doc = json.loads(path.read_text())
        key = [q["answer_index"] for q in doc["questions"]]
        if len(answers) != len(key):
            raise ValueError(f"expected {len(key)} answers, got {len(answers)}")
        correct = [a == k for a, k in zip(answers, key)]
        feedback = [
            {
                "correct": c,
                "explanation": q["explanation"],
                "sources": q["sources"],
                "your_answer": answers[i],
                "correct_answer": key[i],
            }
            for i, (c, q) in enumerate(zip(correct, doc["questions"]))
        ]
        return Result(score=sum(correct), total=len(key), correct=correct, feedback=feedback)

    def reveal(self, quiz_id: str) -> dict:
        """Unlock solutions + explanations (requested-answers path)."""
        path = self._path(quiz_id)
        doc = json.loads(path.read_text())
        doc["revealed"] = True
        path.write_text(json.dumps(doc, indent=2))
        return self.get(quiz_id, include_solutions=True)

    def list_all(self) -> list[dict]:
        out = []
        for p in sorted(self.store_dir.glob("*.json")):
            d = json.loads(p.read_text())
            out.append({"quiz_id": d["quiz_id"], "title": d["title"], "n": len(d["questions"])})
        return out


def _q_to_dict(q: Question) -> dict:
    return {
        "question": q.question,
        "options": q.options,
        "answer_index": q.answer_index,
        "explanation": q.explanation,
        "sources": q.sources,
    }
