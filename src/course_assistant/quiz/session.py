"""Quiz sessions: persist, hide solutions, grade against the FIXED key.

Requirement: solutions stay hidden until the student answers or requests
them, and scores must match the answer key. The key is stored at generation
time and never regenerated; ``get()`` omits it; ``submit()`` grades; only
``reveal()`` returns explanations with their sources.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
    def __init__(self, store_dir: str | Path, quiz_id: str = ""):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.quiz_id = quiz_id or uuid4().hex[:12]

    @property
    def path(self) -> Path:
        return self.store_dir / f"{self.quiz_id}.json"

    def create(self, title: str, questions: list[Question]) -> str:
        if not questions:
            raise ValueError("cannot create an empty quiz")
        doc = {
            "quiz_id": self.quiz_id,
            "title": title,
            "questions": [asdict(q) for q in questions],  # includes the key (server-side)
            "revealed": False,
        }
        self.path.write_text(json.dumps(doc, indent=2))
        return self.quiz_id

    def get(self, include_solutions: bool = False) -> dict:
        """Student view hides answer_index/explanation unless revealed."""
        if not self.path.exists():
            raise KeyError(f"quiz {self.quiz_id} not found")
        doc = json.loads(self.path.read_text())
        questions = []
        for q in doc["questions"]:
            qq = {k: q[k] for k in ("question", "options")}
            if include_solutions:
                qq["answer_index"] = q["answer_index"]
                qq["explanation"] = q["explanation"]
                qq["sources"] = q["sources"]
            questions.append(qq)
        return {"quiz_id": doc["quiz_id"], "title": doc["title"], "questions": questions}

    def submit(self, answers: list[int]) -> Result:
        doc = json.loads(self.path.read_text())
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

    def reveal(self) -> dict:
        """Unlock solutions + explanations (requested answers path)."""
        if self.path.exists():
            doc = json.loads(self.path.read_text())
            doc["revealed"] = True
            self.path.write_text(json.dumps(doc, indent=2))
        return self.get(include_solutions=True)

    def list_all(self) -> list[dict]:
        out = []
        for p in sorted(self.store_dir.glob("*.json")):
            d = json.loads(p.read_text())
            out.append({"quiz_id": d["quiz_id"], "title": d["title"], "n": len(d["questions"])})
        return out
