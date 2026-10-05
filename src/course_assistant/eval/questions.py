"""Evaluation question set (assignment requirement 5).

8 questions covering syllabus text, slide text, and visuals (including the
Week 2 "Vibe Coding on Prod" meme), plus one question the materials cannot
answer (missing-information check).

Each entry:
- id: short label for result tables
- question: the exact question a student would ask
- kind: "text" | "visual" | "unanswerable"
- doc: substring that must appear in the supporting document name
- page: expected 1-based page/slide (None = any page of that document)
"""

QUESTION_SET = [
    {
        "id": "Q1-syllabus-software",
        "question": "What are the prerequisite requirements for this course, and which software do you need to install for the class?",
        "kind": "text",
        "doc": "Syllabus",
        "page": None,
    },
    {
        "id": "Q2-vibe-vs-agentic",
        "question": "What is vibe coding and how does it differ from agentic coding?",
        "kind": "text",
        "doc": "Week 2",
        "page": 30,
    },
    {
        "id": "Q3-meme",
        "question": "Find the meme about Vibe Coding on \"Prod\" in the Week 2 slides and explain what it shows.",
        "kind": "visual",
        "doc": "Week 2",
        "page": 33,
    },
    {
        "id": "Q4-hardware",
        "question": "What does the hardware requirements slide say about running models locally?",
        "kind": "visual",
        "doc": "Week 2",
        "page": 24,
    },
    {
        "id": "Q5-open-closed",
        "question": "According to the Week 2 slides, what are the trade-offs between open-weight and closed-weight LLMs?",
        "kind": "text",
        "doc": "Week 2",
        "page": 4,
    },
    {
        "id": "Q6-speculative",
        "question": "What is speculative decoding and what problem does it solve?",
        "kind": "text",
        "doc": "Week 2",
        "page": 13,
    },
    {
        "id": "Q7-serving-debugging",
        "question": "What does the Week 4 deck recommend for serving and debugging local models?",
        "kind": "text",
        "doc": "Week 4",
        "page": None,
    },
    {
        "id": "Q8-unanswerable",
        "question": "What was OpenAI's annual revenue in fiscal year 2025?",
        "kind": "unanswerable",
        "doc": None,
        "page": None,
    },
]
