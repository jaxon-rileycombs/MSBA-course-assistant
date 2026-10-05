# MSBA Course Assistant

A course assistant that answers questions and generates practice quizzes from
course materials (slides, syllabus, and other files from Canvas).

Built with hybrid RAG: keyword search (BM25) + text embeddings + visual
embeddings, fused and reranked, with grounded answers and citable evidence.

> **Status: planning.** Implementation plan lives in
> `.hermes/plans/2026-10-05_105200-assignment2-course-assistant.md`.
> Task breakdown and ownership live in the GitHub issues.
>
> Setup, usage screenshots, findings, and limitations will be documented here
> as the project is built (see issue #8).

## Supported formats & conversion

| Format | How it's parsed | Page/slide images |
|---|---|---|
| `.pdf` | PyMuPDF (per-page text) | per-page rendered PNG |
| `.pptx` | python-pptx (per-slide text + notes) | embedded pictures per slide |
| `.docx` | python-docx (paragraphs + tables) | inline images |
| `.md` / `.txt` | raw text | — |
| `.html` | tag-stripped text | — |

- Loading the same file twice is a **no-op** — a document's identity is the
  sha256 of its bytes, so duplicates are never ingested twice.
- Removing a document removes its searchable content (index entries,
  extracted images, registry) — wired up with the index layer (issue #3).
- Anything else → the app explains exactly which formats it accepts.

**PowerPoint rendering (optional).** Direct PPTX parsing needs no extra
software and is the preferred path. To render full slides as images:
1. Install LibreOffice: `brew install --cask libreoffice`
2. `python -m course_assistant.ingest.convert materials/` — converts every
   deck to PDF and **checks that converted page count == slide count**
3. Manually exporting a deck to PDF before uploading is an acceptable
   workaround.

## Evaluation (answer quality)

Question set (8 questions, `src/course_assistant/eval/questions.py`): syllabus
text (Q1), slide text (Q2, Q5–Q7), **visuals** (Q3 = Week 2 "Vibe Coding on
Prod" meme, Q4 = hardware-requirements slide), and one question the materials
cannot answer (Q8 — missing-information check). Same questions + same course
files for every configuration. Full machine-readable rows in
`results/eval_local.json`; report regenerated with `python scripts/evaluate.py --save-md`.

### Retrieval comparison — chunking method (local, no API keys needed)

BM25 retrieval, recall@5, same corpus and question set:

| config | Q1 | Q2 | Q3 meme | Q4 | Q5 | Q6 | Q7 | Q8 | recall@5 |
|---|---|---|---|---|---|---|---|---|---|
| chunk_page (1 chunk/slide) | ✓ pos 3 | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 2 | ✓ pos 1 | ✓ pos 2 | n/a (no source) | 7/7 |
| chunk_recursive (1200, ol 150) | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 2 | ✓ pos 1 | ✓ pos 2 | n/a (no source) | 7/7 |

All text/visual questions were answered with the correct source within the
top-5 in both setups; the meme question retrieves **Week 2 · slide 33** at
position 1. The one difference: the syllabus question moves from position 3
to position 1 with recursive chunking — syllabus pages (~2.5K chars each) get
diluted when kept as single chunks.

**Which I'd keep:** `chunk_recursive` (1200 chars / 150 overlap) for the
*text* index — better precision on long pages, no regressions elsewhere.
The *visual* index stays page/slide-unit regardless: images belong to a page,
so visual retrieval always keys on the page render, not on text chunk size.

### Answer-quality results (pending)

Correctness, source support, and end-to-end timing per question require the
class services (LLM + rerank + embeddings) — blocked on the team's API keys.
Once available: `python scripts/evaluate.py --with-answers` extends this
report with per-question answers, source-support checks, and timings
(README "tested vs. unchecked" will be updated with what ran).
