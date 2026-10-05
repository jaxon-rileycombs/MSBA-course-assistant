# MSBA Course Assistant

A course assistant that answers questions and generates practice quizzes from
course materials (slides, syllabus, and other files from Canvas).

Built with hybrid RAG: keyword search (BM25) + text embeddings + visual
embeddings, fused and reranked, with grounded answers and citable evidence
(document + page/slide + verbatim excerpt + original images).

> **Status: building incrementally** against the task list in the GitHub
> issues. What works today: parsing & chunking, dedupe, registry, retrieval
> evaluation, config/clients/CI. What is coming: vector indexes (issue #3),
> hybrid retrieval + rerank (issue #4), grounded answering (issue #5),
> citations (issue #6), quiz engine (issue #7), UI + screenshots + final
> report (issue #8). The architecture diagram below matches this design; the
> team verifies it against the finished app at submission and explains the
> findings in their own words.

---

## Deployment

### 1. Requirements

- **Python 3.11+** (developed and tested on 3.13)
- Network access to the class services at `dobolyi.com` (ports 9001–9005)
  and a working set of **API keys** (step 3)
- Optional: **LibreOffice** (macOS: `brew install --cask libreoffice`) — only
  needed to render full slides as images from PPTX via PDF conversion.
  Manual PDF export of a deck is an acceptable workaround.

### 2. Install

```bash
git clone https://github.com/jaxon-rileycombs/MSBA-course-assistant.git
cd MSBA-course-assistant
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All dependencies are pinned ranges in `requirements.txt` (bm25s,
langchain-text-splitters, chromadb, gradio, pymupdf, python-pptx,
python-docx, requests, python-dotenv, pytest).

### 3. Configure API keys (never committed)

The app reads class-service settings from environment variables. Real keys
live ONLY in a local, gitignored `.env`:

```bash
cp .env.example .env
# edit .env — paste the real keys from the team's class-endpoints.txt
# into the CLASS_*_API_KEY= lines (five services: vision LLM, text embeddings,
# visual embeddings, rerank, document parsing)
```

- `.env` is covered by `.gitignore` and CI **fails the build** if `.env` is
  committed or if a real-looking key string appears in the repository.
- Committed examples (`.env.example`) contain dummy values only.
- Keys never appear in app logs, error messages, reprs, the UI, or screenshots
  (unit-tested).

### 4. Verify the install

```bash
pytest -m "not integration" -q     # expect: all tests pass (48+)
python scripts/evaluate.py --save-md   # local retrieval evaluation over materials/
```

Both run **without** API keys (integration tests that need the live services
are marked and skipped).

### 5. Ingest materials

Course materials (slides, syllabus, other files from Canvas) go in
`materials/` (gitignored):

```
materials/
  MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx
  MBAX 6418 Syllabus Fall 2026.pdf
  ...
```

The ingest pipeline (`scripts/ingest.py`, issue #3) parses, dedupes by
content hash, extracts page/slide images, and indexes text + visual
embeddings. A document can then be removed — which deletes its searchable
content everywhere.

### 6. Launch the app

`python -m course_assistant.ui` starts the Gradio interface (issue #8) with
tabs for Materials, Ask, and Quiz. Requires the class services (issued keys).

---

## Usage

- **Add a document:** upload a file (PDF, PPTX, DOCX, MD, TXT, HTML) in the
  Materials tab. Loading the same file twice is a no-op (sha256 identity).
- **Remove a document:** delete it in the Materials tab — its index entries,
  extracted images, and registry entry are removed, so later answers never
  rely on it.
- **Ask a question:** type it in the Ask tab (optionally restrict to one
  material). The app shows the answer with document + page/slide number +
  verbatim excerpt, and the original slide image for visual questions.
  Questions the materials can't answer are acknowledged as such — the app
  never invents answers or citations.
- **Ask about a diagram/chart:** the vision LLM describes the retrieved
  slide image; the image itself is shown as evidence.
- **Generate a quiz:** pick a material (+ optional topic filter) in the Quiz
  tab. Multiple-choice questions are grounded in that material; the answer
  key is fixed server-side; solutions stay hidden until you answer or click
  "Show answers"; scores and explanations (with sources) follow.
- **Rerun tests:** `pytest -m "not integration" -q`.

---

## Architecture

![Architecture diagram](docs/architecture.svg)

**Component / service map** — what runs where:

| Component | Model / tool | Where it runs |
|---|---|---|
| Gradio UI (Materials · Ask · Quiz) | gradio | local |
| Parsing (text) | PyMuPDF, python-pptx, python-docx, stdlib (dots.mocr on 9005 optional) | local |
| Page/slide image extraction | PyMuPDF renders, embedded picture extraction | local |
| Chunking | langchain-text-splitters (recursive, 1200/150) | local |
| Registry / dedupe / removal cascade | JSON registry + ChromaDB delete-by-doc | local |
| Keyword retrieval | bm25s | local |
| Text embeddings | Nemotron-3-Embed-1B (9002) | dobolyi.com |
| Visual embeddings | Qwen3-VL-Embedding-2B (9003) | dobolyi.com |
| Text / visual indexes | ChromaDB (separate collections) | local storage |
| Reranking | Qwen3-VL-Reranker-2B (9004) | dobolyi.com |
| Answer + quiz generation | Qwen3.6-35B vision LLM (9001) | dobolyi.com |

Text and visual indexes are kept **separate** (assignment requirement);
retrieval runs keyword + text-embedding + visual-embedding in parallel, fuses
the candidates, reranks, and supplies the best evidence (chunks + images) to
the generative steps.

---

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

---

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

---

## Findings — limitations and failures investigated

- **Same-second registry timestamps** broke newest-first ordering — found by
  a unit test, fixed with microsecond timestamps (regression test added).
- **HTML text stripper split inline fragments** ("Para one" became
  "Para\none") — found by a test, fixed by joining fragments with spaces.
- **Slide titles carried control characters** (`\x0b` soft line breaks from
  PowerPoint) — found while parsing the real Week 2 deck, normalized.
- **Unhandled network failures** on service outage — a raw requests exception
  escaped; now retries then raises a clean `ClassAPIError`, and
  `require_keys()` gives a clear, key-free message before any request.
- **pypdf could not parse the syllabus PDF** ("wrong pointing object" xref
  errors) — switched parsing to PyMuPDF, which handles the file cleanly.
- **API keys are redacted in the shared endpoint doc** (verified: all class
  services return 401 without keys). Whole-service integration and
  answer-quality evaluation remain blocked until the team supplies keys —
  everything else is exercised with mocks/fakes.
- **LibreOffice is not installed here** — PPTX→PDF slide rendering is
  implemented but only mock-tested; the `pages == slides` check runs once
  LibreOffice is installed.
- **Markdown/HTML image extraction is not implemented** — text is indexed;
  embedded images in those formats are a known gap.

---

## Screenshots

Pending the UI build-out (issue #8). Final submission will include:

1. An answer displaying a supporting slide image (e.g. the Week 2 "Vibe
   Coding on Prod" meme with its source citation: document + slide 33).
2. Quiz feedback with sources (score + explanations citing the material).

---

## Repository layout

```
src/course_assistant/
  config.py          env config; redacted reprs; require_keys() guard
  clients.py         typed clients for the 5 class services (retries, sanitized errors)
  ingest/            parsing (formats.py), chunking, registry, LibreOffice convert
  eval/              question set + evaluation harness
scripts/             evaluate.py (repeatable comparisons; results/ saved)
docs/architecture.svg
tests/               pytest: unit + env-gated integration
materials/           course files (gitignored — Canvas downloads)
```

## Security rules (assignment requirement)

- Keys only in server-side env vars / a local `.env`; never in the UI,
  browser code, logs, errors, screenshots, docs, tests, or GitHub.
- `.env.example` carries dummy values; CI scans for leaked key material.
