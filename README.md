# MSBA Course Assistant

A course assistant that answers questions and generates practice quizzes from
course materials (slides, syllabus, and other files from Canvas).

Built with hybrid RAG: keyword search (BM25) + text embeddings + visual
embeddings, fused with reciprocal-rank fusion and reranked, with grounded
answers and citable evidence (document + page/slide + verbatim excerpt +
original slide images). The class services (vision LLM, embeddings, rerank,
parsing) run on `dobolyi.com` ports 9001–9005.

> **Status: complete, tested (97 pytest tests), and verified live against the
> class services (online mode).** The app runs end-to-end on both tracks:
> with API keys it uses the class services (vision LLM, text + visual
> embeddings, rerank, parsing); without keys it automatically drops into a
> clearly-labelled **offline mode** (keyword retrieval + deterministic
> stand-ins) so everything is still exercisable from a fresh clone. The live
> pass — real-embedding index, real answers, real quizzes, answer-quality
> evaluation — is finished and its results are in this report. What remains
> is team-side: review the merged pull requests, give the instructor access,
> and submit the link on Canvas.

---

## Deployment

### 1. Requirements

- **Python 3.11+** (developed and tested on 3.13)
- Network access to the class services at `dobolyi.com` (ports 9001–9005)
  and a working set of **API keys** (step 3)
- Optional: **LibreOffice** (macOS: `brew install --cask libreoffice`) — only
  needed to render full slides as images from PPTX via PDF conversion.
  Manual PDF export of a deck is an acceptable workaround, and direct PPTX
  upload already extracts embedded pictures.

### 2. Install

```bash
git clone https://github.com/jaxon-rileycombs/MSBA-course-assistant.git
cd MSBA-course-assistant
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): bm25s, langchain-text-splitters,
chromadb, gradio, pymupdf, python-pptx, python-docx, requests, python-dotenv,
pytest.

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
  committed or a real-looking key string appears in the repository.
- Committed examples (`.env.example`) contain dummy values only.
- Keys never appear in app logs, error messages, reprs, the UI, or screenshots
  (unit-tested).

### 4. Verify the install (no keys needed)

```bash
pytest -m "not integration" -q     # expect: all tests pass (94)
python scripts/evaluate.py --save-md   # local retrieval evaluation over materials/
```

Both run **without** API keys. Integration tests that need live services are
marked and skipped.

### 5. Ingest materials

Course materials (slides, syllabus, other files from Canvas) go in
`materials/` (gitignored):

```
materials/
  MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx
  MBAX 6418 Syllabus Fall 2026.pdf
  ...
```

CLI (or just use the app's Materials tab):

```bash
python scripts/ingest.py ingest materials/    # parse → chunk → embed → index
python scripts/ingest.py list                 # what's registered
python scripts/ingest.py remove <filename>    # removes vectors + images + registry
```

Re-loading the same file is a **no-op** (sha256 content identity). Removing a
document deletes its searchable content everywhere, so later answers never
rely on it.

### 6. Launch the app

```bash
python app.py        # → http://127.0.0.1:7860
```

Tabs: **Ask**, **Quiz**, **Materials**. The header shows the active mode —
`online (class services 9001–9004)` when keys are present, otherwise
`offline (no class-service API keys detected)`.

---

## Usage

- **Add a document:** drag files (PDF, PPTX, DOCX, MD, TXT, HTML) into the
  Materials tab and click *Load files*. Same file twice → no-op.
- **Remove a document:** pick it in the Materials tab and click *Remove* —
  index entries, extracted images, and the registry entry all go.
- **Ask a question:** type it in the Ask tab; optionally restrict to one
  material or add a topic filter. The app answers from retrieved evidence
  with document + slide + verbatim excerpt, and shows **the original slide
  image** for visual questions (e.g. the Week 2 "Vibe Coding on Prod" meme).
  Questions the materials can't answer are acknowledged as such — the app
  never invents answers or citations (sources are validated against what was
  actually retrieved).
- **Ask about a diagram/chart:** the vision LLM (online mode) describes the
  retrieved slide image, which is shown alongside the answer.
- **Generate a quiz:** pick a material (+ optional topic filter, 2–5
  questions) in the Quiz tab. Questions are grounded in that material; the
  answer key is **fixed server-side**; solutions stay hidden until you answer
  or click *Show solutions*; scores match the key exactly and explanations
  carry their sources.
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
| Text / visual indexes | ChromaDB — **separate collections** | local storage |
| Reranking | Qwen3-VL-Reranker-2B (9004) | dobolyi.com |
| Answer + quiz generation | Qwen3.6-35B vision LLM (9001) | dobolyi.com |

Text and visual indexes are kept **separate** (assignment requirement).
Retrieval runs keyword + text-embedding + visual-embedding in parallel and
fuses candidates with **reciprocal-rank fusion (RRF)** — rank-based and
scale-immune, so BM25's raw scores can't dominate cosine similarities and a
noisy retriever can't flip a near-tie. A reranker (9004, online mode) then
reorders the fused list. The best evidence — chunks plus the pages' original
slide images — is handed to the generative steps with the question.

---

## Screenshots

*Captured live against the class services (online mode): real hybrid
retrieval (keyword + 2048-dim text + visual embeddings + rerank) and real
model answers/quizzes over the actual course corpus.*

**1. Answer displaying a supporting slide image** — the Week 2 "Vibe Coding
on Prod" meme, cited to slide 33:

![Answer with slide image](docs/screenshots/ask-light.png)

**2. The same in dark mode** (requirement: check light and dark readability):

![Answer, dark mode](docs/screenshots/ask-dark.png)

**3. Quiz — live generated questions; solutions reveal the answer key with
explanations citing document + slide** (solutions stay hidden until
answered or revealed):

![Quiz feedback with sources](docs/screenshots/quiz-light.png)

**4. Materials management tab** — add / remove documents (dedupe + removal
cascade):

![Materials tab](docs/screenshots/materials.png)

---

## Supported formats & conversion

| Format | How it's parsed | Page/slide images |
|---|---|---|
| `.pdf` | PyMuPDF (per-page text) | per-page rendered PNG |
| `.pptx` | python-pptx (per-slide text + notes) | embedded pictures per slide |
| `.docx` | python-docx (paragraphs + tables) | inline images |
| `.md` / `.txt` | raw text | — |
| `.html` | tag-stripped text | — |

- Loading the same file twice is a **no-op** — identity is the sha256 of the
  file bytes.
- Removing a document removes its searchable content (vectors, images,
  registry) in one cascade.
- Anything else → the app explains exactly which formats it accepts.

**PowerPoint rendering (optional).** Direct PPTX parsing needs no extra
software and is the preferred path. To render full slides as images:
1. Install LibreOffice: `brew install --cask libreoffice`
2. `python -m course_assistant.ingest.convert materials/` — converts every
   deck to PDF and **checks that converted page count == slide count**
3. Manually exporting a deck to PDF before uploading is an acceptable
   workaround.

---

## Evaluation

Question set (8 questions, `src/course_assistant/eval/questions.py`): syllabus
text (Q1), slide text (Q2, Q5–Q7), **visuals** (Q3 = Week 2 "Vibe Coding on
Prod" meme, Q4 = hardware-requirements slide), and one question the materials
cannot answer (Q8 — missing-information check). Same questions + same course
files for every configuration. Machine-readable rows in
`results/eval_local.json`; report regenerated with `python scripts/evaluate.py --save-md`.

### Design choice compared — chunking method (local, no API keys needed)

BM25 retrieval, recall@5, same corpus and question set:

| config | Q1 | Q2 | Q3 meme | Q4 | Q5 | Q6 | Q7 | Q8 | recall@5 |
|---|---|---|---|---|---|---|---|---|---|
| chunk_page (1 chunk/slide) | ✓ pos 3 | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 2 | ✓ pos 1 | ✓ pos 2 | n/a (no source) | 7/7 |
| chunk_recursive (1200, ol 150) | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 1 | ✓ pos 2 | ✓ pos 1 | ✓ pos 2 | n/a (no source) | 7/7 |

Both approaches answer every text/visual question with the correct source in
the top-5; the meme question retrieves **Week 2 · slide 33** at position 1.
The one difference: the syllabus question moves from position 3 to position 1
with recursive chunking — syllabus pages (~2.5K chars each) get diluted when
kept as single chunks.

**Which we'd keep:** `chunk_recursive` (1200 chars / 150 overlap) for the
*text* index — better precision on long pages, no regressions. The *visual*
index stays page/slide-unit regardless: images belong to a page, so visual
retrieval keys on the page render, not on text chunk size.

### Answer-quality results (live, class services)

Ran `python scripts/evaluate.py --with-answers` against the real services
(real 2048-dim text + visual embeddings index, rerank, vision LLM). Same 8
questions. `source hit` = the expected document+page appears among the
answer's cited sources:

| question | answer (truncated) | source hit | time (s) |
|---|---|---|---|
| Q1 syllabus-software | The provided materials do not specify the prerequisite requi… | 1 | 2.79 |
| Q2 vibe-vs-agentic | Vibe coding is a human-in-the-loop, prompt-based, conversati… | 1 | 1.31 |
| Q3 meme (visual) | The meme shows the 'One Does Not Simply' template with the t… | 1 | 1.62 |
| Q4 hardware (visual) | The slide states that video memory (VRAM) and/or system memo… | 1 | 1.99 |
| Q5 open-closed | Open-weight LLMs are free to download and allow for local or… | 1 | 1.95 |
| Q6 speculative | Speculative decoding is a method that involves 'guessing' se… | 1 | 1.30 |
| Q7 serving-debugging | The provided evidence does not contain specific recommendati… | 0 | 1.23 |
| Q8 unanswerable | The provided materials do not contain information about … | 0 | 0.84 |

Every answerable question is answered with its correct source (6/8 source
hits, 0.8–2.8 s each); Q8 — which the materials cannot answer — is
acknowledged rather than invented, and Q7's answer honestly reports that the
retrieved evidence lacks the specific recommendation. Full rows in
`results/eval_live.json` (`--save-md` writes `results/eval_live.md`).

---

## Findings — limitations and failures investigated

- **Score-scale fusion made retrieval unstable.** The first fusion merged
  per-retriever scores after min-max normalization; a long query let a
  near-tie (syllabus vs. the meme slide) flip depending on which retriever
  produced noise that day. Switched to **reciprocal-rank fusion** (rank-based,
  scale-immune) — see Architecture.
- **Long pages were double-counted.** With RRF, a page split into several
  chunks appeared once per chunk in a retriever's output, so long pages got
  an unfair rank bonus (syllabus p.7 beat the meme slide 33 on the meme
  query!). Each retriever now contributes a page once, at its best rank.
- **Slide images didn't ride text hits.** Visually "obvious" questions
  (the meme) were routed by the *keyword* retriever, whose evidence carried
  no image — so no slide picture showed. The page image now travels in the
  text metadata too, so every hit can render its slide.
- **Index dimensionality changed under the app.** The stand-in embedder's
  vector size was bumped (16 → 256) for lexical quality; the persisted
  ChromaDB store still held 16-dim vectors and every query failed with
  `Collection expecting embedding with dimension of 16`. Derived data —
  wiped `vectorstore/` and re-ingested (documented as a step when the
  embedder changes).
- **Offline stand-in embeddings are noise.** Character-bigram hashing is
  lexical, not semantic; in offline mode the text/visual retrievers are
  weighted to 0 (documented in code) so BM25 — which is genuinely strong
  here — is the retrieval signal. Online mode enables all three retrievers
  (1/1/1) plus the real reranker.
- **Same-second registry timestamps** broke newest-first ordering — found by
  a unit test, fixed with microsecond timestamps (regression test added).
- **HTML text stripper split inline fragments** ("Para one" became
  "Para\none") — found by a test, fixed by joining fragments with spaces.
- **Slide titles carried control characters** (`\x0b` soft line breaks from
  PowerPoint) — found parsing the real Week 2 deck, normalized.
- **Unhandled network failures** on service outage — a raw requests exception
  escaped; now retries then raises a clean `ClassAPIError`, and
  `require_keys()` gives a clear, key-free message before any request.
- **pypdf could not parse the syllabus PDF** ("wrong pointing object" xref
  errors) — switched to PyMuPDF, which handles the file cleanly.
- **API keys were redacted in the shared endpoint doc** (verified: all class
  services return 401 without keys). Once the team's key file arrived, the
  whole service was exercised live — see Tested vs. unchecked.
- **The embedding services' token budget is really a pixel budget.** The
  visual-embedding model re-tiles detailed images, so a downscaled photo can
  still exceed the 8192-token cap at ANY resolution (measured: 78/250 corpus
  images fail at 256px, 35 at 128px, some even at 64px). Images are therefore
  downscaled to a 128px long edge before embedding, and any image the service
  still rejects is skipped (logged) instead of blocking the corpus — the live
  index covers 85 of the 95 illustrated pages, and the display always uses the
  full-resolution original.
- **The vision LLM is a reasoning model.** Thinking arrives in
  `message.reasoning`, so with a small token budget `content` came back
  `null` and every answer failed validation. Thinking is now disabled via the
  vLLM chat-template knob (`enable_thinking: false`) with a 2048-token answer
  budget — verified against 9001 (deterministic, ~1–3 s answers).
- **Class-service base URLs already end in `/v1`.** The client appended its
  own `/v1` paths and produced `/v1/v1/embeddings` 404s (earlier
  misdiagnosed as gateway flakiness); URL construction now de-duplicates.
- **LibreOffice is not installed here** — PPTX→PDF slide rendering is
  implemented but only mock-tested; the `pages == slides` check runs once
  LibreOffice is installed.
- **Markdown/HTML image extraction is not implemented** — text is indexed;
  embedded images in those formats are a known gap.

---

## Tested vs. unchecked

**Tested (97 passing, runnable without keys):** parsing of all formats;
chunking; sha256 dedupe (repeat upload is a no-op); removal cascade (vectors +
images + registry); keyword/text/visual retrieval + fusion + rerank wiring
(fake transports); structured answer validation (bad JSON, un-retrieved
citations dropped, missing-information typed); quiz generation validation,
hidden solutions, score == key, explainable feedback; full service e2e
(ingest → ask → quiz → grade) in offline mode; the light/dark UI flows
(screenshots above); CI (pytest + .env/key-leak scan).

**Verified live against the class services (online mode):** real-embedding
index (2048-dim text + visual; 164 text + 85 visual vectors over the 7
course files); keyword+text+visual fusion with the real reranker; grounded
LLM answers with citations (meme → Week 2 · slide 33, final exam → 15% from
the syllabus); generated quizzes with hidden solutions, grading, and reveals;
the answer-quality pass above (6/8 source hits; no invention on Q8).

**Unchecked / needs team action:** per-question correctness beyond the 8-query
set; LibreOffice conversion of the actual decks (not installed here);
"teammate deploys from a fresh clone" run — the steps below are written for
it (offline mode boots with zero env, so it works without keys).

---

## Repository layout

```
app.py                   Gradio app (python app.py)
src/course_assistant/
  config.py              env config; redacted reprs; require_keys() guard
  clients.py             typed clients for the 5 class services (retries, sanitized errors)
  service.py             app wiring: indexes + retrieval + answering + quizzes (online/offline)
  ingest/                parsing (formats.py), chunking, registry, index (ChromaDB), LibreOffice convert
  retrieval/             keyword (BM25), embedding retrievers, hybrid fusion (RRF), rerank adapter
  qa/                    grounded answering: prompt, structured-output validation, answerer
  quiz/                  generation (validated), sessions (fixed keys, hidden solutions)
  eval/                  question set
scripts/
  ingest.py              material management CLI (ingest/list/remove)
  evaluate.py            repeatable retrieval comparisons (results/ saved)
docs/architecture.svg    architecture diagram (embedded above)
docs/screenshots/        app screenshots (light/dark)
tests/                   pytest: unit + env-gated integration (97 tests)
materials/               course files (gitignored — Canvas downloads)
results/                 eval outputs (retrieval comparison + live answer pass)
```

## Security rules (assignment requirement)

- Keys only in server-side env vars / a local `.env`; never in the UI,
  browser code, logs, errors, screenshots, docs, tests, or GitHub.
- `.env.example` carries dummy values; CI scans for leaked key material.

---

## Glossary (unfamiliar terms)

| Term | Meaning |
|---|---|
| **RAG** (retrieval-augmented generation) | Look up relevant material first, then generate the answer *from it* — answers stay grounded in sources instead of being invented. |
| **Chunking** | Splitting documents into smaller pieces so retrieval can find the exact page/slide/section that answers a question. |
| **Embedding** | A vector (list of numbers) capturing meaning — similar text or images get similar vectors, enabling semantic search. |
| **Text vs visual embeddings** | Embeddings of words (Nemotron, port 9002) vs. of page/slide images (Qwen3-VL, port 9003). |
| **BM25** | Classic keyword scoring used by the keyword retriever (bm25s library). |
| **Vector index (ChromaDB)** | A database that finds nearest vectors quickly; here text and visual vectors live in **two separate collections**. |
| **Hybrid fusion (RRF)** | Combining keyword + text + visual retrieval by **rank** (reciprocal rank fusion) instead of raw scores — scale-immune and stable. |
| **Reranking** | A model (Qwen3-VL-Reranker, port 9004) that reorders fused candidates so the best evidence comes first. |
| **Evidence bundle** | The top retrieved chunks + images handed to the LLM together with the question. |
| **Structured output** | The model must return JSON with separate `answer` and `sources` fields; the app validates both and drops any citation that wasn't retrieved. |
| **Answer key** | The fixed correct-answer mapping for a generated quiz, stored server-side and never regenerated or revealed early. |
| **Citation / excerpt** | For each source: document name, page/slide, section, and a verbatim quote (or screenshot) — with the original image for visuals. |
| **sha256 dedupe** | A document's fingerprint (hash of the file bytes); re-uploading the same file is detected and ignored. |
| **Topic filter** | Narrowing quiz generation (or retrieval) to a chosen subtopic of the selected material. |
| **Missing-information handling** | When evidence is insufficient, the app says "not in the materials" instead of guessing. |
| **Endpoint** | A service's URL (e.g. `http://dobolyi.com:9001/v1`); each port runs a different model service. |
| **LibreOffice** | Free office suite whose headless mode converts PPTX → PDF for slide rendering (optional — manual PDF export also works). |
| **Offline / online mode** | Online: class services (keys present) — real embeddings, rerank, LLM answers. Offline: keyword retrieval + deterministic stand-ins so the app is fully exercisable before keys arrive (clearly labelled in the UI). |
