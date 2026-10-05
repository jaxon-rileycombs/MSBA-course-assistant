# Assignment 2 — Course Assistant: Implementation Plan

**Goal:** Build a course assistant that answers questions and generates
practice quizzes using course materials, with hybrid RAG (keyword + text +
visual retrieval), grounded structured answers, and citable visual evidence.

**Architecture:** Ingestion pipeline parses course files (text + original
page/slide images) and chunks them with source metadata. Two separate indexes
(text chunks, visual pages) feed a hybrid retriever: BM25 (bm25s) + text
embeddings + visual embeddings, fused and reranked by the class services. A
class LLM answers with strict `answer` + `sources` JSON from the retrieved
evidence; a quiz engine generates grounded multiple-choice questions with a
fixed server-side answer key. Gradio UI on top.

**Tech stack:** Python 3.11+, bm25s, langchain-text-splitters, chromadb, gradio;
class services at dobolyi.com (role map confirmed from the team's
`class-endpoints.txt`): vision LLM (9001, Qwen3.6-35B VLM), text embeddings
(9002, Nemotron-3-Embed-1B), visual embeddings (9003, Qwen3-VL-Embedding-2B),
multimodal rerank (9004, Qwen3-VL-Reranker-2B), document parsing (9005,
dots.mocr). Port 9000 (Hermes) is reserved — do not use. Keys live only in
local `.env`.

---

## Current context & assumptions

- Repo: `jaxon-rileycombs/MSBA-course-assistant` (new, private). Fresh, no code.
- Class services are reachable and have FastAPI `/docs` (confirmed 9001–9005).
- Auth keys: the user adds them to local `.env`; nothing real is committed.
- Materials: user downloads slides/syllabus/other from Canvas into
  `materials/` (gitignored). Golden test set: 1 PDF + 1 PPTX + 1 DOCX + 1 MD.
- Owner: Hermes builds; the team chooses issue ownership and reviews PRs.
- Plain HTTP on a campus network — no TLS assumption; keys still required.

## Repo layout

```
src/course_assistant/
  config.py            # .env loading; errors never echo values
  clients.py           # typed clients for each class service (per /docs)
  ingest/
    formats.py         # supported formats + page/slide image extraction
    parse.py           # class parse service (fallback: local libs)
    chunking.py        # langchain splitters; metadata: doc, page, section
    index.py           # chromadb: text_chunks + visual_pages, separate
  retrieval/
    keyword.py         # bm25s
    text_embed.py      # class text embeddings
    visual_embed.py    # class visual embeddings
    rerank.py          # class rerank service
    hybrid.py          # fusion + rerank -> evidence list
  qa/
    answer.py          # vision LLM call w/ evidence + instructions
    schema.py          # {answer, sources[], missing_info}
    validate.py        # schema + quote + source-support checks
    citations.py       # render doc/page/section + excerpt or screenshot + image
  quiz/
    generate.py        # grounded MCQ generation (seeded, fixed key)
    session.py         # key storage, scoring, hidden solutions
  ui/
    app.py             # gradio: Materials/About, Ask, Quiz tabs
scripts/ingest.py      # idempotent ingest
scripts/evaluate.py    # seeded A/B comparison (issue 8)
tests/                 # pytest, unit + integration (env-gated)
materials/             # gitignored Canvas downloads
data/images/           # extracted page/slide images (gitignored)
vectorstore/           # chroma persistence (gitignored)
results/               # eval artifacts (committed where small)
```

## Tasks (mapped to GitHub issues)

Each issue below has completion checks; the team picks ownership. Branch
`issue/N-<slug>` → PR with review → merge (closes issue).

1. **Foundation** — scaffold, config, CI, class API client. Pinned deps
   (`bm25s`, `langchain-text-splitters`, `chromadb`, `gradio`, others).
   `.env.example` with dummy values; `.env` gitignored; CI greps for leaked
   keys; integration tests env-gated. *(issue 1)*
2. **Ingestion + document management** — supported formats (PDF/PPTX/DOCX/
   MD/TXT/HTML) + explicit "we accept …" message; extract original page/slide
   images; parse via class service where documented (fallback local libs);
   chunk with source-preserving metadata; deterministic chunk IDs.
   **Requirement 1:** add/remove documents through the app; doc_id = sha256 →
   same file twice is a no-op; remove() purges text+visual index entries,
   images, and the registry so later answers never rely on removed material.
   **PowerPoint:** direct PPTX parsing is the primary path; optional
   LibreOffice headless → PDF conversion (brew install --cask libreoffice)
   for full-slide rendering, with a converted-slide correctness check
   (pages == slides + sample renders); manual PDF export documented as a
   workaround; formats + conversion steps go in README/UI. *(issue 2)*
3. **Indexing** — two separate chroma collections (`text_chunks`,
   `visual_pages`); embeddings from class services; `scripts/ingest.py`
   idempotent (upsert by ID, sha256 dedupe, removes deleted docs via a
   `data/registry.json` cascade). *(issue 3)*
4. **Hybrid retrieval** — BM25 + text emb + visual emb → score-normalized
   fusion (weights configurable) → class rerank → evidence list (doc, page,
   section, excerpt, image path). Query→visual mapping: use text-embedding of
   the query against visual index; document the choice.
   **Requirement 2:** visual retrieval must surface relevant slide IMAGES
   (visual index over page/slide renders), not just text chunks — evidence
   lists carry image paths up to answer rendering so pictures of the actual
   slides can be shown. *(issue 4)*
5. **Grounded answering** — evidence-bundle prompt; vision LLM; strict JSON
   `answer` + `sources`; validators (schema, source resolves, excerpt
   verbatim); one corrective retry; else "cannot answer from materials";
   `missing_info` flag; keys never logged. *(issue 5)*
6. **Citations** — render doc + page/slide/section + verbatim excerpt or page
   screenshot, original image shown for visual evidence; shared renderer used
   by quiz feedback.
   **Requirement 2:** answers show pictures of the actual slides PLUS document
   name + slide number; users can ask the app to explain slide content
   (pictures/diagrams/charts) via the vision LLM. Acceptance: Week 2 "Vibe
   Coding on Prod" meme slide is retrieved, displayed, and summarized with a
   source that checks against the original slide. *(issue 6)*
7. **Quiz engine** — material picker + topic filter; seeded MCQ generation
   grounding every question in a chunk; fixed server-side answer key;
   solutions hidden until answered or explicitly requested; scoring +
   explanations with citations.
   **Requirement 3 (verification):** confirm generated questions are
   answerable from the selected material; solutions stay hidden until the
   user answers or requests them; scores match the answer key; explanations
   are useful and their cited sources support the correct answer — as
   automated tests AND a documented manual walkthrough. *(issue 7)*
   **Requirement 4 (test the app thoroughly):** unit tests + end-to-end tests
   (complete workflows) + manual runs across: add/remove files, text and
   image Q&A, quiz generation/scoring; edge cases: missing information,
   repeated uploads, unavailable model services (graceful degradation);
   fix problems found and re-run the affected tests. Light AND dark mode
   (Gradio) checked for readability of text, controls, and source images.
   **Requirement 6 (deploy & test independently):** submission includes code,
   dependencies, and clear install/configure/launch instructions; list extra
   software (LibreOffice) and how to supply API keys without committing them;
   validate the docs from a FRESH repository copy (teammate simulation) and
   fix any missing steps discovered. *(issue 8)*
8. **UI + evaluation + README** — Gradio tabs; the one design-choice
   comparison; README (setup, screenshots, findings, limitations, tested-vs-
   unchecked, glossary); security pass. *(issue 8)*

## The one design-choice comparison (required by the assignment)

- **Dimension:** retrieval configuration.
- **Configs:** (A) BM25-only · (B) hybrid BM25+text-embeddings, equal weight ·
  (C) full hybrid + rerank.
- **Protocol:** fixed seeded set of ~20 questions over the golden corpus;
  metric = evidence recall@8 for the true page/slide + answer groundedness
  (class LLM judges "answer fully supported by cited sources"). Three runs per
  config; results saved to `results/`; README Findings documents the winner
  and why, plus limitations.
- **Repeatable:** `make evaluate` reproduces the saved numbers; CI keeps the
  harness green (full run marked integration/manual).

## Tests / validation

- Unit tests run in CI (`pytest -m "not integration"`). Integration tests hit
  dobolyi.com and are skipped unless `CLASS_*` env vars are set.
- Golden-sample tests: parse correctness, image/page parity, chunk metadata,
  determinism, retrieval top-5 hit, quote-verbatim validation, quiz key
  immutability, hidden-solutions behavior.

## Security rules (assignment requirement)

Keys only in server-side env vars / local file. Never in UI, browser code,
logs, errors, screenshots, documentation, tests, or GitHub. CI grep checklist
is part of issue 1.

## Risks & open questions

- **API keys are REDACTED in the shared class-endpoints.txt** (4-char
  placeholders). All five services return 401 without a key (verified). Real
  keys are pending from the team — they go only into local `.env`. Code and
  unit tests proceed with mocks meanwhile; integration work is gated on keys.
- **LibreOffice is NOT installed on the dev machine** — PPTX stays the
  primary parse path (python-pptx + embedded images). The LibreOffice PDF
  conversion is implemented as an optional, documented script gated on
  `soffice` being present; manual PDF export is the documented workaround.
- dots.mocr (9005) request format — vLLM-served; consult `/docs` and the HF
  page; fall back to local libs (pypdf/pymupdf, python-pptx) for extraction.
- Query→visual-space embedding approach — validate against the visual
  service's documented API (`/v1/embeddings` vs `/v2/embed`) before finalizing.
- Class services are plain HTTP on campus network — treat as trusted-network;
  do not use outside campus without review.
- Materials in place (7 files): syllabus PDF, Weeks 2–6 slide decks, Local
  Inference deck — golden corpus for reproducible tests.

---

## Status at handoff (all buildable work is done)

**Built & tested — 91/91 pytest passing (default suite, no keys):**
- Issues #3 (ChromaDB index: separate text/visual collections, sha256 dedupe,
  remove cascade), #4 (keyword + text + visual retrieval, RRF fusion, rerank
  adapter), #5 (grounded answering, structured output validation,
  missing-information handling), #7 (quiz engine: validated generation, fixed
  server-side keys, hidden solutions), #6 + #8 (Gradio app: Ask / Quiz /
  Materials, online/offline auto-detect, elem_ids, light & dark verified).
- PRs #9–#15 open, one per issue, each with a plain-language body.
  All issues #1–#8 labeled (`foundation`, `ingestion`, per-area labels,
  `blocked-on-keys`) for student readability; README carries a glossary.
- README rewritten as the final report: deployment guide, architecture
  (docs/architecture.svg), evaluation (chunking comparison: recursive
  1200/150 for text, page-unit for visual; meme → slide 33 pos 1),
  screenshots (light + dark ask, quiz feedback with sources, materials),
  tested-vs-unchecked, findings (incl. RRF double-count fix, keyword-image
  metadata, offline-embedding noise), security rules.

**Key decisions:** RRF over score fusion (scale-immune); offline mode =
keyword retrieval only (stand-in embeddings are lexical noise; real embeddings
+ rerank activate online); text metadata carries the page image so every hit
can show its slide; recursive text chunks + page-unit visual chunks.

**Blocked on team keys (cannot be done from here):** real semantic retrieval
& rerank, live LLM answers/quizzes, answer-quality evaluation, real-model
screenshots. Team must also: review/merge PRs #9–#15, grant the instructor
access to the private repo, submit the link on Canvas, and run the fresh-clone
deployment steps in the README.
