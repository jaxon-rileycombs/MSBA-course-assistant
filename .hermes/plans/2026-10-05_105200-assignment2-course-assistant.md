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

- Repo: `jaxon-rileycombs/MSBA-course-assistant` (new, public). Fresh, no code.
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
2. **Ingestion** — supported formats (PDF/PPTX/DOCX/MD/TXT/HTML) + explicit
   "we accept …" message; extract original page/slide images; parse via class
   service where documented (fallback local libs); chunk with
   source-preserving metadata; deterministic chunk IDs. *(issue 2)*
3. **Indexing** — two separate chroma collections (`text_chunks`,
   `visual_pages`); embeddings from class services; `scripts/ingest.py`
   idempotent (upsert by ID, removes deleted docs). *(issue 3)*
4. **Hybrid retrieval** — BM25 + text emb + visual emb → score-normalized
   fusion (weights configurable) → class rerank → evidence list (doc, page,
   section, excerpt, image path). Query→visual mapping: use text-embedding of
   the query against visual index; document the choice. *(issue 4)*
5. **Grounded answering** — evidence-bundle prompt; vision LLM; strict JSON
   `answer` + `sources`; validators (schema, source resolves, excerpt
   verbatim); one corrective retry; else "cannot answer from materials";
   `missing_info` flag; keys never logged. *(issue 5)*
6. **Citations** — render doc + page/slide/section + verbatim excerpt or page
   screenshot, original image shown for visual evidence; shared renderer used
   by quiz feedback. *(issue 6)*
7. **Quiz engine** — material picker + topic filter; seeded MCQ generation
   grounding every question in a chunk; fixed server-side answer key;
   solutions hidden until answered or explicitly requested; scoring +
   explanations with citations. *(issue 7)*
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
- dots.mocr (9005) request format — vLLM-served; consult `/docs` and the HF
  page; fall back to local libs (pypdf, python-pptx) for text extraction.
- Query→visual-space embedding approach — validate against the visual
  service's documented API (`/v1/embeddings` vs `/v2/embed`) before finalizing.
- Class services are plain HTTP on campus network — treat as trusted-network;
  do not use outside campus without review.
- Materials in place (7 files): syllabus PDF, Weeks 2–6 slide decks, Local
  Inference deck — golden corpus for reproducible tests.
