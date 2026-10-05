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
