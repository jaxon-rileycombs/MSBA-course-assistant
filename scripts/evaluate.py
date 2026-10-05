#!/usr/bin/env python3
"""Repeatable evaluation harness (assignment requirement 5).

Compares retrieval configurations on the SAME question set and course files.
Currently active configs (all local, no API keys needed):
    chunk_page        - one chunk per page/slide (native unit) + BM25
    chunk_recursive   - recursive 1200-char split (overlap 150) + BM25
Metrics per question (k = top-k evidence):
    recall@k          - expected document (+page) appears in the top-k
    retrieval_seconds - wall time for the retrieval step

When the class-service keys are available and the QA pipeline lands
(issues 4-5), run `--with-answers` to add: answer text, source-support
check, answer seconds, and a manual-review correctness column.

Usage:
    python scripts/evaluate.py [--configs chunk_page,chunk_recursive]
                               [--materials materials] [--k 5]
                               [--out results/eval_local.json]
Results are written as JSON (rows) + printed as a table. Run with --save-md
to also write a markdown snippet for the README report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bm25s  # noqa: E402

from course_assistant.eval.questions import QUESTION_SET  # noqa: E402
from course_assistant.ingest.chunking import chunk_document  # noqa: E402
from course_assistant.ingest.formats import parse_document  # noqa: E402

CHUNK_CONFIGS = {
    "chunk_page": {"chunk_size": 10**9, "chunk_overlap": 0},
    "chunk_recursive": {"chunk_size": 1200, "chunk_overlap": 150},
}

K = 5


def load_corpus(materials_dir: Path, images_root: Path, chunk_config: dict):
    """Parse every material and chunk it with the given config."""
    chunks = []
    for f in sorted(materials_dir.iterdir()):
        if f.suffix.lower() not in (".pdf", ".pptx", ".docx", ".md", ".txt", ".html"):
            continue
        doc = parse_document(f, images_root)
        chunks.extend(chunk_document(doc, **chunk_config))
    return chunks


def build_bm25(chunks):
    corpus = [c.text for c in chunks]
    tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return retriever


def retrieve(retriever, chunks, question, k: int):
    q_tokens = bm25s.tokenize([question], stopwords="en")
    hits, scores = retriever.retrieve(q_tokens, k=k)
    docs = hits[0]  # per-query result
    sc = scores[0]
    out = []
    for doc_token, score in zip(docs, sc):
        idx = int(doc_token)
        c = chunks[idx]
        out.append((c, float(score)))
    return out


def expected_hit(spec: dict, chunk) -> bool:
    if spec.get("doc") is None:
        return False  # unanswerable questions have no expected source
    if spec["doc"].lower() not in chunk.document.lower():
        return False
    if spec.get("page") is not None and chunk.page != spec["page"]:
        return False
    return True


def answer_pass() -> list[dict]:
    """Live end-to-end pass: full course assistant (hybrid retrieval + rerank
    + validated LLM answers) over the real index. Requires class API keys."""
    from course_assistant.config import load_config
    from course_assistant.service import CourseAssistant

    if not load_config().keys_present:
        sys.exit("--with-answers requires class API keys in .env (offline mode is keyword-only)")

    svc = CourseAssistant()
    rows = []
    for spec in QUESTION_SET:
        t0 = time.perf_counter()
        try:
            ans = svc.ask(spec["question"])
            source_hit = 0
            for s in ans.sources:
                if spec.get("doc") and spec["doc"].lower() in s.document.lower():
                    if spec.get("page") is None or s.page == spec["page"]:
                        source_hit = 1
                        break
            rows.append({
                "config": "live_full_pipeline",
                "question_id": spec["id"],
                "kind": spec["kind"],
                "recall_at_k": None,  # retrieval-side metric; see config rows
                "hit_position": None,
                "retrieval_seconds": round(time.perf_counter() - t0, 2),
                "top_documents": [s.document for s in ans.sources[:3]],
                "top_pages": [s.page for s in ans.sources[:3]],
                "answer": ans.answer,
                "missing_info": ans.missing_info,
                "source_hit": source_hit,
                "note": f"{len(ans.sources)} cited source(s)",
            })
        except Exception as exc:  # never let one flaky question kill the pass
            rows.append({
                "config": "live_full_pipeline",
                "question_id": spec["id"],
                "kind": spec["kind"],
                "recall_at_k": None,
                "hit_position": None,
                "retrieval_seconds": round(time.perf_counter() - t0, 2),
                "top_documents": [],
                "top_pages": [],
                "answer": None,
                "missing_info": None,
                "source_hit": 0,
                "note": f"error: {type(exc).__name__}",
            })
    return rows


def run_config(name: str, chunk_config: dict, materials_dir: Path, images_root: Path, k: int) -> list[dict]:
    chunks = load_corpus(materials_dir, images_root, chunk_config)
    retriever = build_bm25(chunks)
    rows = []
    for spec in QUESTION_SET:
        t0 = time.perf_counter()
        top = retrieve(retriever, chunks, spec["question"], k)
        elapsed = round(time.perf_counter() - t0, 4)
        hits = [expected_hit(spec, c) for c, _ in top]
        rows.append({
            "config": name,
            "question_id": spec["id"],
            "kind": spec["kind"],
            "recall_at_k": 1 if any(hits) else 0,
            "hit_position": (hits.index(True) + 1) if any(hits) else None,
            "retrieval_seconds": elapsed,
            "top_documents": [c.document for c, _ in top[:3]],
            "top_pages": [c.page for c, _ in top[:3]],
            "note": "",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeatable RAG retrieval evaluation (requirement 5)")
    ap.add_argument("--configs", default="chunk_page,chunk_recursive")
    ap.add_argument("--materials", default="materials")
    ap.add_argument("--images-root", default="data/images")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out", default="results/eval_local.json")
    ap.add_argument("--save-md", action="store_true")
    ap.add_argument("--with-answers", action="store_true",
                    help="answer-quality pass; requires class keys + QA pipeline (issues 4-5)")
    args = ap.parse_args()

    all_rows: list[dict] = []

    if args.with_answers:
        print("=== live full-pipeline answer pass ===")
        answer_rows = answer_pass()
        all_rows.extend(answer_rows)
        for r in answer_rows:
            print(f"  {r['question_id']:20} source_hit={r['source_hit']} "
                  f"missing={r['missing_info']} t={r['retrieval_seconds']}s "
                  f"answer: {(r['answer'] or r['note'])[:80]}")

    materials_dir = Path(args.materials)
    images_root = Path(args.images_root)
    for name in [c.strip() for c in args.configs.split(",")]:
        if name not in CHUNK_CONFIGS:
            sys.exit(f"unknown config {name!r}; choose from {', '.join(CHUNK_CONFIGS)}")
        print(f"\n=== config: {name} ===")
        rows = run_config(name, CHUNK_CONFIGS[name], materials_dir, images_root, args.k)
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['question_id']:20} recall@k={r['recall_at_k']} "
                  f"pos={r['hit_position']} t={r['retrieval_seconds']}s "
                  f"top: {', '.join(f'{d}#{p}' for d, p in zip(r['top_documents'], r['top_pages']))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {out} ({len(all_rows)} rows)")

    if args.save_md:
        md_path = out.with_suffix(".md")
        with_answers = any("answer" in r for r in all_rows)
        cols = (["config", "question", "kind", "recall@k", "hit pos", "time (s)"]
                + (["answer (truncated)", "source hit"] if with_answers else []))
        lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for r in all_rows:
            row = [r["config"], r["question_id"], r["kind"],
                   r["recall_at_k"] if r["recall_at_k"] is not None else "-",
                   r["hit_position"] if r.get("hit_position") else "-",
                   r["retrieval_seconds"]]
            if with_answers:
                ans = r.get("answer")
                row.append((ans or r.get("note", ""))[:60].replace("|", "/"))
                row.append(r.get("source_hit", "-"))
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        md_path.write_text("\n".join(lines) + "\n")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
