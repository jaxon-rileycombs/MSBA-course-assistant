#!/usr/bin/env python3
"""Material ingestion CLI — the pipeline behind "add / remove documents".

Commands:
    ingest  <dir | files...>   parse + chunk + index new materials
                              (repeat uploads are a no-op — sha256 dedupe)
    list                        show registered documents (newest first)
    remove  <doc_id | filename> remove a document everywhere: text index,
                              visual index, extracted images, registry

Defaults (all overridable): materials parsed from the given paths, images in
data/images/, vectors in vectorstore/, registry in data/registry.json.

Embeddings: by default uses the deterministic local HashEmbedder (no API
keys needed; NOT semantically meaningful). Pass --real-embeddings to use the
class services once keys are in .env (visual-embedding request format is
finalized in issue 4).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_assistant.ingest.chunking import chunk_document  # noqa: E402
from course_assistant.ingest.embeddings import HashEmbedder  # noqa: E402
from course_assistant.ingest.formats import UnsupportedFormatError, parse_document  # noqa: E402
from course_assistant.ingest.index import IndexStore  # noqa: E402
from course_assistant.ingest.registry import DocumentRegistry  # noqa: E402

DEFAULTS = {
    "vector_dir": "vectorstore",
    "images_root": "data/images",
    "registry_path": "data/registry.json",
}


def build_index(args) -> IndexStore:
    text_embedder = HashEmbedder().embed_texts
    image_embedder = HashEmbedder().embed_images
    if getattr(args, "real_embeddings", False):
        from course_assistant.clients import EmbeddingsClient
        from course_assistant.config import load_config
        from course_assistant.retrieval.embeddings_service import (
            TextEmbedderAdapter,
            VisualEmbedderAdapter,
        )

        cfg = load_config()
        if cfg.keys_present:
            text_embedder = TextEmbedderAdapter(EmbeddingsClient(cfg.text_embed)).embed_texts
            image_embedder = VisualEmbedderAdapter(EmbeddingsClient(cfg.visual_embed)).embed_images
            print("embeddings: class services (9002/9003)")
        else:
            print("  ! --real-embeddings given but no API keys in .env — using local stand-in embeddings")
    return IndexStore(
        args.vector_dir,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )


def resolve_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            print(f"  ! not found: {raw}")
    return files


def cmd_ingest(args) -> int:
    files = resolve_files(args.paths)
    if not files:
        print("no files given")
        return 1
    index = build_index(args)
    registry = DocumentRegistry(args.registry_path)
    added = skipped = errors = 0
    for f in files:
        try:
            doc = parse_document(f, args.images_root)
        except UnsupportedFormatError as exc:
            print(f"  ! {f.name}: {exc}")
            errors += 1
            continue
        except Exception as exc:  # defensive: one bad file must not kill the run
            print(f"  ! {f.name}: parse failed: {exc}")
            errors += 1
            continue
        if registry.contains(doc.doc_id):
            print(f"  ~ {f.name}: duplicate (doc_id {doc.doc_id[:10]}…) — skipped")
            skipped += 1
            continue
        chunks = chunk_document(doc)
        counts = index.upsert_document(doc, chunks)
        registry.add(doc)
        print(f"  + {f.name}: {doc.total_pages} pages, {len(chunks)} chunks, "
              f"{counts['visual']} visual vectors")
        added += 1
    print(f"\nadded {added}, skipped {skipped}, errors {errors} | "
          f"index counts {index.counts()} | registry {len(registry)} docs")
    return 0


def cmd_list(args) -> int:
    registry = DocumentRegistry(args.registry_path)
    if not registry.list():
        print("no documents registered")
        return 0
    for e in registry.list():
        print(f"  {e['doc_id'][:12]}… {e['filename']:50} {e['format']:6} "
              f"{e['pages']} pages  ingested {e['ingest_time']}")
    return 0


def cmd_remove(args) -> int:
    registry = DocumentRegistry(args.registry_path)
    target = args.target
    entry = registry.get(target)
    if entry is None:  # maybe they passed a filename
        matches = [e for e in registry.list() if e["filename"] == target]
        if len(matches) == 1:
            entry = matches[0]
    if entry is None:
        print(f"! no registered document matches {target!r}")
        return 1
    doc_id = entry["doc_id"]
    index = build_index(args)
    removed = index.remove_document(doc_id)
    registry.remove(doc_id)
    images_dir = Path(args.images_root) / doc_id
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)
    print(f"  - removed {entry['filename']} (text {removed['text']}, visual "
          f"{removed['visual']} vectors, images dir {'deleted' if not images_dir.exists() else 'missing'})")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="parse + chunk + index materials")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.set_defaults(func=cmd_ingest)

    p_list = sub.add_parser("list", help="list registered documents")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="remove a document everywhere")
    p_remove.add_argument("target", help="doc_id or filename")
    p_remove.set_defaults(func=cmd_remove)

    for p in (p_ingest, p_list, p_remove):
        p.add_argument("--vector-dir", default=DEFAULTS["vector_dir"])
        p.add_argument("--images-root", default=DEFAULTS["images_root"])
        p.add_argument("--registry", default=DEFAULTS["registry_path"])
        p.add_argument("--real-embeddings", action="store_true",
                       help="use class-service embeddings (requires keys); default: local fake")

    args = ap.parse_args()
    args.vector_dir, args.images_root, args.registry_path = (
        args.vector_dir, args.images_root, getattr(args, "registry", DEFAULTS["registry_path"]),
    )
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
