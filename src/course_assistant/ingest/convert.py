"""Optional PowerPoint → PDF conversion via LibreOffice (headless).

Why: direct PPTX parsing (python-pptx) is the primary path and needs no
extra software. RENDERING full slides as images is occasionally needed
(visual index / screenshots); LibreOffice conversion is the documented way
to get faithful slide images, and converting the deck to PDF also lets other
tools (PyMuPDF) render each slide.

Install (macOS):  brew install --cask libreoffice   (provides `soffice`)
Workaround:      export the deck to PDF manually before uploading.

Usage:
    python -m course_assistant.ingest.convert materials/ [out_dir]

The script checks that converted PDF page count == deck slide count, so a
broken conversion is caught instead of silently producing wrong pages.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_soffice() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_pptx_to_pdf(pptx: Path, out_dir: Path, soffice: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    return out_dir / (pptx.stem + ".pdf")


def _page_count(pdf: Path) -> int:
    import pymupdf

    with pymupdf.open(str(pdf)) as doc:
        return doc.page_count


def convert_decks(materials_dir: str | Path, out_dir: str | Path | None = None) -> list[tuple[str, int, int, bool]]:
    """Convert every .pptx under materials_dir; returns (name, slides, pages, ok)."""
    materials_dir = Path(materials_dir)
    out_dir = Path(out_dir) if out_dir else materials_dir / "_converted"
    soffice = find_soffice()
    if soffice is None:
        raise RuntimeError(
            "LibreOffice not found. Install it with `brew install --cask libreoffice` "
            "or export decks to PDF manually (acceptable workaround)."
        )
    results: list[tuple[str, int, int, bool]] = []
    for pptx in sorted(materials_dir.glob("*.pptx")):
        from pptx import Presentation

        slides = len(Presentation(str(pptx)).slides)
        pdf = convert_pptx_to_pdf(pptx, out_dir, soffice)
        pages = _page_count(pdf)
        results.append((pptx.name, slides, pages, pages == slides))
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    try:
        results = convert_decks(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    ok_all = True
    for name, slides, pages, ok in results:
        print(f"{name}: {slides} slides -> {pages} pages [{'OK' if ok else 'MISMATCH'}]")
        ok_all = ok_all and ok
    if not ok_all:
        print("Converted-slide check FAILED: some decks lost pages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
