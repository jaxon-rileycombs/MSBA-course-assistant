"""Parse course materials into text + original page/slide images.

Supported formats: PDF, PPTX, DOCX, MD, TXT, HTML.

Design notes (assignment requirement 1):
- Document identity is the sha256 of the file bytes, so loading the same
  file twice is a no-op (the registry/ingest layer handles dedupe).
- Parsing preserves original images: PDF pages are rendered to PNG
  (PyMuPDF, tolerant of malformed xrefs), PPTX embedded pictures are
  extracted per slide, DOCX inline images are extracted. Markdown/HTML
  images are NOT extracted (limitation, see README).
- Images are written under ``images_root / <doc_id> /`` so removal of a
  document can delete exactly its directory.
"""

from __future__ import annotations

import html.parser
import io
from pathlib import Path

try:  # pymupdf >= 1.24: modern import name
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore  # noqa: F401  (older alias)

from .documents import SUPPORTED_FORMATS, Document, Page, file_sha256


class UnsupportedFormatError(ValueError):
    def __init__(self, ext: str):
        self.ext = ext
        super().__init__(
            f"Unsupported format {ext!r}. We accept: "
            + ", ".join(sorted(SUPPORTED_FORMATS))
        )


_CLEAN = str.maketrans({"\x0b": "\n", "\x0c": "\n", "\r": " "})


def _clean_text(text: str) -> str:
    """Normalize control characters from office files (soft line breaks)."""
    return text.translate(_CLEAN)


class _TextStripper(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._title = ""

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")
        elif tag == "p":
            self.parts.append("\n")
        elif tag == "title":
            self._in_title = True  # not used; body text only

    def text(self) -> str:
        return " ".join(p.strip() for p in self.parts if p.strip())


def parse_document(path: str | Path, images_root: str | Path) -> Document:
    """Parse one file into a :class:`Document`, extracting images."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(ext)
    images_root = Path(images_root)
    doc_id = file_sha256(path)

    if ext == ".pdf":
        pages = _parse_pdf(path, doc_id, images_root)
    elif ext == ".pptx":
        pages = _parse_pptx(path, doc_id, images_root)
    elif ext == ".docx":
        pages = _parse_docx(path, doc_id, images_root)
    elif ext == ".html" or ext == ".htm":
        pages = _parse_html(path)
    else:  # .md / .txt
        text = path.read_text(errors="replace")
        pages = [Page(number=1, text=text)]

    return Document(
        doc_id=doc_id,
        filename=path.name,
        format=ext,
        source_path=str(path),
        pages=pages,
    )


# ---------------------------------------------------------------- PDF

def _parse_pdf(path: Path, doc_id: str, images_root: Path) -> list[Page]:
    pages: list[Page] = []
    out_dir = images_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(path)
    for i, page in enumerate(doc, 1):
        text = _clean_text(page.get_text("text")).strip()
        # Render the original page as an image (visual evidence + visual index).
        img_path = out_dir / f"page_{i}.png"
        page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(img_path)
        pages.append(Page(number=i, text=text, image_paths=[str(img_path)]))
    doc.close()
    return pages


# ---------------------------------------------------------------- PPTX

from pptx import Presentation as _Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER


def _iter_shapes(shapes):
    """Iterate shapes recursively, descending into group shapes."""
    for shape in shapes:
        yield shape
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)


def _slide_title(slide) -> str:
    for shape in _iter_shapes(slide.shapes):
        if shape.is_placeholder:
            try:
                if shape.placeholder_format.type in (
                    PP_PLACEHOLDER.TITLE,
                    PP_PLACEHOLDER.CENTER_TITLE,
                ):
                    return _clean_text(shape.text_frame.text.strip())
            except Exception:
                pass
    for shape in _iter_shapes(slide.shapes):
        if shape.has_text_frame and shape.text_frame.text.strip():
            return _clean_text(shape.text_frame.text.strip())
    return ""


def _slide_body(slide) -> str:
    parts: list[str] = []
    for shape in _iter_shapes(slide.shapes):
        if not shape.has_text_frame:
            continue
        text = _clean_text(shape.text_frame.text.strip())
        if not text:
            continue
        if shape.is_placeholder:
            try:
                if shape.placeholder_format.type in (
                    PP_PLACEHOLDER.TITLE,
                    PP_PLACEHOLDER.CENTER_TITLE,
                ):
                    continue  # title is captured separately as the section
            except Exception:
                pass
        parts.append(text)
    return "\n".join(parts)


def _parse_pptx(path: Path, doc_id: str, images_root: Path) -> list[Page]:
    prs = _Presentation(str(path))
    out_dir = images_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Page] = []
    for n, slide in enumerate(prs.slides, 1):
        title = _slide_title(slide)
        text = _slide_body(slide)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                text = f"{text}\nNotes: {notes}" if text else f"Notes: {notes}"
        image_paths: list[str] = []
        idx = 0
        for shape in _iter_shapes(slide.shapes):
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                img = shape.image
            except Exception:
                continue
            idx += 1
            img_path = out_dir / f"slide_{n}_{idx}.{img.ext}"
            img_path.write_bytes(img.blob)
            image_paths.append(str(img_path))
        pages.append(Page(number=n, text=text, section=title, image_paths=image_paths))
    return pages


# ---------------------------------------------------------------- DOCX

from docx import Document as _DocxDocument  # noqa: E402


def _parse_docx(path: Path, doc_id: str, images_root: Path) -> list[Page]:
    doc = _DocxDocument(str(path))
    out_dir = images_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    section = ""
    for para in doc.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        style = para.style.name if para.style is not None else ""
        if style.lower().startswith("heading") and not section:
            section = t
        parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            parts.append(" | ".join(c for c in cells if c))

    image_paths: list[str] = []
    for i, shape in enumerate(doc.inline_shapes, 1):
        try:
            blip = shape._inline.graphic.graphicData.pic.blipFill.blip
            part = doc.part.related_parts[blip.embed]
            ext = part.content_type.split("/")[-1] or "bin"
            img_path = out_dir / f"img_{i}.{ext}"
            img_path.write_bytes(part.blob)
            image_paths.append(str(img_path))
        except Exception:
            continue

    text = "\n".join(parts)
    return [Page(number=1, text=text, section=section, image_paths=image_paths)]


# ---------------------------------------------------------------- HTML

def _parse_html(path: Path) -> list[Page]:
    parser = _TextStripper()
    parser.feed(path.read_text(errors="replace"))
    return [Page(number=1, text=parser.text())]
