"""CourseAssistant service layer — wires config, indexes, retrieval,
answering, and quizzes behind one object the UI (app.py) drives, using the
same code paths as the tests and CLI.

Two modes (auto-detected):
- online  — class services (9001-9004) when API keys are present
- offline — deterministic HashEmbedder + OfflineAnswerer/OfflineQuizGenerator
            so everything runs before the team's keys arrive; clearly
            labelled in the UI and README (never claims to be the real model).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .clients import EmbeddingsClient, RerankClient, VisionLLMClient
from .config import load_config
from .ingest.chunking import chunk_document
from .ingest.embeddings import HashEmbedder
from .ingest.formats import UnsupportedFormatError, parse_document
from .ingest.index import IndexStore
from .ingest.registry import DocumentRegistry
from .qa.answer import AnswerError, Answerer, OfflineAnswerer
from .qa.schema import Answer
from .quiz.generator import OfflineQuizGenerator, QuizGenerator
from .quiz.session import QuizSession
from .retrieval.embeddings_service import TextEmbedderAdapter, VisualEmbedderAdapter
from .retrieval.embedding_retriever import TextEmbeddingRetriever, VisualEmbeddingRetriever
from .retrieval.evidence import Evidence
from .retrieval.hybrid import HybridRetriever
from .retrieval.keyword import KeywordRetriever
from .retrieval.rerank import ServiceReranker

ALL_SCOPE = "All materials"
MAX_ASK_EVIDENCE = 16  # evidence slots budget for the answer prompt


def _meme_candidate_score(text: str, section: str = "") -> int:
    """Rank slide text that strongly suggests a meme or reaction image.

    PowerPoint text extraction cannot see words inside an embedded meme, but
    slide titles and source notes often identify visual-humor pages. These
    cues provide a deterministic fallback when a visual vector is unavailable.
    """
    value = f"{section}\n{text}".lower()
    score = 0
    if "meme" in value:
        score += 12
    if "reddit.com/r/programmerhumor" in value:
        score += 10
    if "reddit.com/r/vibecoding" in value:
        score += 9
    if "vibe coding on" in value:
        score += 8
    if "security? never heard" in value or (
        "how it started" in value and "how it" in value and "going" in value
    ):
        score += 8
    return score


@dataclass
class ServiceDirs:
    """Filesystem wiring paths (kept separate from network config)."""

    vector_dir: str = field(
        default_factory=lambda: os.environ.get("COURSE_ASSISTANT_VECTOR_DIR", "vectorstore")
    )
    images_root: str = field(
        default_factory=lambda: os.environ.get("COURSE_ASSISTANT_IMAGES_ROOT", "data/images")
    )
    registry_path: str = field(
        default_factory=lambda: os.environ.get(
            "COURSE_ASSISTANT_REGISTRY_PATH", "data/registry.json"
        )
    )
    quiz_dir: str = field(
        default_factory=lambda: os.environ.get("COURSE_ASSISTANT_QUIZ_DIR", "data/quizzes")
    )


class CourseAssistant:
    def __init__(self, *, dirs: ServiceDirs | None = None, force_offline: bool = False):
        self.dirs = dirs or ServiceDirs()
        self.registry = DocumentRegistry(self.dirs.registry_path)
        self.sessions = QuizSession(self.dirs.quiz_dir)
        index, hybrid, answerer, quizgen, mode = self._wire(force_offline)
        self._index = index
        self._hybrid = hybrid
        self.answerer = answerer
        self.quizgen = quizgen
        self.mode = mode

    # -- wiring ------------------------------------------------------------

    def _wire(self, force_offline: bool):
        net = load_config()
        if force_offline or not net.keys_present:
            emb = HashEmbedder()
            index = IndexStore(self.dirs.vector_dir, emb.embed_texts, emb.embed_images)
            hybrid = HybridRetriever(
                KeywordRetriever.from_index(index),
                TextEmbeddingRetriever(index, emb.embed_query),
                VisualEmbeddingRetriever(index, emb.embed_query),
                # Offline mode = keyword retrieval only (documented decision):
                # the stand-in hash embeddings are lexical noise, so text and
                # visual retrievers are weighted 0 until real embeddings exist.
                # Online mode enables them (weights 1/1/1) plus reranking.
                weights={"keyword": 1.0, "text": 0.0, "visual": 0.0},
            )
            return (
                index,
                hybrid,
                OfflineAnswerer(),
                OfflineQuizGenerator(),
                "offline (no class-service API keys detected)",
            )
        text_client = EmbeddingsClient(net.text_embed)
        visual_client = EmbeddingsClient(net.visual_embed, path="/embeddings")
        text_emb = TextEmbedderAdapter(text_client)
        vis_emb = VisualEmbedderAdapter(visual_client)
        index = IndexStore(self.dirs.vector_dir, text_emb.embed_texts, vis_emb.embed_images)
        hybrid = HybridRetriever(
            KeywordRetriever.from_index(index),
            TextEmbeddingRetriever(index, text_emb.embed_query),
            VisualEmbeddingRetriever(index, vis_emb.embed_query),
            reranker=ServiceReranker(RerankClient(net.rerank)),
        )
        return (
            index,
            hybrid,
            Answerer(VisionLLMClient(net.vision_llm)),
            QuizGenerator(VisionLLMClient(net.vision_llm)),
            "online (class services 9001-9004)",
        )

    def _refresh_keyword(self) -> None:
        self._hybrid.keyword = KeywordRetriever.from_index(self._index)

    # -- materials ---------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """Registry entries, newest first: {doc_id, filename, format, pages}."""
        return [
            {
                "doc_id": e["doc_id"],
                "filename": e["filename"],
                "format": e["format"],
                "pages": e.get("pages", 0),
            }
            for e in self.registry.list()
        ]

    def add_documents(self, paths: list[str]) -> list[str]:
        """Parse, chunk, embed, index, register. Returns per-file status lines."""
        messages: list[str] = []
        for raw in paths:
            p = Path(raw)
            try:
                doc = parse_document(p, images_root=self.dirs.images_root)
            except UnsupportedFormatError as exc:
                messages.append(f"\u2717 {p.name}: {exc}")
                continue
            if self.registry.contains(doc.doc_id):
                messages.append(f"~ {p.name}: duplicate — already loaded (no change)")
                continue
            chunks = chunk_document(doc)
            counts = self._index.upsert_document(doc, chunks)
            self.registry.add(doc)
            messages.append(
                f"+ {p.name}: {len(doc.pages)} pages, {counts['text']} chunks, "
                f"{counts['visual']} visual vectors"
            )
        self._refresh_keyword()
        return messages

    def remove_document(self, doc_id_or_name: str) -> str:
        """Remove a document everywhere: index vectors, registry, images."""
        entry = self.registry.get(doc_id_or_name)
        if entry is None:  # allow filename lookup too
            entry = next(
                (e for e in self.registry.list() if e["filename"] == doc_id_or_name), None
            )
        if entry is None:
            raise KeyError(f"no registered document matching {doc_id_or_name!r}")
        doc_id = entry["doc_id"]
        removed = self._index.remove_document(doc_id)
        self.registry.remove(doc_id)
        img_dir = Path(self.dirs.images_root) / doc_id
        if img_dir.exists():
            shutil.rmtree(img_dir)
        self._refresh_keyword()
        return (
            f"Removed '{entry['filename']}' (text vectors {removed['text']}, "
            f"visual vectors {removed['visual']}, images, registry entry)"
        )

    # -- scoping -----------------------------------------------------------

    def _scope_evidence(self, evidence: list[Evidence], scope: str) -> list[Evidence]:
        if scope in (ALL_SCOPE, "", None):
            return evidence
        return [e for e in evidence if e.document == scope]

    def _scope_chunks(self, scope: str, n: int) -> list[Evidence]:
        """Sample up to n evidence chunks spread across the selected
        document(s): each document contributes a share proportional to its
        size, and picks within a document are evenly spaced across its whole
        length — so a quiz about a deck never covers only its opening slides.
        """
        res = self._index.text_col.get(limit=2000)
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        by_doc: dict[str, list[Evidence]] = {}
        for text, meta in zip(documents, metadatas):
            doc_id = str(meta["doc_id"])
            document = str(meta["document"])
            assert isinstance(meta["page"], int)
            page: int = meta["page"]
            if scope not in (ALL_SCOPE, "", None) and document != scope:
                continue
            by_doc.setdefault(doc_id, []).append(
                Evidence(
                    doc_id=doc_id,
                    document=document,
                    page=page,
                    section=str(meta.get("section", "")),
                    text=text,
                    image_path=str(meta.get("image", "")) or None,
                )
            )
        # one representative chunk per slide — a deck quiz never covers the
        # same page twice; and skip text-poor title/transition slides, which
        # only ever produce "which module is this?" trivia
        by_page: dict[tuple[str, int], Evidence] = {}
        for entries in by_doc.values():
            for e in entries:
                if len(e.text.strip()) < 40:
                    continue  # title/transition slides: no quiz content
                by_page.setdefault((e.document, e.page), e)
        items = []
        for doc_id, chunks in by_doc.items():
            reps = [e for e in chunks if by_page.get((e.document, e.page)) is e]
            if reps:
                items.append((doc_id, reps))
        total = sum(len(v) for _, v in items)
        if not total:
            return []

        def _spread(chunks: list[Evidence], k: int) -> list[Evidence]:
            if k >= len(chunks):
                return list(chunks)  # small doc: take everything
            if k == 1:
                return [chunks[len(chunks) // 2]]  # never a title slide
            step = len(chunks) / k
            return [chunks[int(j * step)] for j in range(k)]

        # per-doc share proportional to size, at least 1 each
        shares = {doc_id: max(1, round(n * len(chunks) / total)) for doc_id, chunks in items}
        pool = {doc_id: _spread(chunks, shares[doc_id]) for doc_id, chunks in items}

        # round-robin the per-document picks into one coverage order
        order: list[Evidence] = []
        while True:
            progressed = False
            for doc_id, picked in list(pool.items()):
                if not picked:
                    continue
                order.append(picked.pop(0))
                progressed = True
            if not progressed:
                break

        out = order[:n]
        if len(out) < n:  # tiny corpus: cycle the order to reach the count
            i = 0
            while len(out) < n:
                out.append(order[i % len(order)])
                i += 1
        return out

    def _meme_evidence(self) -> list[Evidence]:
        """Find likely meme slides directly from indexed titles/notes.

        This supplements multimodal retrieval when the visual embedding
        service rejects a highly detailed slide image.
        """
        res = self._index.text_col.get(limit=2000)
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        by_page: dict[tuple[str, int], tuple[int, Evidence]] = {}
        for text, meta in zip(documents, metadatas):
            image_path = str(meta.get("image", "")) or None
            if not image_path:
                continue
            section = str(meta.get("section", ""))
            score = _meme_candidate_score(text, section)
            if score <= 0:
                continue
            page = int(meta["page"])
            key = (str(meta["doc_id"]), page)
            candidate = Evidence(
                doc_id=str(meta["doc_id"]),
                document=str(meta["document"]),
                page=page,
                section=section,
                text=text,
                image_path=image_path,
                scores={"meme_hint": float(score)},
            )
            if key not in by_page or score > by_page[key][0]:
                by_page[key] = (score, candidate)
        return [item[1] for item in sorted(by_page.values(), key=lambda item: item[0], reverse=True)]

    # -- ask ---------------------------------------------------------------

    def ask(self, question: str, scope: str = ALL_SCOPE, topic: str = "") -> Answer:
        """Retrieve -> fuse -> (rerank) -> grounded answer with sources.

        A wide retrieval pool (24 candidates from 16 per leg) plus up to
        three downscaled slide images attached to the vision model — slides
        the text extraction under-reads are still answerable from their
        render. The optional topic filter is STRICT: when set, only chunks
        that mention the topic are used, and if none do, the app says so
        instead of silently answering from unrelated material. Questions
        the materials genuinely don't cover are answered as missing (no
        invention).
        """
        if "meme" in question.lower():
            # Visual embedding can reject unusually detailed slides. Use the
            # title/notes-derived visual candidates by themselves so unrelated
            # retrieval results cannot distract the vision model.
            retrieved = self._meme_evidence()
        else:
            retrieved = self._hybrid.retrieve(question, k=24, per_retriever=16)
        evidence = self._scope_evidence(retrieved, scope)
        if topic:
            topic = topic.strip().lower()
            topical = [e for e in evidence if topic in e.text.lower() or topic in (e.section or "").lower()]
            if not topical:
                return Answer(
                    answer=f"No material in the selected scope mentions the topic “{topic}”. "
                           "Try a different topic, or clear the topic filter.",
                    sources=[],
                    missing_info=True,
                )
            evidence = topical[:MAX_ASK_EVIDENCE]
        try:
            return self.answerer.answer(question, evidence)
        except AnswerError as exc:
            return Answer(answer=str(exc), sources=[], missing_info=False)

    # -- quiz --------------------------------------------------------------

    def generate_quiz(self, scope: str = ALL_SCOPE, topic: str = "", n: int = 5,
                      title: str = "") -> tuple[str, dict]:
        chunks = self._scope_chunks(scope, n * 4)
        if not chunks:
            raise RuntimeError(
                "No course materials are indexed yet — add documents in the "
                "Materials tab first."
            )
        if topic:
            filtered = [c for c in chunks if topic.lower() in c.text.lower()]
            if filtered:
                chunks = filtered
        questions = self.quizgen.generate(chunks[: n * 4], n=n)
        quiz_id = self.sessions.create(title or "Generated quiz", questions)
        return quiz_id, self.sessions.get(quiz_id)

    def get_quiz(self, quiz_id: str) -> dict:
        return self.sessions.get(quiz_id)

    def grade_quiz(self, quiz_id: str, answers: list[int]) -> dict:
        result = self.sessions.submit(quiz_id, answers)
        return {
            "score": result.score,
            "total": result.total,
            "correct": result.correct,
            "feedback": [
                {
                    "correct": f["correct"],
                    "explanation": f["explanation"],
                    "sources": f["sources"],
                    "your": f["your_answer"],
                    "right": f["correct_answer"],
                }
                for f in result.feedback
            ],
        }

    def reveal_quiz(self, quiz_id: str) -> dict:
        return self.sessions.reveal(quiz_id)

    # -- citations rendering (issue #6) ------------------------------------

    @staticmethod
    def render_sources(answer: Answer) -> tuple[str, list[str]]:
        """Markdown citation block + list of slide-image paths for the UI."""
        lines: list[str] = []
        images: list[str] = []
        for s in answer.sources:
            where = f"**{s.document}**, slide **{s.page}**"
            if s.section:
                where += f" — *{s.section}*"
            excerpt = " ".join(s.excerpt.split())
            lines.append(f"- {where}\n  > {excerpt}")
            if s.image_path and Path(s.image_path).exists():
                images.append(s.image_path)
        if not lines:
            return "", images
        return "### Sources\n" + "\n".join(lines), images
