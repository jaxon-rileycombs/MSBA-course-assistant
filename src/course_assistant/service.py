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

import shutil
from dataclasses import dataclass
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


@dataclass
class ServiceDirs:
    """Filesystem wiring paths (kept separate from network config)."""

    vector_dir: str = "vectorstore"
    images_root: str = "data/images"
    registry_path: str = "data/registry.json"
    quiz_dir: str = "data/quizzes"


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
        """Sample up to n evidence chunks across the selected document(s)."""
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
                )
            )
        out: list[Evidence] = []
        doc_items = list(by_doc.items())
        total = sum(len(v) for _, v in doc_items)
        for i in range(min(n, total)):
            chunks = doc_items[i % len(doc_items)][1]
            if chunks:
                out.append(chunks.pop(0))
        return out

    # -- ask ---------------------------------------------------------------

    def ask(self, question: str, scope: str = ALL_SCOPE, topic: str = "") -> Answer:
        """Retrieve -> fuse -> (rerank) -> grounded answer with sources."""
        query = f"{question} {topic}".strip() if topic else question
        evidence = self._scope_evidence(self._hybrid.retrieve(query, k=16), scope)
        try:
            return self.answerer.answer(question, evidence)
        except AnswerError as exc:
            return Answer(answer=str(exc), sources=[], missing_info=False)

    # -- quiz --------------------------------------------------------------

    def generate_quiz(self, scope: str = ALL_SCOPE, topic: str = "", n: int = 5,
                      title: str = "") -> tuple[str, dict]:
        chunks = self._scope_chunks(scope, n * 2)
        if not chunks:
            raise RuntimeError(
                "No course materials are indexed yet — add documents in the "
                "Materials tab first."
            )
        if topic:
            filtered = [c for c in chunks if topic.lower() in c.text.lower()]
            if filtered:
                chunks = filtered
        questions = self.quizgen.generate(chunks[: n * 2])
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
