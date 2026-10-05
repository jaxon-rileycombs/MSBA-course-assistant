"""Gradio interface.

Run:  python app.py   (from the repo root; SHARE=1 python app.py for a public link)

Three tabs — Ask, Quiz, Materials — driven by :class:`CourseAssistant`.
Runs "online" with class-service keys present, otherwise drops into an
explicitly-labelled offline mode (deterministic stand-ins) so the app is
usable end-to-end before the keys arrive.
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from course_assistant.clients import ClassAPIError  # noqa: E402
from course_assistant.service import ALL_SCOPE, CourseAssistant  # noqa: E402

MAX_QUIZ = 5
LETTERS = "ABCDEFGH"

FUN_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
)

FUN_CSS = """
/* hero title — gradient "LLM" treatment */
#hero h1 {
  font-size: 2rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
  background: linear-gradient(90deg, #7c3aed 0%, #0d9488 55%, #f59e0b 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  margin-bottom: 0.2rem;
}
#hero { padding: 0.1rem 0 0.6rem; }
#hero p { color: var(--body-text-color-subdued); }
/* sample-question chips */
.chip-row { margin-top: 0.4rem; }
.chip-row + footer { display: block !important; }
#app-footer {
  margin-top: 1.4rem; padding-top: 0.8rem;
  border-top: 1px dashed var(--border-color-primary, #d1d5db);
  color: var(--body-text-color-subdued);
  font-size: 0.85rem;
}
"""


def _sample_buttons(question) -> None:
    """Clickable sample questions — fill the ask box with one tap."""
    samples = [
        ("🎯 Final exam worth?", "What is the final exam worth?"),
        ("🧮 Grade breakdown", "How is my grade calculated?"),
        ("🖼️ Find the meme", "Where is the vibe-coding meme in the slides?"),
        ("⏰ Late policy", "What happens if I submit an assignment late?"),
    ]
    with gr.Row(elem_classes=["chip-row"]):
        for label, q in samples:
            gr.Button(label, size="sm").click(lambda q=q: q, inputs=None, outputs=question)


def _header_markdown(service) -> str:
    mode_icon = "🟢" if "offline" not in service.mode else "🟡"
    return (
        "# 🎓✨ MBAX 6418 Course Assistant\n"
        "Hybrid RAG over your course materials — **ask** it anything, **quiz** yourself "
        "on the slides, and it answers with the exact slide, excerpt, and image in hand. "
        "No hallucinating, no guessing: if it isn't in the material, it says so. 🤝\n\n"
        f"Mode: **{service.mode}** {mode_icon}"
    )


def _friendly(exc: Exception) -> str:
    """Turn any exception into a short, key-free user message."""
    if isinstance(exc, ClassAPIError):
        return "the class service is temporarily unavailable — please try again"
    text = str(exc).strip()
    return text if text else type(exc).__name__


def _file_value(files) -> list[str]:
    """gr.File values are dicts (name/orig_name) in current Gradio; handle both."""
    out = []
    for f in files or []:
        out.append(f["name"] if isinstance(f, dict) else str(f))
    return out


def scope_choices(service: CourseAssistant) -> list[str]:
    return [ALL_SCOPE] + [d["filename"] for d in service.list_documents()]


# ---- Ask -------------------------------------------------------------

def do_ask(service, question: str, scope: str, topic: str):
    if not question.strip():
        gr.Info("Type a question first.")
        return "", []
    try:
        answer = service.ask(question, scope=scope, topic=topic)
    except Exception as exc:  # never surface a raw error to the UI
        return f"*Couldn't answer right now — {_friendly(exc)}.*", []
    sources_md, images = service.render_sources(answer)
    body = f"**{answer.answer}**" + (
        "\n\n*The materials don't cover this — nothing was invented.*" if answer.missing_info else ""
    )
    return body + ("\n\n" + sources_md if sources_md else ""), images


# ---- Quiz ------------------------------------------------------------

def do_generate(service, scope: str, topic: str, n: int, qid_state):
    try:
        quiz_id, view = service.generate_quiz(scope=scope, topic=topic, n=n)
    except Exception as exc:  # model flake or service hiccup → friendly, never a 500
        gr.Info(f"Could not generate the quiz: {_friendly(exc)}")
        empty = [gr.update(visible=False)] * MAX_QUIZ
        return qid_state, *empty, "", ""
    radios = []
    for i in range(MAX_QUIZ):
        if i < len(view["questions"]):
            q = view["questions"][i]
            choices = [f"{LETTERS[j]}: {o}" for j, o in enumerate(q["options"])]
            radios.append(gr.update(visible=True, label=f"{i + 1}. {q['question']}", choices=choices, value=None))
        else:
            radios.append(gr.update(visible=False))
    return quiz_id, *radios, f"Quiz ready — {len(view['questions'])} questions (solutions hidden).", ""


def do_submit(service, qid, *values):
    if not qid:
        gr.Info("Generate a quiz first.")
        return "Generate a quiz first."
    try:
        questions = service.get_quiz(qid)["questions"]
        answers = []
        for i, (val, q) in enumerate(zip(values, questions)):
            if val is None:
                answers.append(-1)  # unanswered -> wrong
            else:
                answers.append(
                    [f"{LETTERS[j]}: {o}" for j, o in enumerate(q["options"])].index(val)
                )
        result = service.grade_quiz(qid, answers)
    except Exception as exc:
        return f"*Couldn't grade right now — {_friendly(exc)}.*"
    lines = [f"### Score: {result['score']} / {result['total']}"]
    for i, fb in enumerate(result["feedback"]):
        mark = "✅" if fb["correct"] else "❌"
        lines.append(f"{mark} **Q{i + 1}:** {fb['explanation']}")
        srcs = ", ".join(f"*{s['document']}* slide {s['page']}" for s in fb["sources"])
        lines.append(f"{'   - sources: ' + srcs if srcs else ''}")
    return "\n\n".join(lines)


def do_reveal(service, qid):
    if not qid:
        gr.Info("Generate a quiz first.")
        return ""
    try:
        revealed = service.reveal_quiz(qid)
    except Exception as exc:
        return f"*Couldn't reveal the solutions right now — {_friendly(exc)}.*"
    lines = ["### Solutions"]
    for i, q in enumerate(revealed["questions"]):
        lines.append(
            f"**Q{i + 1}:** {q['question']}\n\n"
            f"✅ **{LETTERS[q['answer_index']]}:** {q['options'][q['answer_index']]}\n\n{q['explanation']}"
        )
    return "\n\n".join(lines)


# ---- Materials -------------------------------------------------------

def do_add(service, files):
    paths = _file_value(files)
    if not paths:
        gr.Info("Choose files first.")
        return "Choose files first.", gr.update(choices=scope_choices(service)), gr.update(choices=scope_choices(service))
    try:
        messages = service.add_documents(paths)
    except Exception as exc:  # e.g. embedding service hiccup during ingest
        messages = [f"✗ Could not index the selected file(s) right now — {_friendly(exc)}."]
    return "\n".join(messages), gr.update(choices=scope_choices(service)), gr.update(choices=scope_choices(service))


def do_remove(service, name: str):
    if not name:
        gr.Info("Pick a document to remove.")
        return "Pick a document to remove.", gr.update(choices=scope_choices(service)), gr.update(choices=scope_choices(service))
    try:
        msg = service.remove_document(name)
    except KeyError:
        msg = f"Not found: {name}"
    except Exception as exc:
        msg = f"Could not remove '{name}' right now — {_friendly(exc)}."
    return msg, gr.update(choices=scope_choices(service)), gr.update(choices=scope_choices(service))


# ---- Layout ----------------------------------------------------------

def build_app(service: CourseAssistant | None = None) -> gr.Blocks:
    service = service or CourseAssistant()

    with gr.Blocks(title="MBAX 6418 Course Assistant 🎓") as demo:
        with gr.Column(elem_id="hero"):
            gr.Markdown(_header_markdown(service))
        with gr.Tab("💬 Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    scope = gr.Dropdown(choices=scope_choices(service), value=ALL_SCOPE, label="📁 Material")
                    topic = gr.Textbox(placeholder="Optional — limits answers to material mentioning this word (e.g. 'grading')", label="🔍 Topic filter")
                    question = gr.Textbox(show_label=False, placeholder="Ask something about the course materials…", lines=2, elem_id="ask_question")
                    ask_btn = gr.Button("✨ Ask", variant="primary", elem_id="ask_btn")
                    gr.Markdown("**Try one of these:**", elem_classes=["chip-row"])
                    _sample_buttons(question)
                    answer_md = gr.Markdown()
                with gr.Column(scale=2):
                    gr.Markdown("#### 🖼️ Slide images from the retrieved sources")
                    gallery = gr.Gallery(columns=1, height=420, show_label=False)

        with gr.Tab("📝 Quiz"):
            gr.Markdown(
                "Practice like it's midterms. Questions come **straight from the deck** — "
                "stand-alone concepts, answers never hidden in the question, each question "
                "from a different slide. The key stays locked until you answer or reveal. 🔐"
            )
            with gr.Row():
                q_scope = gr.Dropdown(choices=scope_choices(service), value=ALL_SCOPE, label="📁 Quiz material")
                q_topic = gr.Textbox(placeholder="Optional topic filter", label="🔍 Topic filter")
                q_n = gr.Slider(2, MAX_QUIZ, value=5, step=1, label="🔢 Number of questions")
                gen_btn = gr.Button("🎲 Generate quiz", variant="primary", elem_id="gen_btn")
            qid_state = gr.State("")
            quiz_md = gr.Markdown()
            radios = [gr.Radio(visible=False, label=f"Q{i + 1}") for i in range(MAX_QUIZ)]
            with gr.Row():
                submit_btn = gr.Button("✅ Submit answers", variant="primary", elem_id="submit_btn")
                reveal_btn = gr.Button("🔓 Show solutions", elem_id="reveal_btn")
            feedback_md = gr.Markdown()

        with gr.Tab("📂 Materials"):
            gr.Markdown(
                "Add or remove course materials. The same file twice is a no-op "
                "(content-hash dedupe); removing a document also removes its "
                "vectors and extracted images so later answers can't rely on it."
            )
            files = gr.File(file_count="multiple", label="Files (PDF, PPTX, DOCX, MD, TXT, HTML)")
            add_btn = gr.Button("📥 Load files", variant="primary")
            status_md = gr.Markdown()
            with gr.Row():
                remove_scope = gr.Dropdown(choices=scope_choices(service), label="Remove a document")
                remove_btn = gr.Button("🗑️ Remove")

        ask_btn.click(functools.partial(do_ask, service), [question, scope, topic], [answer_md, gallery])
        gen_btn.click(functools.partial(do_generate, service), [q_scope, q_topic, q_n, qid_state], [qid_state, *radios, quiz_md, feedback_md])
        submit_btn.click(functools.partial(do_submit, service), [qid_state, *radios], feedback_md)
        reveal_btn.click(functools.partial(do_reveal, service), [qid_state], feedback_md)
        add_btn.click(functools.partial(do_add, service), [files], [status_md, scope, q_scope])
        remove_btn.click(functools.partial(do_remove, service), [remove_scope], [status_md, scope, q_scope])

        gr.Markdown(
            "Built for **MBAX 6418** · hybrid RAG: BM25 keyword × text embeddings × visual embeddings → "
            "fusion → rerank → grounded answers. Feel free to poke it — every answer is verifiable, "
            "every quiz question is from a real slide. 🚀",
            elem_id="app-footer",
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=os.environ.get("SHARE") == "1",
        theme=FUN_THEME,
        css=FUN_CSS,
    )
