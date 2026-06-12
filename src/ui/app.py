import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import html as html_lib
import re

from dotenv import load_dotenv

load_dotenv()

from nicegui import context, run, ui

from src.agent.graph import graph
from src.agent.state import TutorState

NODE_LABELS = {
    "load_state_node": "Loading student profile",
    "select_topic_node": "Selecting topic and difficulty",
    "generate_problem_node": "Generating problem (Ollama)",
    "evaluate_answer_node": "Checking answer (SymPy symbolic solver)",
    "retrieve_explanation_node": "Searching knowledge base (ChromaDB RAG)",
    "generate_feedback_node": "Writing explanation (Ollama)",
    "update_state_node": "Saving progress and updating mastery",
    "adapt_next_node": "Adjusting difficulty",
}

TOPIC_LABELS = {
    "fractions": "Fractions",
    "ratios": "Ratios",
    "algebra": "Algebra",
    "geometry": "Geometry",
}

DIFFICULTY_LABELS = {1: "Intro", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Challenge"}

# KaTeX typesetting, scoped strictly to .math-content containers (ui.html
# elements whose innerHTML is opaque to Vue). NEVER run KaTeX on document.body:
# its text-node scan merges/normalizes adjacent text nodes, which destroys the
# empty-text-node anchors Vue 3 uses for fragments — the next Vue patch then
# crashes with "Cannot read property 'insertBefore' of null" and the UI dies.
KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"></script>
<script>
  window.typesetMath = function () {
    if (!window.renderMathInElement) return;
    document.querySelectorAll('.math-content').forEach(function (el) {
      renderMathInElement(el, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
        ],
        throwOnError: false,
      });
    });
  };
</script>
"""

_SENTINEL = object()


def initial_tutor_state(student_id: str) -> TutorState:
    return {
        "student_id": student_id,
        "current_topic": "",
        "current_subtopic": "",
        "current_difficulty": 2,
        "current_problem": "",
        "sympy_expression": "",
        "sympy_answer": "",
        "solution_steps": [],
        "student_answer": "",
        "evaluation": {},
        "retrieved_chunks": [],
        "feedback": "",
        "mastery": {},
        "session_history": [],
    }


def feedback_to_html(text: str) -> str:
    """Plain LLM text → HTML. Escapes everything, preserves line breaks.
    LaTeX ($...$) is left intact for KaTeX."""
    escaped = html_lib.escape(text)
    # LLM sometimes emits markdown bold — convert it
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br>")


def steps_to_html(steps: list[str]) -> str:
    """SymPy-verified derivation steps → numbered HTML list for KaTeX."""
    items = "".join(
        f'<li class="py-1">{html_lib.escape(step)}</li>' for step in steps
    )
    return f'<ol class="list-decimal list-inside">{items}</ol>'


@ui.page("/")
def main_page():
    ui.add_head_html(KATEX_HEAD)
    ui.colors(primary="#4f46e5")
    ui.query("body").classes("bg-slate-100")

    session = {
        "student_id": None,
        "config": None,
        "phase": "name",  # name | working | answering | reviewing
        "attempts": 0,
        "last": None,       # snapshot shown in the reviewing phase
        "status_col": None,  # live operation feed target while working
    }

    def graph_values() -> dict:
        return graph.get_state(session["config"]).values

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with ui.left_drawer(value=True).classes("bg-white border-r border-slate-200"):
        drawer = ui.column().classes("w-full gap-1 p-2")

    def refresh_drawer():
        drawer.clear()
        with drawer:
            if not session["student_id"]:
                ui.label("Math Tutor").classes("text-lg font-semibold text-slate-700")
                ui.label("Adaptive K-12 practice").classes("text-xs text-slate-400")
                return
            name = session["student_id"].replace("_", " ").title()
            ui.label(f"👋 {name}").classes("text-lg font-semibold text-slate-700")
            ui.label(f"Attempts this session: {session['attempts']}").classes(
                "text-xs text-slate-500"
            )
            ui.separator().classes("my-2")
            ui.label("Mastery").classes("text-sm font-semibold text-slate-600")
            mastery = graph_values().get("mastery", {}) if session["config"] else {}
            for topic, label in TOPIC_LABELS.items():
                score = mastery.get(topic, 0.0)
                with ui.row().classes("w-full items-center justify-between mt-2"):
                    ui.label(label).classes("text-sm text-slate-700")
                    ui.label(f"{score:.0%}").classes("text-xs text-slate-400")
                ui.linear_progress(value=round(score, 2), show_value=False).classes(
                    "w-full"
                ).props("rounded size=8px")

    # ── Main area ─────────────────────────────────────────────────────────────
    @ui.refreshable
    def main_area():
        if session["phase"] == "name":
            with ui.card().classes(
                "w-full max-w-md mx-auto mt-24 p-8 rounded-xl shadow-sm border border-slate-200"
            ):
                ui.label("📐 Math Tutor").classes("text-2xl font-bold text-slate-800")
                ui.label("What's your name?").classes("text-sm text-slate-500 mb-2")
                name_input = ui.input(placeholder="Your name").classes("w-full").props(
                    "outlined dense"
                )
                ui.button(
                    "Start learning",
                    on_click=lambda: start_session(name_input.value),
                ).classes("w-full mt-4")
                name_input.on("keydown.enter", lambda: start_session(name_input.value))
            return

        values = graph_values()
        problem = (
            session["last"]["problem"]
            if session["phase"] == "reviewing" and session["last"]
            else values.get("current_problem", "")
        )
        topic = values.get("current_topic", "")
        subtopic = values.get("current_subtopic", "")
        difficulty = values.get("current_difficulty", 2)

        with ui.column().classes("w-full max-w-2xl mx-auto mt-10 px-4 gap-4"):
            # Problem card
            with ui.card().classes(
                "w-full p-8 rounded-xl shadow-sm border border-slate-200"
            ):
                if topic:
                    with ui.row().classes("gap-2 mb-3"):
                        ui.badge(TOPIC_LABELS.get(topic, topic)).props(
                            "color=indigo-1 text-color=indigo-9"
                        ).classes("px-2 py-1")
                        ui.badge(f"{subtopic.replace('_', ' ')}").props(
                            "color=slate-2 text-color=slate-8"
                        ).classes("px-2 py-1")
                        ui.badge(DIFFICULTY_LABELS.get(difficulty, "")).props(
                            "color=amber-1 text-color=amber-9"
                        ).classes("px-2 py-1")
                ui.html(
                    f'<div class="math-content text-2xl text-slate-800">{html_lib.escape(problem)}</div>'
                )

            if session["phase"] == "working":
                with ui.card().classes(
                    "w-full p-4 rounded-xl shadow-sm border border-slate-200"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.spinner(size="sm", color="primary")
                        ui.label("Working…").classes("text-sm text-slate-500")
                    session["status_col"] = ui.column().classes("gap-0 mt-2")

            elif session["phase"] == "answering":
                with ui.row().classes("w-full gap-2"):
                    answer_input = (
                        ui.input(placeholder="Your answer — e.g. 5, 3/4, or 1.5")
                        .classes("flex-grow")
                        .props("outlined dense")
                    )
                    ui.button(
                        "Submit", on_click=lambda: submit_answer(answer_input.value)
                    )
                    answer_input.on(
                        "keydown.enter", lambda: submit_answer(answer_input.value)
                    )

            elif session["phase"] == "reviewing":
                fb = session["last"] or {}
                if fb.get("is_correct"):
                    banner_classes = "bg-green-50 border-green-200 text-green-800"
                    banner_text = f"✓ Correct — your answer: {fb.get('student_answer', '')}"
                elif fb.get("parse_error"):
                    banner_classes = "bg-amber-50 border-amber-200 text-amber-800"
                    banner_text = (
                        f"⚠ Couldn't read \"{fb.get('student_answer', '')}\" as a number"
                    )
                else:
                    banner_classes = "bg-red-50 border-red-200 text-red-800"
                    banner_text = f"✗ Not quite — your answer: {fb.get('student_answer', '')}"

                with ui.card().classes(
                    f"w-full p-4 rounded-xl border {banner_classes} shadow-none"
                ):
                    ui.label(banner_text).classes("font-medium")

                with ui.card().classes(
                    "w-full p-6 rounded-xl shadow-sm border border-slate-200"
                ):
                    ui.html(
                        f'<div class="math-content text-base leading-relaxed text-slate-700">'
                        f'{feedback_to_html(fb.get("feedback", "") or "No explanation was generated.")}'
                        f"</div>"
                    )
                    steps = fb.get("steps") or []
                    if steps:
                        ui.separator().classes("my-3")
                        ui.label("Solution").classes(
                            "text-sm font-semibold text-slate-600 mb-1"
                        )
                        ui.html(
                            f'<div class="math-content text-lg text-slate-800">'
                            f"{steps_to_html(steps)}</div>"
                        )

                ui.button("Next problem →", on_click=next_problem).classes("mt-1")

    # Captured at page build: the ambient client context is lost after awaits
    # in event handlers, so ui.run_javascript would silently target nothing.
    page_client = context.client

    def typeset() -> None:
        """Typeset $...$ inside .math-content containers. The 50 ms delay lets
        Vue finish applying the preceding DOM updates first."""
        page_client.run_javascript("setTimeout(window.typesetMath, 50)")

    # ── Graph execution ───────────────────────────────────────────────────────
    async def stream_graph(graph_input) -> None:
        """Consume graph.stream chunk by chunk on a worker thread, appending
        each completed node to the live status feed on the event loop."""
        iterator = graph.stream(
            graph_input, config=session["config"], stream_mode="updates"
        )
        while True:
            chunk = await run.io_bound(lambda: next(iterator, _SENTINEL))
            if chunk is _SENTINEL:
                break
            node = next(iter(chunk))
            if node.startswith("__"):
                continue
            if session["status_col"] is not None:
                with session["status_col"]:
                    ui.label(f"✓ {NODE_LABELS.get(node, node)}").classes(
                        "text-xs text-slate-400"
                    )

    async def start_session(name: str) -> None:
        if not name or not name.strip():
            ui.notify("Please enter your name", type="warning")
            return
        session["student_id"] = name.strip().lower().replace(" ", "_")
        session["config"] = {"configurable": {"thread_id": session["student_id"]}}
        session["phase"] = "working"
        refresh_drawer()
        main_area.refresh()
        await stream_graph(initial_tutor_state(session["student_id"]))
        session["phase"] = "answering"
        refresh_drawer()
        main_area.refresh()
        typeset()

    async def submit_answer(answer: str) -> None:
        if not answer or not answer.strip():
            return
        student_answer = answer.strip()
        graph.update_state(session["config"], {"student_answer": student_answer})
        session["phase"] = "working"
        main_area.refresh()

        # Resume from the interrupt: evaluate → retrieve → feedback → update → adapt → END
        await stream_graph(None)

        values = graph_values()
        evaluation = values.get("evaluation", {})
        session["last"] = {
            "is_correct": evaluation.get("is_correct", False),
            "parse_error": evaluation.get("parse_error", False),
            "feedback": values.get("feedback", ""),
            "problem": values.get("current_problem", ""),
            "steps": values.get("solution_steps", []),
            "student_answer": student_answer,
        }
        session["attempts"] += 1
        session["phase"] = "reviewing"
        refresh_drawer()
        main_area.refresh()
        typeset()

    async def next_problem() -> None:
        session["phase"] = "working"
        session["last"] = None
        main_area.refresh()
        # Fresh turn: load_state → select_topic → generate_problem → pause
        await stream_graph({"student_answer": ""})
        session["phase"] = "answering"
        refresh_drawer()
        main_area.refresh()
        typeset()

    refresh_drawer()
    main_area()






if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Math Tutor", port=8501, reload=False, show=False, reconnect_timeout=30)
