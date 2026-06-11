import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agent.graph import graph
from src.agent.state import TutorState

THREAD_CONFIG = lambda sid: {"configurable": {"thread_id": sid}}

NODE_LABELS = {
    "load_state_node":          "Loading student profile",
    "select_topic_node":        "Selecting topic and difficulty",
    "generate_problem_node":    "Generating problem (Ollama)",
    "evaluate_answer_node":     "Checking answer (SymPy symbolic solver)",
    "retrieve_explanation_node": "Searching knowledge base (ChromaDB RAG)",
    "generate_feedback_node":   "Writing feedback (Ollama)",
    "update_state_node":        "Saving progress and updating mastery",
    "adapt_next_node":          "Adjusting difficulty",
}


def run_until_pause(graph_input, config, status_label):
    """Stream the graph from its current point to the next interrupt (or END),
    showing a live per-node status. Returns the resulting state values."""
    with st.status(status_label, expanded=True) as status:
        for chunk in graph.stream(graph_input, config=config, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            label = NODE_LABELS.get(node_name, node_name)
            status.update(label=f"⚙ {label}…")
            st.write(f"✓ {label}")
        status.update(label="Done", state="complete", expanded=False)
    return graph.get_state(config).values


st.set_page_config(page_title="Math Tutor", page_icon="📐", layout="centered")
st.title("📐 Math Tutor")

# ── Student identity ──────────────────────────────────────────────────────────
if "student_id" not in st.session_state:
    st.session_state.student_id = None

if st.session_state.student_id is None:
    st.subheader("Welcome! What's your name?")
    name = st.text_input("Your name", key="name_input")
    if st.button("Start learning") and name.strip():
        st.session_state.student_id = name.strip().lower().replace(" ", "_")
        st.rerun()
    st.stop()

student_id = st.session_state.student_id
config = THREAD_CONFIG(student_id)

# ── Bootstrap: generate the first problem ─────────────────────────────────────
if "graph_started" not in st.session_state:
    initial: TutorState = {
        "student_id": student_id,
        "current_topic": "",
        "current_subtopic": "",
        "current_difficulty": 2,
        "current_problem": "",
        "sympy_answer": "",
        "student_answer": "",
        "evaluation": {},
        "retrieved_chunks": [],
        "feedback": "",
        "mastery": {},
        "session_history": [],
    }
    run_until_pause(initial, config, "Starting session…")
    st.session_state.graph_started = True
    st.session_state.phase = "answering"   # "answering" | "reviewing"
    st.session_state.last_feedback = None
    st.session_state.attempt_count = 0

# ── Read current graph state ──────────────────────────────────────────────────
state_values: TutorState = graph.get_state(config).values

# ── Sidebar: mastery dashboard ────────────────────────────────────────────────
with st.sidebar:
    st.subheader(f"👋 {student_id.replace('_', ' ').title()}")
    st.markdown(f"**Attempts this session:** {st.session_state.attempt_count}")
    st.divider()
    st.subheader("Mastery")
    mastery = state_values.get("mastery", {})
    topic_labels = {"fractions": "Fractions", "ratios": "Ratios", "algebra": "Algebra", "geometry": "Geometry"}
    for topic, label in topic_labels.items():
        score = mastery.get(topic, 0.0)
        st.write(f"**{label}**")
        st.progress(score, text=f"{score:.0%}")

# ── Current problem ───────────────────────────────────────────────────────────
topic = state_values.get("current_topic", "")
subtopic = state_values.get("current_subtopic", "")
difficulty = state_values.get("current_difficulty", 1)
problem_text = state_values.get("current_problem", "")

difficulty_label = {1: "Intro", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Challenge"}.get(difficulty, "")
if topic:
    st.caption(f"{topic.capitalize()} · {subtopic.replace('_', ' ')}  ·  {difficulty_label}")

st.markdown(f"### {problem_text}" if problem_text else "### Loading problem…")

# ══════════════════════════════════════════════════════════════════════════════
# ANSWERING PHASE — show the answer form
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == "answering":
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_input("Your answer", placeholder="e.g.  5  or  3/4  or  x=2")
        submitted = st.form_submit_button("Submit")

    if submitted and answer.strip():
        student_answer_text = answer.strip()
        graph.update_state(config, {"student_answer": student_answer_text})

        # Resume from the interrupt: evaluate → retrieve → feedback → update → adapt → END
        updated = run_until_pause(None, config, "Checking your answer…")
        evaluation = updated.get("evaluation", {})

        st.session_state.last_feedback = {
            "is_correct": evaluation.get("is_correct", False),
            "parse_error": evaluation.get("parse_error", False),
            "error_category": evaluation.get("error_category"),
            "feedback": updated.get("feedback", ""),
            "problem": problem_text,
            "student_answer": student_answer_text,
        }
        st.session_state.attempt_count += 1
        st.session_state.phase = "reviewing"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# REVIEWING PHASE — show full feedback + "Next problem" button
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == "reviewing":
    fb = st.session_state.last_feedback or {}

    if fb.get("is_correct"):
        st.success(f"✓ Correct!  Your answer: **{fb.get('student_answer', '')}**")
    elif fb.get("parse_error"):
        st.warning(f"⚠ Couldn't read your answer as a number.  You entered: **{fb.get('student_answer', '')}**")
    else:
        cat = fb.get("error_category")
        cat_note = f"  ·  error type: *{cat}*" if cat else ""
        st.error(f"✗ Not quite.  Your answer: **{fb.get('student_answer', '')}**{cat_note}")

    # Full LLM explanation — preserve section structure regardless of LLM newline habits
    feedback_text = fb.get("feedback", "") or "_No explanation was generated._"
    for section in ("Result:", "Explanation:", "What went wrong:"):
        feedback_text = feedback_text.replace(section, f"\n\n**{section}**")

    with st.container(border=True):
        st.markdown(feedback_text.strip())

    if st.button("Next problem →", type="primary"):
        # Start a fresh turn: load_state → select_topic → generate_problem → pause.
        # generate_problem_node clears the previous answer/evaluation/feedback.
        run_until_pause({"student_answer": ""}, config, "Generating next problem…")
        st.session_state.last_feedback = None
        st.session_state.phase = "answering"
        st.rerun()
