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

# ── Bootstrap: run graph until first interrupt ────────────────────────────────
if "graph_started" not in st.session_state:
    with st.spinner("Loading your session…"):
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
        graph.invoke(initial, config=config)
    st.session_state.graph_started = True
    st.session_state.awaiting_answer = True
    st.session_state.last_feedback = None
    st.session_state.attempt_count = 0

# ── Read current graph state ──────────────────────────────────────────────────
graph_state = graph.get_state(config)
state_values: TutorState = graph_state.values

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

# ── Show last feedback if available ──────────────────────────────────────────
if st.session_state.last_feedback:
    fb = st.session_state.last_feedback
    if fb.get("is_correct"):
        st.success(f"✓ Correct!  {fb['feedback']}")
    elif fb.get("parse_error"):
        st.warning(f"⚠️ Couldn't parse your answer. {fb['feedback']}")
    else:
        st.error(f"✗ Not quite.  {fb['feedback']}")
    st.session_state.last_feedback = None

# ── Current problem ───────────────────────────────────────────────────────────
topic = state_values.get("current_topic", "")
difficulty = state_values.get("current_difficulty", 1)
problem_text = state_values.get("current_problem", "")

difficulty_label = {1: "Intro", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Challenge"}.get(difficulty, "")
if topic:
    st.caption(f"{topic.capitalize()}  ·  {difficulty_label}")

st.markdown(f"### {problem_text}" if problem_text else "### Loading problem…")

# ── Answer form ───────────────────────────────────────────────────────────────
with st.form("answer_form", clear_on_submit=True):
    answer = st.text_input("Your answer", placeholder="e.g.  5  or  3/4  or  x=2")
    submitted = st.form_submit_button("Submit")

if submitted and answer.strip():
    with st.spinner("Checking…"):
        graph.update_state(config, {"student_answer": answer.strip()})
        graph.invoke(None, config=config)

    updated = graph.get_state(config).values
    evaluation = updated.get("evaluation", {})
    st.session_state.last_feedback = {
        "is_correct": evaluation.get("is_correct", False),
        "parse_error": evaluation.get("parse_error", False),
        "feedback": updated.get("feedback", ""),
    }
    st.session_state.attempt_count += 1
    st.rerun()
