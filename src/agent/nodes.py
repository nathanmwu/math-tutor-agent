from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from langchain_ollama import ChatOllama
from sympy import Eq, simplify, solve, sympify

from src.agent.state import TutorState
from src.agent.prompts import (
    CATEGORIZE_ERROR_PROMPT,
    FEEDBACK_CORRECT_PROMPT,
    FEEDBACK_INCORRECT_PROMPT,
    GENERATE_PROBLEM_PROMPT,
)
from src.agent.solution_steps import generate_solution_steps
from src.knowledge.retriever import retrieve_content
from src.state import store

CURRICULUM_ORDER = ["fractions", "ratios", "algebra", "geometry"]

SUBTOPICS: dict[str, list[str]] = {
    "fractions": ["equivalent_fractions", "addition_subtraction", "multiplication_division"],
    "ratios": ["unit_rates", "proportions", "percentages"],
    "algebra": ["linear_equations", "inequalities", "substitution"],
    "geometry": ["area_perimeter", "pythagorean_theorem", "coordinate_plane"],
}

STUDENT_STATE_DIR = Path(os.getenv("STUDENT_STATE_DIR", "data/students"))
CHROMA_DIR = Path(os.getenv("CHROMADB_PATH", "data/chromadb"))

VALID_ERROR_CATEGORIES = {
    "sign_error",
    "wrong_operation",
    "arithmetic_mistake",
    "conceptual_error",
    "other",
}


def _parse_problem_json(raw: str) -> dict:
    """Parse the problem-generation JSON, repairing LaTeX backslash escaping.

    LLMs frequently emit single-backslash LaTeX inside JSON strings. Two failure modes:
    - "\\cdot", "\\sqrt": invalid JSON escapes → json.loads raises
    - "\\frac", "\\times", "\\neq", "\\binom": collide with valid JSON escapes
      (\\f, \\t, \\n, \\b) and silently decode to control characters
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Double any lone backslash not followed by a valid JSON escape char
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    data = json.loads(repaired)

    # Restore LaTeX commands that decoded to control characters
    control_repairs = {"\f": "\\f", "\t": "\\t", "\b": "\\b", "\r": "\\r", "\n": "\\n"}
    for key, value in data.items():
        if isinstance(value, str):
            for ctrl, fixed in control_repairs.items():
                value = value.replace(ctrl, fixed)
            data[key] = value
    return data


def symbolic_check(student_input: str, sympy_answer: str) -> bool | None:
    try:
        cleaned = student_input.strip()

        # Strip "x = " prefix so students can write "x = 3" for algebra answers
        cleaned = re.sub(r"^[a-zA-Z_]\w*\s*=\s*", "", cleaned).strip()

        mixed = re.search(r"(\d+)\s+(\d+)/(\d+)", cleaned)
        if mixed:
            whole = int(mixed.group(1))
            num = int(mixed.group(2))
            den = int(mixed.group(3))
            improper_num = whole * den + num
            cleaned = re.sub(r"\d+\s+\d+/\d+", f"{improper_num}/{den}", cleaned)

        cleaned = re.sub(
            r"\s*(sq\.?\s*)?(cm|m|km|mm|in|ft|yd|meters?|feet|inches?|miles?|pounds?|lbs?|kg|grams?)\b.*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        student_expr = sympify(cleaned, rational=True)
        correct_expr = sympify(sympy_answer, rational=True)
        # sympify('abc') → Symbol('abc') without raising; treat as unparseable
        if student_expr.free_symbols and not correct_expr.free_symbols:
            return None
        return simplify(student_expr - correct_expr) == 0
    except Exception:
        return None


def load_state_node(state: TutorState) -> dict:
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    mastery = {topic: tm.mastery_score for topic, tm in student.topic_mastery.items()}
    return {"mastery": mastery}


def select_topic_node(state: TutorState) -> dict:
    mastery = state.get("mastery", {})

    if random.random() < 0.7:
        topic = min(
            CURRICULUM_ORDER,
            key=lambda t: mastery.get(t, 0.0),
        )
    else:
        topic = random.choice(CURRICULUM_ORDER)

    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    tm = student.topic_mastery.get(topic)

    subtopics = SUBTOPICS[topic]
    subtopic = None
    if tm is not None:
        attempted_counts: dict[str, int] = {}
        for record in student.attempt_history:
            if record.topic == topic:
                attempted_counts[record.subtopic] = attempted_counts.get(record.subtopic, 0) + 1
        for st in subtopics:
            if attempted_counts.get(st, 0) < 3:
                subtopic = st
                break

    if subtopic is None:
        subtopic = random.choice(subtopics)

    if tm is not None:
        difficulty = tm.current_difficulty
    else:
        difficulty = 2

    return {
        "current_topic": topic,
        "current_subtopic": subtopic,
        "current_difficulty": difficulty,
    }


def generate_problem_node(state: TutorState) -> dict:
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), temperature=0.7)

    history = state.get("session_history", [])
    recent = history[-3:] if len(history) >= 3 else history
    recent_problems = "; ".join(item.get("topic", "") for item in recent) if recent else "none"

    prompt = GENERATE_PROBLEM_PROMPT.format(
        topic=state["current_topic"],
        subtopic=state["current_subtopic"],
        difficulty=state["current_difficulty"],
        recent_problems=recent_problems,
    )

    problem_text = None
    sympy_answer = None
    sympy_expression = None

    for _ in range(3):
        response = llm.invoke(prompt)
        raw = response.content.strip()

        try:
            data = _parse_problem_json(raw)
            pt = data.get("problem_text", "").strip()
            expr_str = data.get("sympy_expression", "").strip()
            if pt and expr_str:
                # Evaluate the expression ourselves — never trust the LLM's arithmetic
                computed = sympify(
                    expr_str,
                    locals={"Rational": __import__("sympy").Rational, "Eq": Eq},
                    rational=True,
                )
                # Eq(lhs, rhs) — solve for the single free symbol
                if isinstance(computed, Eq):
                    free = computed.free_symbols
                    if len(free) != 1:
                        continue
                    solutions = solve(computed, list(free)[0])
                    if len(solutions) != 1:
                        continue
                    candidate = solutions[0]
                    # Verify: substitute back — catches LLM typos in the equation
                    check = computed.subs(list(free)[0], candidate)
                    if not check:
                        continue
                    computed = candidate
                # solve() returning a list (legacy path)
                elif isinstance(computed, list):
                    if len(computed) != 1:
                        continue
                    computed = computed[0]
                # must be a concrete number — rejects leftover symbols AND
                # booleans (an Eq of two constants evaluates to True/False)
                if not getattr(computed, "is_number", False):
                    continue
                problem_text = pt
                sympy_answer = str(computed)
                sympy_expression = expr_str
                break
        except Exception:
            continue

    if problem_text is None or sympy_answer is None:
        problem_text = "$2 + 2 =$"
        sympy_answer = "4"
        sympy_expression = "2 + 2"

    return {
        "current_problem": problem_text,
        "sympy_answer": sympy_answer,
        "sympy_expression": sympy_expression,
        "solution_steps": generate_solution_steps(sympy_expression),
        "student_answer": "",
        "evaluation": {},
        "retrieved_chunks": [],
        "feedback": "",
    }


def evaluate_answer_node(state: TutorState) -> dict:
    result = symbolic_check(state["student_answer"], state["sympy_answer"])

    if result is None:
        return {
            "evaluation": {
                "is_correct": False,
                "error_category": None,
                "parse_error": True,
            }
        }

    if result:
        return {
            "evaluation": {
                "is_correct": True,
                "error_category": None,
                "parse_error": False,
            }
        }

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), temperature=0.7)
    prompt = CATEGORIZE_ERROR_PROMPT.format(
        problem=state["current_problem"],
        correct_answer=state["sympy_answer"],
        student_answer=state["student_answer"],
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category not in VALID_ERROR_CATEGORIES:
        category = "other"

    return {
        "evaluation": {
            "is_correct": False,
            "error_category": category,
            "parse_error": False,
        }
    }


def retrieve_explanation_node(state: TutorState) -> dict:
    evaluation = state.get("evaluation", {})
    error_category = evaluation.get("error_category")

    chunks = retrieve_content(
        topic=state["current_topic"],
        subtopic=state["current_subtopic"],
        difficulty=state["current_difficulty"],
        chroma_dir=CHROMA_DIR,
        error_category=error_category,
        query_text=state.get("current_problem"),
    )
    return {"retrieved_chunks": chunks}


def generate_feedback_node(state: TutorState) -> dict:
    evaluation = state.get("evaluation", {})
    chunks = state.get("retrieved_chunks", [])
    retrieved_content = "\n---\n".join(chunks) if chunks else ""
    steps = state.get("solution_steps", [])

    template = (
        FEEDBACK_CORRECT_PROMPT
        if evaluation.get("is_correct", False)
        else FEEDBACK_INCORRECT_PROMPT
    )
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), temperature=0.7)
    prompt = template.format(
        problem=state["current_problem"],
        student_answer=state["student_answer"],
        correct_answer=state["sympy_answer"],
        solution_steps="\n".join(steps) if steps else "(not available)",
        retrieved_content=retrieved_content,
    )
    response = llm.invoke(prompt)
    return {"feedback": response.content.strip()}


def update_state_node(state: TutorState) -> dict:
    evaluation = state.get("evaluation", {})
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    student = store.record_attempt(
        state=student,
        topic=state["current_topic"],
        subtopic=state["current_subtopic"],
        difficulty=state["current_difficulty"],
        problem_text=state["current_problem"],
        student_answer=state["student_answer"],
        is_correct=evaluation.get("is_correct", False),
        error_category=evaluation.get("error_category"),
        parse_error=evaluation.get("parse_error", False),
    )
    store.save_student(student, STUDENT_STATE_DIR)

    mastery = {topic: tm.mastery_score for topic, tm in student.topic_mastery.items()}

    history = list(state.get("session_history", []))
    history.append(
        {
            "topic": state["current_topic"],
            "is_correct": evaluation.get("is_correct", False),
            "difficulty": state["current_difficulty"],
        }
    )

    return {"mastery": mastery, "session_history": history}


def adapt_next_node(state: TutorState) -> dict:
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    tm = student.topic_mastery.get(state["current_topic"])

    if tm is not None:
        new_difficulty = tm.current_difficulty
    else:
        new_difficulty = state.get("current_difficulty", 2)

    return {"current_difficulty": new_difficulty}
