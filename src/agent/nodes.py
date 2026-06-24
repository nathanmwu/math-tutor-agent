from __future__ import annotations

import json
import os
import random
import re
from math import gcd
from pathlib import Path

from langchain_ollama import ChatOllama
from sympy import Eq, simplify, solve, sympify

from src.agent.state import TutorState
from src.agent.prompts import (
    CATEGORIZE_ERROR_PROMPT,
    FEEDBACK_CORRECT_PROMPT,
    FEEDBACK_INCORRECT_PROMPT,
    SUBTOPIC_VARIANTS,
    build_generate_problem_prompt,
)
from src.agent.solution_steps import (
    generate_solution_steps,
    linear_eval_solution_steps,
    slope_solution_steps,
)
from src.knowledge.retriever import retrieve_content
from src.state import store
from src.state.models import SubtopicMastery, subtopic_key

# Fraction of turns that ignore weakness-priority and pick a subtopic uniformly,
# preserving coverage/variety so the agent never tunnels into one weak subtopic.
EXPLORATION_FRACTION = 0.25
# Probability of preferring a never-attempted subtopic during cold start, so
# breadth is guaranteed before weakness-weighting takes over.
COLD_START_FRACTION = 0.6

CURRICULUM_ORDER = ["fractions_ratios", "algebra"]

SUBTOPICS: dict[str, list[str]] = {
    "fractions_ratios": [
        "equivalent_fractions",
        "addition_subtraction",
        "multiplication_division",
        "proportions",
        "percentages",
    ],
    "algebra": ["linear_equations", "evaluating_expressions", "linear_relationships"],
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

    # Double any LONE backslash not followed by a valid JSON escape char. The
    # negative lookbehind (?<!\\) avoids touching the second backslash of an
    # already-correct "\\" pair — e.g. the model correctly emits "\\%" for LaTeX
    # \%, which must stay "\\%" (valid) and not become "\\\%" (invalid escape).
    repaired = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", raw)
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

        # Accept a trailing percent sign for "what percent" answers ("50%" == "50")
        cleaned = re.sub(r"\s*%\s*$", "", cleaned).strip()

        student_expr = sympify(cleaned, rational=True)
        correct_expr = sympify(sympy_answer, rational=True)
        # sympify('abc') → Symbol('abc') without raising; treat as unparseable
        if student_expr.free_symbols and not correct_expr.free_symbols:
            return None
        return simplify(student_expr - correct_expr) == 0
    except Exception:
        return None


def _subtopic_mastery_summary(student) -> dict[str, dict]:
    """Compact per-subtopic mastery view passed through TutorState to the UI."""
    return {
        key: {
            "topic": sm.topic,
            "subtopic": sm.subtopic,
            "mastery_score": sm.mastery_score,
        }
        for key, sm in student.subtopic_mastery.items()
    }


def load_state_node(state: TutorState) -> dict:
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    mastery = {topic: tm.mastery_score for topic, tm in student.topic_mastery.items()}
    return {
        "mastery": mastery,
        "subtopic_mastery": _subtopic_mastery_summary(student),
    }


def _subtopic_priority(sm: SubtopicMastery) -> float:
    """Selection weight: higher = weaker = practice more.

    Combines low mastery with a recent-error boost. The floor keeps every
    subtopic reachable (a zero weight would make it unselectable).
    """
    weakness = 1.0 - sm.mastery_score
    error_rate = min(sum(sm.error_pattern_counts.values()) / max(sm.attempts, 1), 1.0)
    return max(0.05, weakness + 0.5 * error_rate)


def select_topic_node(state: TutorState) -> dict:
    """Pick a (topic, subtopic) pair, favoring the student's weak subtopics.

    Weighted-random over a transparent per-subtopic priority score, with a
    cold-start path that guarantees breadth and an exploration fraction that
    preserves variety so the agent never tunnels into one subtopic.
    """
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    candidates = [(t, st) for t in CURRICULUM_ORDER for st in SUBTOPICS[t]]

    never = [
        (t, st) for (t, st) in candidates if subtopic_key(t, st) not in student.subtopic_mastery
    ]

    if never and random.random() < COLD_START_FRACTION:
        topic, subtopic = random.choice(never)
    elif random.random() < EXPLORATION_FRACTION:
        topic, subtopic = random.choice(candidates)
    else:
        weights = [
            _subtopic_priority(student.subtopic_mastery[subtopic_key(t, st)])
            if subtopic_key(t, st) in student.subtopic_mastery
            else 1.0
            for (t, st) in candidates
        ]
        topic, subtopic = random.choices(candidates, weights=weights, k=1)[0]

    sm = student.subtopic_mastery.get(subtopic_key(topic, subtopic))
    difficulty = sm.current_difficulty if sm is not None else 2

    return {
        "current_topic": topic,
        "current_subtopic": subtopic,
        "current_difficulty": difficulty,
    }


def _coef(m: int) -> str:
    """Coefficient as it should appear before x: '', '-', or the number."""
    if m == 1:
        return ""
    if m == -1:
        return "-"
    return str(m)


def _build_linear_relationship_problem(
    difficulty: int, seen_problems: set[str], avoid_signatures: set[str] | None = None
) -> dict | None:
    """Deterministically build a ``linear_relationships`` problem.

    Both ``problem_text`` and the answer derive from one set of integers, so the
    prose and the computed answer cannot drift — the failure mode that produced
    wrong slope answers (e.g. ``3/7`` for points $(-2,4),(5,-1)$) when the LLM
    authored the prose and the arithmetic expression independently. The matching
    derivation is built by the verified slope / linear-eval step builders.
    """
    span = 4 + 2 * difficulty  # coordinate magnitude grows with difficulty

    for _ in range(50):
        if random.random() < 0.5:
            # Form A: slope through two points
            a, c = random.randint(-span, span), random.randint(-span, span)
            if a == c:  # vertical line — slope undefined
                continue
            if c < a:  # keep x2 > x1 so the denominator stays positive
                a, c = c, a
            b, d = random.randint(-span, span), random.randint(-span, span)
            problem_text = rf"Find the slope of the line through $({a}, {b})$ and $({c}, {d})$"
            sympy_expression = f"Rational({d} - {b}, {c} - {a})"
            steps = slope_solution_steps((a, b), (c, d))
        else:
            # Form B: evaluate y = m x + k at x = v
            m = random.choice([n for n in range(-5, 6) if n != 0])
            k = random.choice([n for n in range(-9, 10) if n != 0])
            v = random.randint(-span, span)
            sign = "+" if k > 0 else "-"
            problem_text = (
                rf"If $y = {_coef(m)}x {sign} {abs(k)}$, find $y$ when $x = {v}$"
            )
            sympy_expression = f"{m}*{v} + {k}"
            steps = linear_eval_solution_steps(m, k, v)

        if problem_text in seen_problems:
            continue
        if avoid_signatures and store.problem_signature(problem_text) in avoid_signatures:
            continue
        return {
            "current_problem": problem_text,
            "sympy_answer": str(sympify(sympy_expression, rational=True)),
            "sympy_expression": sympy_expression,
            "solution_steps": steps,
            "student_answer": "",
            "evaluation": {},
            "retrieved_chunks": [],
            "feedback": "",
        }
    return None


def generate_problem_node(state: TutorState) -> dict:
    history = state.get("session_history", [])
    seen_problems = {item.get("problem_text", "") for item in history}

    # Persisted cross-session avoid-list of problem SHAPES for this subtopic.
    student = store.load_student(state["student_id"], STUDENT_STATE_DIR)
    key = subtopic_key(state["current_topic"], state["current_subtopic"])
    persisted_signatures = student.recent_signatures.get(key, [])

    # linear_relationships is generated deterministically (no LLM): the prose and
    # the answer must come from one source of truth, never drift apart.
    if state["current_subtopic"] == "linear_relationships":
        built = _build_linear_relationship_problem(
            state["current_difficulty"], seen_problems, set(persisted_signatures)
        )
        if built is not None:
            return built

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), temperature=0.7)

    recent = history[-5:] if len(history) >= 5 else history
    recent_problems = (
        "; ".join(item.get("problem_text", item.get("topic", "")) for item in recent)
        if recent else "none"
    )

    # Rotate problem shape and pass the structural avoid-list to the generator.
    variants = SUBTOPIC_VARIANTS.get(state["current_subtopic"])
    variant_hint = random.choice(variants) if variants else "any valid form"
    avoid_signatures = "; ".join(persisted_signatures[-5:]) if persisted_signatures else "none"

    prompt = build_generate_problem_prompt(
        topic=state["current_topic"],
        subtopic=state["current_subtopic"],
        difficulty=state["current_difficulty"],
        recent_problems=recent_problems,
        variant_hint=variant_hint,
        avoid_signatures=avoid_signatures,
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
                # Reject exact duplicates of problems already seen this session
                if pt in seen_problems:
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

    # For equivalent_fractions, the answer must be fully reduced — reject any
    # fraction a/b where gcd(a, b) != 1 (e.g. "36/48" is wrong; "3/4" is right).
    if result and state.get("current_subtopic") == "equivalent_fractions":
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", state["student_answer"].strip())
        if m and gcd(int(m.group(1)), int(m.group(2))) != 1:
            result = False

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
            "subtopic": state["current_subtopic"],
            "is_correct": evaluation.get("is_correct", False),
            "difficulty": state["current_difficulty"],
            "problem_text": state.get("current_problem", ""),
        }
    )

    return {
        "mastery": mastery,
        "subtopic_mastery": _subtopic_mastery_summary(student),
        "session_history": history,
    }


def adapt_next_node(state: TutorState) -> dict:
    # Difficulty is now per-subtopic and authoritative: every "Next problem"
    # re-enters the graph at load_state -> select_topic, which recomputes
    # difficulty from the freshly-saved subtopic mastery. Nothing to adjust here.
    return {"current_difficulty": state.get("current_difficulty", 2)}
