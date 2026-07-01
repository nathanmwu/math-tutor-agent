from __future__ import annotations

import json
import os
import random
import re
from math import gcd
from pathlib import Path
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sympy import Eq, Rational, simplify, solve, sympify

from src.prompts import (
    CATEGORIZE_ERROR_PROMPT,
    FEEDBACK_CORRECT_PROMPT,
    FEEDBACK_INCORRECT_PROMPT,
    SUBTOPIC_VARIANTS,
    build_generate_problem_prompt,
)
from src.solution_steps import (
    generate_solution_steps,
    linear_eval_solution_steps,
    poly_eval_solution_steps,
    polynomial_latex,
    slope_solution_steps,
)
from src.knowledge import retrieve_content
from src.student import (
    StudentState,
    SubtopicMastery,
    difficulty_for_mastery,
    problem_signature,
    record_attempt,
    subtopic_key,
)


class TutorState(TypedDict):
    student_id: str
    current_topic: str
    current_subtopic: str
    current_difficulty: int
    current_problem: str
    sympy_expression: str       # the raw computation, e.g. "Eq(2*x + 5, 11)"
    sympy_answer: str           # SymPy-computed result; never shown before answering
    solution_steps: list[str]   # SymPy-verified LaTeX derivation, rendered verbatim
    student_answer: str
    evaluation: dict
    feedback: str
    mastery: dict[str, float]
    # Per-subtopic mastery summary for the UI focus-areas panel, keyed by
    # subtopic_key(topic, subtopic) -> {"topic", "subtopic", "mastery_score"}.
    subtopic_mastery: dict[str, dict]
    session_history: list[dict]

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


def _get_llm() -> ChatOllama:
    """The single Ollama client used for every generation step."""
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3:8b"), temperature=0.7)


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


def _subtopic_priority(sm: SubtopicMastery) -> float:
    """Selection weight: higher = weaker = practice more.

    Combines low mastery with a recent-error boost. The floor keeps every
    subtopic reachable (a zero weight would make it unselectable).
    """
    weakness = 1.0 - sm.mastery_score
    error_rate = min(sum(sm.error_pattern_counts.values()) / max(sm.attempts, 1), 1.0)
    return max(0.05, weakness + 0.5 * error_rate)


def setup_node(state: TutorState) -> dict:
    """Load the student, surface their mastery for the UI, and pick the next
    (topic, subtopic, difficulty) — favoring weak subtopics.

    Weighted-random over a transparent per-subtopic priority score, with a
    cold-start path that guarantees breadth and an exploration fraction that
    preserves variety. Difficulty tracks the demonstrated topic mastery (the
    same signal as the progress bar), so a strong student keeps getting
    challenge problems even when a fresh subtopic is selected.
    """
    student = StudentState.load_or_new(state["student_id"], STUDENT_STATE_DIR)
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

    tm = student.topic_mastery.get(topic)
    topic_score = tm.mastery_score if tm is not None else 0.0

    return {
        "mastery": {t: m.mastery_score for t, m in student.topic_mastery.items()},
        "subtopic_mastery": _subtopic_mastery_summary(student),
        "current_topic": topic,
        "current_subtopic": subtopic,
        "current_difficulty": difficulty_for_mastery(topic_score),
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
        if avoid_signatures and problem_signature(problem_text) in avoid_signatures:
            continue
        return {
            "current_problem": problem_text,
            "sympy_answer": str(sympify(sympy_expression, rational=True)),
            "sympy_expression": sympy_expression,
            "solution_steps": steps,
            "student_answer": "",
            "evaluation": {},
            "feedback": "",
        }
    return None


def _substituted_expr(coeffs: list[int], v: int) -> str:
    """Python/SymPy string of the polynomial with x replaced by v (for storage)."""
    d = len(coeffs) - 1
    vv = f"({v})" if v < 0 else str(v)
    terms = []
    for i, c in enumerate(coeffs):
        p = d - i
        mono = "" if p == 0 else (f"*{vv}" if p == 1 else f"*{vv}**{p}")
        terms.append((c, f"{abs(c)}{mono}"))
    s = ("-" if terms[0][0] < 0 else "") + terms[0][1]
    for c, t in terms[1:]:
        s += (" - " if c < 0 else " + ") + t
    return s


def _build_evaluating_expression_problem(
    difficulty: int, seen_problems: set[str], avoid_signatures: set[str] | None = None
) -> dict | None:
    """Deterministically build an ``evaluating_expressions`` problem (linear or
    quadratic). The prose, the answer, and the verified derivation all derive
    from one set of integer coefficients and a substitution value, so they cannot
    drift — the failure mode of letting the LLM hand-substitute (it would author
    prose for one expression and a substituted arithmetic for another, e.g. show
    "3x - 2x^2 + 1 at x=-8" but compute -511).
    """
    hi = 3 + 2 * difficulty

    def _nz(lo: int, h: int) -> int:
        return random.choice([n for n in range(lo, h + 1) if n != 0])

    for _ in range(50):
        v = _nz(-hi, hi)  # never x = 0 (it collapses the expression)
        if difficulty <= 2:
            coeffs = [_nz(-hi, hi), _nz(-hi, hi)]                 # a x + b
        else:
            coeffs = [_nz(-hi, hi), _nz(-hi, hi), _nz(-hi, hi)]   # a x^2 + b x + c

        problem_text = rf"Evaluate ${polynomial_latex(coeffs, 'x')}$ at $x = {v}$"
        if problem_text in seen_problems:
            continue
        if avoid_signatures and problem_signature(problem_text) in avoid_signatures:
            continue

        d = len(coeffs) - 1
        answer = sum(c * v ** (d - i) for i, c in enumerate(coeffs))
        return {
            "current_problem": problem_text,
            "sympy_answer": str(answer),
            "sympy_expression": _substituted_expr(coeffs, v),
            "solution_steps": poly_eval_solution_steps(coeffs, v),
            "student_answer": "",
            "evaluation": {},
            "feedback": "",
        }
    return None


def _finalize_problem(
    problem_text: str, sympy_expression: str, seen_problems: set[str]
) -> dict | None:
    """Validate a candidate problem and build the node result, or return None.

    We evaluate the expression ourselves (never trusting an LLM's arithmetic):
    an ``Eq`` must solve to a single value that checks back, anything else must
    reduce to one concrete number, and exact session duplicates are rejected.
    Shared by the LLM loop and the deterministic fallback so both are validated
    identically.
    """
    try:
        computed = sympify(
            sympy_expression, locals={"Rational": Rational, "Eq": Eq}, rational=True
        )
        if isinstance(computed, Eq):
            free = computed.free_symbols
            if len(free) != 1:
                return None
            var = list(free)[0]
            solutions = solve(computed, var)
            if len(solutions) != 1:
                return None
            candidate = solutions[0]
            if not computed.subs(var, candidate):  # substitute back — catch typos
                return None
            computed = candidate
        elif isinstance(computed, list):
            if len(computed) != 1:
                return None
            computed = computed[0]
        if not getattr(computed, "is_number", False):
            return None
        if problem_text in seen_problems:
            return None
        return {
            "current_problem": problem_text,
            "sympy_answer": str(computed),
            "sympy_expression": sympy_expression,
            "solution_steps": generate_solution_steps(sympy_expression),
            "student_answer": "",
            "evaluation": {},
            "feedback": "",
        }
    except Exception:
        return None


def _fallback_problem(
    subtopic: str, difficulty: int, seen_problems: set[str]
) -> dict | None:
    """Deterministic, on-topic, difficulty-scaled problem used when the LLM
    generator fails — so the student is never dropped to an off-topic placeholder
    (the old ``2 + 2``). Number magnitude grows with difficulty; every candidate
    passes the same validation as an LLM-authored one.
    """
    hi = 3 + 2 * difficulty  # number magnitude grows with difficulty

    def _nonzero(lo: int, h: int) -> int:
        return random.choice([n for n in range(lo, h + 1) if n != 0])

    for _ in range(40):
        problem_text = sympy_expression = None

        if subtopic == "linear_relationships":
            built = _build_linear_relationship_problem(difficulty, seen_problems)
            if built is not None:
                return built
            continue
        elif subtopic == "equivalent_fractions":
            b = random.randint(2, 6)
            a = random.randint(1, b - 1)
            k = random.randint(2, 2 + difficulty)
            problem_text = rf"$\frac{{{a * k}}}{{{b * k}}} =$"
            sympy_expression = f"Rational({a * k},{b * k})"
        elif subtopic == "addition_subtraction":
            d1, d2 = random.randint(2, hi), random.randint(2, hi)
            n1, n2 = random.randint(1, d1), random.randint(1, d2)
            latex_op, py_op = random.choice([("+", "+"), ("-", "-")])
            problem_text = rf"$\frac{{{n1}}}{{{d1}}} {latex_op} \frac{{{n2}}}{{{d2}}} =$"
            sympy_expression = f"Rational({n1},{d1}) {py_op} Rational({n2},{d2})"
        elif subtopic == "multiplication_division":
            d1, d2 = random.randint(2, hi), random.randint(2, hi)
            n1, n2 = random.randint(1, d1), random.randint(1, d2)
            latex_op, py_op = random.choice([(r"\times", "*"), (r"\div", "/")])
            problem_text = rf"$\frac{{{n1}}}{{{d1}}} {latex_op} \frac{{{n2}}}{{{d2}}} =$"
            sympy_expression = f"Rational({n1},{d1}) {py_op} Rational({n2},{d2})"
        elif subtopic == "percentages":
            n = random.randint(1, hi) * 20  # multiple of 20 keeps the answer whole
            p = random.choice([5, 10, 20, 25, 40, 50, 75])
            problem_text = rf"What is ${p}\%$ of ${n}$?"
            sympy_expression = f"Rational({p},100) * {n}"
        elif subtopic == "proportions":
            b = random.randint(2, 6)
            a = random.randint(1, hi)
            c = b * random.randint(1, 1 + difficulty)  # keeps x a whole number
            problem_text = rf"$\frac{{{a}}}{{{b}}} = \frac{{x}}{{{c}}}$,  $x = ?$"
            sympy_expression = f"Eq(Rational({a},{b}), x/{c})"
        elif subtopic == "linear_equations":
            x0 = random.randint(1, hi)
            a = random.randint(2, 2 + difficulty)
            b = _nonzero(-hi, hi)
            c = a * x0 + b
            sign = "+" if b >= 0 else "-"
            problem_text = rf"${a}x {sign} {abs(b)} = {c}$,  $x = ?$"
            sympy_expression = f"Eq({a}*x {sign} {abs(b)}, {c})"
        elif subtopic == "evaluating_expressions":
            built = _build_evaluating_expression_problem(difficulty, seen_problems)
            if built is not None:
                return built
            continue

        if problem_text is None:
            continue
        result = _finalize_problem(problem_text, sympy_expression, seen_problems)
        if result is not None:
            return result

    return None


def generate_problem_node(state: TutorState) -> dict:
    history = state.get("session_history", [])
    seen_problems = {item.get("problem_text", "") for item in history}

    # Persisted cross-session avoid-list of problem SHAPES for this subtopic.
    student = StudentState.load_or_new(state["student_id"], STUDENT_STATE_DIR)
    key = subtopic_key(state["current_topic"], state["current_subtopic"])
    persisted_signatures = student.recent_signatures.get(key, [])

    # linear_relationships and evaluating_expressions are generated deterministically
    # (no LLM): the prose, the answer, and the derivation come from one source of
    # truth, so they can never drift apart (the LLM hand-substituting was unreliable).
    if state["current_subtopic"] == "linear_relationships":
        built = _build_linear_relationship_problem(
            state["current_difficulty"], seen_problems, set(persisted_signatures)
        )
        if built is not None:
            return built
    if state["current_subtopic"] == "evaluating_expressions":
        built = _build_evaluating_expression_problem(
            state["current_difficulty"], seen_problems, set(persisted_signatures)
        )
        if built is not None:
            return built

    llm = _get_llm()

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

    for _ in range(3):
        response = llm.invoke(prompt)
        try:
            data = _parse_problem_json(response.content.strip())
            pt = data.get("problem_text", "").strip()
            expr_str = data.get("sympy_expression", "").strip()
        except Exception:
            continue
        if not (pt and expr_str):
            continue
        result = _finalize_problem(pt, expr_str, seen_problems)
        if result is not None:
            return result

    # The LLM failed to produce a valid problem — serve a DETERMINISTIC, on-topic,
    # difficulty-scaled problem instead of an off-topic placeholder.
    fallback = _fallback_problem(
        state["current_subtopic"], state["current_difficulty"], seen_problems
    )
    if fallback is not None:
        return fallback

    # Absolute last resort: a difficulty-scaled sum (never a stale "2 + 2").
    a, b = random.randint(2, 3 + 2 * state["current_difficulty"]), random.randint(
        2, 3 + 2 * state["current_difficulty"]
    )
    return _finalize_problem(rf"${a} + {b} =$", f"{a} + {b}", set())


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

    llm = _get_llm()
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


def explain_node(state: TutorState) -> dict:
    """Retrieve grounding chunks (RAG) and write the concept-note feedback.

    The math derivation is the SymPy-verified ``solution_steps``; the LLM writes
    only a short note, grounded in the retrieved knowledge-base content.
    """
    evaluation = state.get("evaluation", {})
    chunks = retrieve_content(
        topic=state["current_topic"],
        subtopic=state["current_subtopic"],
        difficulty=state["current_difficulty"],
        chroma_dir=CHROMA_DIR,
        error_category=evaluation.get("error_category"),
        query_text=state.get("current_problem"),
    )
    steps = state.get("solution_steps", [])
    template = (
        FEEDBACK_CORRECT_PROMPT
        if evaluation.get("is_correct", False)
        else FEEDBACK_INCORRECT_PROMPT
    )
    prompt = template.format(
        problem=state["current_problem"],
        student_answer=state["student_answer"],
        correct_answer=state["sympy_answer"],
        solution_steps="\n".join(steps) if steps else "(not available)",
        retrieved_content="\n---\n".join(chunks) if chunks else "",
    )
    return {"feedback": _get_llm().invoke(prompt).content.strip()}


def update_state_node(state: TutorState) -> dict:
    evaluation = state.get("evaluation", {})
    student = StudentState.load_or_new(state["student_id"], STUDENT_STATE_DIR)
    student = record_attempt(
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
    student.save(STUDENT_STATE_DIR)

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


# ── Graph ─────────────────────────────────────────────────────────────────────
# A linear tutoring loop with one human-in-the-loop pause: after generate_problem
# the graph interrupts; the UI submits the student's answer and resumes at
# evaluate_answer. Next-turn difficulty is recomputed in setup_node from the
# freshly-saved mastery, so no separate "adapt" step is needed.
_builder = StateGraph(TutorState)
_builder.add_node("setup_node", setup_node)
_builder.add_node("generate_problem_node", generate_problem_node)
_builder.add_node("evaluate_answer_node", evaluate_answer_node)
_builder.add_node("explain_node", explain_node)
_builder.add_node("update_state_node", update_state_node)

_builder.add_edge(START, "setup_node")
_builder.add_edge("setup_node", "generate_problem_node")
_builder.add_edge("generate_problem_node", "evaluate_answer_node")
_builder.add_edge("evaluate_answer_node", "explain_node")
_builder.add_edge("explain_node", "update_state_node")
_builder.add_edge("update_state_node", END)

graph = _builder.compile(
    checkpointer=MemorySaver(), interrupt_before=["evaluate_answer_node"]
)
