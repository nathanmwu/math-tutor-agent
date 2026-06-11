"""
Tests for the problem generation stage.
Calls the LLM directly and verifies:
  1. The raw JSON response can be parsed
  2. sympy_expression is present and evaluable
  3. The computed sympy_answer has no free symbols (is a concrete number)

Run with: .venv/bin/python -m pytest tests/test_problem_generation.py -v -s
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest
from sympy import sympify

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from src.agent.prompts import GENERATE_PROBLEM_PROMPT


MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
REQUIRED_FIELDS = {"problem_text", "sympy_expression", "topic", "subtopic", "difficulty"}

CASES = [
    ("fractions",  "addition_subtraction",   2),
    ("fractions",  "multiplication_division", 2),
    ("algebra",    "linear_equations",        2),
    ("algebra",    "linear_equations",        3),
    ("ratios",     "unit_rates",              2),
    ("geometry",   "area_perimeter",          2),
]


def _call_llm(topic, subtopic, difficulty):
    """Call generate prompt once, return raw text."""
    llm = ChatOllama(model=MODEL, temperature=0.0)  # temp=0 for determinism
    prompt = GENERATE_PROBLEM_PROMPT.format(
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        recent_problems="none",
    )
    return llm.invoke(prompt).content.strip()


def _parse_response(raw):
    """Strip code fences and parse JSON."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def _evaluate_expression(expr_str):
    """Evaluate sympy_expression to a concrete numeric value.
    Handles Eq(lhs, rhs) by solving, and solve() lists by unwrapping."""
    from sympy import Eq as SymEq, solve as sym_solve
    result = sympify(
        expr_str,
        locals={"Rational": __import__("sympy").Rational, "Eq": SymEq},
        rational=True,
    )
    if isinstance(result, SymEq):
        free = result.free_symbols
        if len(free) != 1:
            raise ValueError(f"Eq has {len(free)} free symbols: {free}")
        solutions = sym_solve(result, list(free)[0])
        if len(solutions) != 1:
            raise ValueError(f"Eq has {len(solutions)} solutions: {solutions}")
        result = solutions[0]
    elif isinstance(result, list):
        if len(result) != 1:
            raise ValueError(f"solve() returned {len(result)} solutions: {result}")
        result = result[0]
    return result


@pytest.mark.parametrize("topic,subtopic,difficulty", CASES)
def test_llm_returns_valid_json(topic, subtopic, difficulty):
    raw = _call_llm(topic, subtopic, difficulty)
    print(f"\nRaw LLM output:\n{raw}")
    data = _parse_response(raw)
    assert isinstance(data, dict), "Response is not a JSON object"


@pytest.mark.parametrize("topic,subtopic,difficulty", CASES)
def test_required_fields_present(topic, subtopic, difficulty):
    raw = _call_llm(topic, subtopic, difficulty)
    data = _parse_response(raw)
    missing = REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Missing fields: {missing}  |  got: {list(data.keys())}"


@pytest.mark.parametrize("topic,subtopic,difficulty", CASES)
def test_sympy_expression_evaluates(topic, subtopic, difficulty):
    raw = _call_llm(topic, subtopic, difficulty)
    data = _parse_response(raw)
    expr_str = data.get("sympy_expression", "")
    print(f"\nproblem_text:    {data.get('problem_text')}")
    print(f"sympy_expression: {expr_str}")
    result = _evaluate_expression(expr_str)
    print(f"evaluated to:    {result}  (free_symbols={result.free_symbols})")
    assert not result.free_symbols, (
        f"sympy_expression '{expr_str}' evaluated to '{result}' "
        f"which still contains free symbols {result.free_symbols}"
    )


@pytest.mark.parametrize("topic,subtopic,difficulty", CASES)
def test_computed_answer_matches_problem(topic, subtopic, difficulty):
    """Smoke test: computed answer is a finite number (not symbolic, not error)."""
    raw = _call_llm(topic, subtopic, difficulty)
    data = _parse_response(raw)
    result = _evaluate_expression(data["sympy_expression"])
    print(f"\n  problem: {data['problem_text']}")
    print(f"  answer:  {result}")
    assert result.is_number, f"Expected a numeric answer, got: {result}"
