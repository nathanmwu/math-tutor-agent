"""
Tests for the answer evaluation stage.
Given a known sympy_answer (pre-computed by us, not the LLM), verifies
that symbolic_check correctly marks student answers right or wrong.

These tests do NOT call the LLM — they isolate the evaluation logic.

Run with: .venv/bin/python -m pytest tests/test_answer_evaluation.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes import symbolic_check


# Each entry: (student_answer, sympy_answer, expected, description)
CASES = [
    # Integers
    ("3",    "3",    True,  "2x+5=11 -> x=3, student writes 3"),
    ("4",    "3",    False, "2x+5=11 -> x=3, student writes 4 (wrong)"),
    ("-2",   "-2",   True,  "negative solution"),
    ("0",    "0",    True,  "zero solution"),

    # Fractions — exact and equivalent forms
    ("5/6",  "5/6",  True,  "fraction exact match"),
    ("2/3",  "4/6",  True,  "2/3 == 4/6 (equivalent)"),
    ("4/6",  "2/3",  True,  "4/6 == 2/3 (equivalent, reversed)"),
    ("10/12","5/6",  True,  "10/12 == 5/6"),
    ("1/2",  "5/6",  False, "1/2 != 5/6"),

    # Decimals
    ("0.5",  "1/2",  True,  "0.5 == 1/2"),
    ("0.75", "3/4",  True,  "0.75 == 3/4"),
    ("0.25", "3/4",  False, "0.25 != 3/4"),

    # Geometry answers with units stripped
    ("24",   "24",   True,  "area answer, no units"),

    # Wrong types / unparseable
    ("",     "3",    None,  "empty input"),
    ("abc",  "3",    None,  "word instead of number"),
]


def test_all_cases():
    failures = []
    for student, answer, expected, desc in CASES:
        result = symbolic_check(student, answer)
        if result != expected:
            failures.append(
                f"FAIL [{desc}]  "
                f"symbolic_check({student!r}, {answer!r}) = {result}  "
                f"(expected {expected})"
            )
        else:
            print(f"PASS [{desc}]")

    if failures:
        raise AssertionError("\n" + "\n".join(failures))


def test_integer_answers():
    assert symbolic_check("3", "3") is True
    assert symbolic_check("4", "3") is False
    assert symbolic_check("-2", "-2") is True


def test_equivalent_fractions():
    assert symbolic_check("2/3", "4/6") is True
    assert symbolic_check("4/6", "2/3") is True
    assert symbolic_check("10/12", "5/6") is True


def test_wrong_fractions():
    assert symbolic_check("1/2", "5/6") is False
    assert symbolic_check("3/9", "5/6") is False


def test_unparseable_returns_none():
    assert symbolic_check("", "3") is None
    assert symbolic_check("abc", "3") is None
