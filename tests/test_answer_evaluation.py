"""
Tests for the answer evaluation stage.
Given a known sympy_answer (pre-computed by us, not the LLM), verifies
that symbolic_check correctly marks student answers right or wrong.

These tests do NOT call the LLM — they isolate the evaluation logic.

Run with: .venv/bin/python -m pytest tests/test_answer_evaluation.py -v
"""
import re
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes import symbolic_check


def _equiv_frac_check(student_answer: str, sympy_answer: str) -> bool | None:
    """Mirror the evaluate_answer_node logic for equivalent_fractions subtopic."""
    result = symbolic_check(student_answer, sympy_answer)
    if result:
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", student_answer.strip())
        if m and gcd(int(m.group(1)), int(m.group(2))) != 1:
            result = False
    return result


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

    # Answers with units stripped (e.g. rate answers like "5 miles")
    ("24",   "24",   True,  "answer with no units"),

    # Percentages — trailing % is stripped ("what percent" answers)
    ("50%",  "50",   True,  "50% accepted for a 'what percent' answer of 50"),
    ("50",   "50",   True,  "plain percent value without sign"),

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


# ── equivalent_fractions subtopic: must be in lowest terms ────────────────────

def test_equiv_frac_reduced_accepted():
    assert _equiv_frac_check("3/4", "3/4") is True

def test_equiv_frac_unreduced_rejected():
    # 36/48 is numerically correct (== 3/4) but not reduced — must be rejected
    assert _equiv_frac_check("36/48", "3/4") is False

def test_equiv_frac_partially_reduced_rejected():
    # 6/8 simplifies to 3/4 but still has gcd=2 — also wrong
    assert _equiv_frac_check("6/8", "3/4") is False

def test_equiv_frac_decimal_accepted():
    # Decimals bypass the fraction-form check (user typed 0.75, not a/b)
    assert _equiv_frac_check("0.75", "3/4") is True
