"""Tests for SymPy-verified solution step generation (no LLM required).

Run with: .venv/bin/python -m pytest tests/test_solution_steps.py -v
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sympy import simplify

from src.agent.solution_steps import (
    generate_solution_steps,
    linear_eval_solution_steps,
    slope_solution_steps,
)


# ── Linear relationships: slope ──────────────────────────────────────────────

def test_slope_negative_coordinates_reported_case():
    # The exact problem that previously produced a wrong stored answer (3/7):
    # the derivation must show the substitution and reduce to -5/7.
    steps = slope_solution_steps((-2, 4), (5, -1))
    assert len(steps) == 1
    step = steps[0]
    assert r"\frac{y_2 - y_1}{x_2 - x_1}" in step      # formula shown
    assert r"\frac{-1 - 4}{5 - (-2)}" in step          # substitution, signs intact
    assert step.endswith(r"-\frac{5}{7}$")             # reduced answer


def test_slope_integer_result_shows_reduction():
    steps = slope_solution_steps((0, 0), (2, 6))
    assert r"\frac{6 - 0}{2 - 0}" in steps[0]
    assert r"\frac{6}{2}" in steps[0]
    assert steps[0].endswith("= 3$")


def test_slope_already_reduced_omits_trailing_step():
    # 9/3 reduces to 3 (shown), but a slope already in lowest terms must not
    # append a redundant identical step.
    steps = slope_solution_steps((1, 2), (4, 5))   # (5-2)/(4-1) = 3/3 -> 1
    assert steps[0].endswith("= 1$")
    steps = slope_solution_steps((1, 1), (3, 2))   # (2-1)/(3-1) = 1/2, already reduced
    assert steps[0].count("=") == 3                # formula = sub = raw  (no extra)
    assert steps[0].endswith(r"\frac{1}{2}$")


def test_slope_vertical_line_raises():
    with pytest.raises(Exception):
        slope_solution_steps((3, 1), (3, 9))       # undefined slope


# ── Linear relationships: evaluate y ─────────────────────────────────────────

def test_linear_eval_basic():
    steps = linear_eval_solution_steps(2, 1, 3)
    assert steps == ["$y = 2x + 1 = 2(3) + 1 = 6 + 1 = 7$"]


def test_linear_eval_unit_and_negative_coefficients():
    steps = linear_eval_solution_steps(-1, -3, 4)
    assert steps == ["$y = -x - 3 = -1(4) - 3 = -4 - 3 = -7$"]


# ── Linear equations ──────────────────────────────────────────────────────────

def test_classic_two_step_equation():
    steps = generate_solution_steps("Eq(2*x + 5, 11)")
    assert len(steps) == 3
    assert steps[0] == "$2 x + 5 = 11$"
    assert "11 - 5" in steps[1] and "= 6" in steps[1]
    assert steps[2].endswith("= 3$")


def test_subtraction_equation():
    steps = generate_solution_steps("Eq(3*x - 9, 12)")
    assert len(steps) == 3
    assert "12 + 9" in steps[1] and "= 21" in steps[1]
    assert steps[2].endswith("= 7$")


def test_unit_coefficient():
    steps = generate_solution_steps("Eq(x + 4, 7)")
    assert len(steps) == 2
    assert "7 - 4" in steps[1] and "= 3" in steps[1]


def test_no_constant():
    steps = generate_solution_steps("Eq(3*x, 12)")
    assert len(steps) == 2
    assert steps[0] == "$3 x = 12$"
    assert steps[1].endswith("= 4$")


def test_variable_on_right():
    steps = generate_solution_steps("Eq(11, 2*x + 5)")
    assert steps[-1].endswith("= 3$")


def test_negative_solution():
    steps = generate_solution_steps("Eq(2*x + 10, 4)")
    assert steps[-1].endswith("= -3$")


def test_fractional_solution():
    steps = generate_solution_steps("Eq(2*x - 7, 0)")
    assert "\\frac{7}{2}" in steps[-1]


# ── Fraction addition / subtraction ──────────────────────────────────────────

def test_fraction_addition_different_denominators():
    steps = generate_solution_steps("Rational(1,6) + Rational(2,3)")
    assert len(steps) == 2
    assert "\\frac{4}{6}" in steps[0]            # converted to the LCD
    assert steps[1].endswith("\\frac{5}{6}$")


def test_fraction_addition_same_denominator_with_simplification():
    steps = generate_solution_steps("Rational(3,8) + Rational(1,8)")
    assert "\\frac{4}{8}" in steps[0]
    assert steps[-1].endswith("\\frac{1}{2}$")


def test_fraction_subtraction():
    steps = generate_solution_steps("Rational(3,4) - Rational(1,6)")
    assert "\\frac{9}{12}" in steps[0] and "\\frac{2}{12}" in steps[0]
    assert steps[-1].endswith("\\frac{7}{12}$")


def test_fraction_addition_integer_result():
    steps = generate_solution_steps("Rational(1,2) + Rational(1,2)")
    assert steps[-1].endswith("1$")


# ── Fraction multiplication / division ────────────────────────────────────────

def test_fraction_multiplication():
    steps = generate_solution_steps("Rational(3,4) * Rational(2,5)")
    assert "3 \\times 2" in steps[0] and "4 \\times 5" in steps[0]
    assert steps[-1].endswith("\\frac{3}{10}$")


def test_fraction_multiplication_needs_simplification():
    steps = generate_solution_steps("Rational(2,3) * Rational(3,4)")
    assert "\\frac{6}{12}" in steps[0]
    assert steps[-1].endswith("\\frac{1}{2}$")


def test_fraction_division():
    steps = generate_solution_steps("Rational(1,2) / Rational(3,4)")
    assert "\\div" in steps[0] and "\\times \\frac{4}{3}" in steps[0]
    assert steps[-1].endswith("\\frac{2}{3}$")


# ── Generic fallback ──────────────────────────────────────────────────────────

def test_integer_product_uses_generic_step():
    steps = generate_solution_steps("3 * 4")
    assert len(steps) == 1
    assert steps[0].endswith("= 12$")


def test_three_factor_product_falls_back():
    steps = generate_solution_steps("Rational(1,2) * 6 * 4")
    assert len(steps) == 1
    assert steps[0].endswith("= 12$")


def test_garbage_input_is_safe():
    assert generate_solution_steps("garbage(((") == []
    assert generate_solution_steps("") == []


def test_multivariable_equation_is_safe():
    # cannot derive a unique numeric answer — must not emit wrong steps
    steps = generate_solution_steps("Eq(x + y, 7)")
    assert steps == []


# ── Meta: every emitted equality must be symbolically true ────────────────────

EXPRESSIONS = [
    "Eq(2*x + 5, 11)",
    "Eq(3*x - 9, 12)",
    "Eq(x + 4, 7)",
    "Eq(3*x, 12)",
    "Eq(11, 2*x + 5)",
    "Eq(2*x - 7, 0)",
    "Rational(1,6) + Rational(2,3)",
    "Rational(3,8) + Rational(1,8)",
    "Rational(3,4) - Rational(1,6)",
    "Rational(3,4) * Rational(2,5)",
    "Rational(2,3) * Rational(3,4)",
    "Rational(1,2) / Rational(3,4)",
    "3 * 4",
    "Rational(1,2) * 6 * 4",
]


def _latex_atom_to_value(atom: str):
    """Parse the simple LaTeX atoms our generator emits into a SymPy value."""
    atom = atom.strip()
    atom = atom.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    atom = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", atom)
    atom = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", atom)  # nested
    if re.search(r"[a-zA-Z\\]", atom):
        return None  # contains a variable or unhandled command — skip
    return simplify(atom)


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_all_numeric_equalities_hold(expression):
    """Within each step, every `a = b = c` chain of pure numbers must be true."""
    for step in generate_solution_steps(expression):
        inner = step.strip("$")
        parts = inner.split(" = ")
        values = [_latex_atom_to_value(p) for p in parts]
        numeric = [v for v in values if v is not None]
        for left, right in zip(numeric, numeric[1:]):
            assert simplify(left - right) == 0, (
                f"step {step!r} contains a false equality: {left} != {right}"
            )
