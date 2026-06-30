"""Tests for SymPy-verified solution step generation (no LLM required).

Run with: .venv/bin/python -m pytest tests/test_solution_steps.py -v
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sympy import Integer, simplify, sympify

from src.solution_steps import (
    generate_solution_steps,
    linear_eval_solution_steps,
    poly_eval_solution_steps,
    polynomial_latex,
    slope_solution_steps,
)


# ── Linear relationships: slope ──────────────────────────────────────────────

def test_slope_negative_coordinates_reported_case():
    # The exact problem that previously produced a wrong stored answer (3/7):
    # the derivation must restate the formula, show the substitution, and reduce
    # to -5/7 — one operation per step.
    steps = slope_solution_steps((-2, 4), (5, -1))
    assert steps[0] == r"$m = \frac{y_2 - y_1}{x_2 - x_1}$"     # restatement / formula
    assert any(r"\frac{-1 - 4}{5 - (-2)}" in s for s in steps)  # substitution, signs intact
    assert steps[-1].endswith(r"-\frac{5}{7}$")                # reduced answer


def test_slope_integer_result_shows_reduction():
    steps = slope_solution_steps((0, 0), (2, 6))
    assert any(r"\frac{6 - 0}{2 - 0}" in s for s in steps)
    assert any(r"\frac{6}{2}" in s for s in steps)
    assert steps[-1].endswith("= 3$")


def test_slope_already_reduced_omits_trailing_step():
    # 9/3 reduces to 3 (shown), but a slope already in lowest terms must not
    # append a redundant identical reduction step.
    steps = slope_solution_steps((1, 2), (4, 5))   # (5-2)/(4-1) = 3/3 -> 1
    assert steps[-1].endswith("= 1$")
    steps = slope_solution_steps((1, 1), (3, 2))   # (2-1)/(3-1) = 1/2, already reduced
    assert len(steps) == 3                          # formula, substitution, raw (no reduce)
    assert steps[-1].endswith(r"\frac{1}{2}$")


def test_slope_vertical_line_raises():
    with pytest.raises(Exception):
        slope_solution_steps((3, 1), (3, 9))       # undefined slope


# ── Linear relationships: evaluate y ─────────────────────────────────────────

def test_linear_eval_basic():
    steps = linear_eval_solution_steps(2, 1, 3)
    assert steps == ["$y = 2x + 1$", "$y = 2(3) + 1$", "$y = 6 + 1$", "$y = 7$"]


def test_linear_eval_unit_and_negative_coefficients():
    steps = linear_eval_solution_steps(-1, -3, 4)
    assert steps == ["$y = -x - 3$", "$y = -1(4) - 3$", "$y = -4 - 3$", "$y = -7$"]


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
    assert steps[0] == r"$\frac{1}{6} + \frac{2}{3}$"   # bare restatement first
    assert "\\frac{4}{6}" in steps[1]                   # converted to the LCD
    assert steps[-1].endswith("\\frac{5}{6}$")


def test_fraction_addition_same_denominator_with_simplification():
    steps = generate_solution_steps("Rational(3,8) + Rational(1,8)")
    assert steps[0] == r"$\frac{3}{8} + \frac{1}{8}$"   # bare restatement first
    assert any("\\frac{4}{8}" in s for s in steps)
    assert steps[-1].endswith("\\frac{1}{2}$")


def test_fraction_subtraction():
    steps = generate_solution_steps("Rational(3,4) - Rational(1,6)")
    assert steps[0] == r"$\frac{3}{4} - \frac{1}{6}$"   # bare restatement first
    assert "\\frac{9}{12}" in steps[1] and "\\frac{2}{12}" in steps[1]
    assert steps[-1].endswith("\\frac{7}{12}$")


def test_fraction_addition_integer_result():
    steps = generate_solution_steps("Rational(1,2) + Rational(1,2)")
    assert steps[-1].endswith("1$")


# ── Fraction multiplication / division ────────────────────────────────────────

def test_fraction_multiplication():
    steps = generate_solution_steps("Rational(3,4) * Rational(2,5)")
    assert steps[0] == r"$\frac{3}{4} \times \frac{2}{5}$"   # bare restatement first
    assert any("3 \\times 2" in s and "4 \\times 5" in s for s in steps)
    assert steps[-1].endswith("\\frac{3}{10}$")


def test_fraction_multiplication_needs_simplification():
    steps = generate_solution_steps("Rational(2,3) * Rational(3,4)")
    assert any("\\frac{6}{12}" in s for s in steps)
    assert steps[-1].endswith("\\frac{1}{2}$")


def test_fraction_division():
    steps = generate_solution_steps("Rational(1,2) / Rational(3,4)")
    assert "\\div" in steps[0]                               # restate the quotient
    assert any("\\times \\frac{4}{3}" in s for s in steps)   # invert and multiply
    assert steps[-1].endswith("\\frac{2}{3}$")


# ── Generic fallback ──────────────────────────────────────────────────────────

def test_integer_product_uses_generic_step():
    steps = generate_solution_steps("3 * 4")
    assert len(steps) == 2                       # restatement + answer
    assert steps[0] == "$3 \\cdot 4$"
    assert steps[-1].endswith("12$")


def test_three_factor_product_falls_back():
    steps = generate_solution_steps("Rational(1,2) * 6 * 4")
    assert len(steps) == 2                       # restatement + answer
    assert steps[-1].endswith("12$")


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


# ── Evaluating expressions (substitute a value into a polynomial) ─────────────

@pytest.mark.parametrize("coeffs,v", [
    ([-2, 3, 1], -8),    # the reported regression: 3x - 2x^2 + 1 at x=-8  ->  -151
    ([8, -14, 7], 23),
    ([1, 0, -5], 4),     # x^2 - 5 at x=4  ->  11   (a missing middle term)
    ([5, 3], 4),         # linear: 5x + 3 at x=4  ->  23
    ([-3, -8, 5], -9),
])
def test_poly_eval_steps(coeffs, v):
    d = len(coeffs) - 1
    expected = sum(Integer(c) * Integer(v) ** (d - i) for i, c in enumerate(coeffs))
    steps = poly_eval_solution_steps(coeffs, v)
    # starts from the displayed problem expression
    assert steps[0] == f"${polynomial_latex(coeffs, 'x')}$"
    # shows the substitution, not a bare jump to the answer
    assert len(steps) >= 3
    assert f"({v})" in steps[1]
    # ends at the correct, verified answer
    assert sympify(steps[-1].strip("$")) == expected
