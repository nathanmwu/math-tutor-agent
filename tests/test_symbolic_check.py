"""Tests for symbolic_check() and the answer evaluation pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.nodes import symbolic_check


# ── Exact match ───────────────────────────────────────────────────────────────

def test_integer_correct():
    assert symbolic_check("4", "4") is True

def test_integer_wrong():
    assert symbolic_check("3", "4") is False

def test_decimal_equals_integer():
    assert symbolic_check("4.0", "4") is True

def test_negative_correct():
    assert symbolic_check("-3", "-3") is True

def test_negative_wrong():
    assert symbolic_check("3", "-3") is False


# ── Fraction equivalence ──────────────────────────────────────────────────────

def test_fraction_exact():
    assert symbolic_check("5/6", "5/6") is True

def test_fraction_equivalent_simplified():
    # 4/6 == 2/3
    assert symbolic_check("4/6", "2/3") is True

def test_fraction_equivalent_unsimplified_answer():
    # sympy_answer stored as 4/6, student writes 2/3
    assert symbolic_check("2/3", "4/6") is True

def test_fraction_addition_result():
    # 1/6 + 2/3 = 5/6; sympy_answer computed as "5/6"
    assert symbolic_check("5/6", "5/6") is True

def test_fraction_addition_equivalent_form():
    # student writes 10/12 which equals 5/6
    assert symbolic_check("10/12", "5/6") is True

def test_fraction_addition_wrong():
    assert symbolic_check("3/9", "5/6") is False

def test_fraction_3_6_plus_1_6():
    # 3/6 + 1/6 = 4/6 = 2/3; student writes 2/3
    assert symbolic_check("2/3", "4/6") is True

def test_fraction_subtraction():
    # 3/4 - 1/4 = 1/2
    assert symbolic_check("1/2", "1/2") is True

def test_fraction_subtraction_equivalent():
    assert symbolic_check("2/4", "1/2") is True

def test_fraction_multiplication():
    # 2/3 * 3/4 = 1/2
    assert symbolic_check("1/2", "1/2") is True

def test_fraction_greater_than_one():
    # 5/4 == 1.25
    assert symbolic_check("5/4", "5/4") is True
    assert symbolic_check("1.25", "5/4") is True


# ── Mixed numbers ─────────────────────────────────────────────────────────────

def test_mixed_number_simple():
    # "1 1/2" == 3/2
    assert symbolic_check("1 1/2", "3/2") is True

def test_mixed_number_equals_improper():
    assert symbolic_check("1 3/4", "7/4") is True

def test_mixed_number_wrong():
    assert symbolic_check("1 1/2", "5/4") is False

def test_mixed_number_equivalent_decimal():
    assert symbolic_check("1 1/2", "1.5") is True


# ── Decimals ──────────────────────────────────────────────────────────────────

def test_decimal_correct():
    assert symbolic_check("0.75", "3/4") is True

def test_decimal_wrong():
    assert symbolic_check("0.5", "3/4") is False

def test_decimal_repeating_approximation():
    # 1/3 ≈ 0.333... — not equal, should be False
    assert symbolic_check("0.33", "1/3") is False


# ── Unit stripping ────────────────────────────────────────────────────────────

def test_strips_cm():
    assert symbolic_check("24 cm", "24") is True

def test_strips_sq_cm():
    assert symbolic_check("36 sq cm", "36") is True

def test_strips_meters():
    assert symbolic_check("5 m", "5") is True

def test_strips_kg():
    assert symbolic_check("10 kg", "10") is True

def test_strips_feet():
    assert symbolic_check("8 feet", "8") is True


# ── Algebra ───────────────────────────────────────────────────────────────────

def test_algebra_integer_solution():
    assert symbolic_check("5", "5") is True

def test_algebra_negative_solution():
    assert symbolic_check("-2", "-2") is True

def test_algebra_fraction_solution():
    assert symbolic_check("3/2", "3/2") is True

def test_algebra_x_equals_format():
    # student writes "x=3" — common for equation problems
    assert symbolic_check("x=3", "3") is True

def test_algebra_x_equals_with_spaces():
    assert symbolic_check("x = 3", "3") is True

def test_algebra_x_equals_negative():
    assert symbolic_check("x = -2", "-2") is True

def test_algebra_x_equals_fraction():
    assert symbolic_check("y = 5/2", "5/2") is True

def test_algebra_x_equals_mixed_number():
    assert symbolic_check("x = 1 1/2", "3/2") is True

def test_algebra_x_equals_wrong():
    assert symbolic_check("x=4", "3") is False

def test_algebra_bare_variable_still_none():
    # "x" alone with no value is still unparseable
    assert symbolic_check("x", "3") is None


# ── Unparseable / garbage input ───────────────────────────────────────────────

def test_empty_string():
    assert symbolic_check("", "5") is None

def test_word_answer():
    assert symbolic_check("five", "5") is None

def test_gibberish():
    assert symbolic_check("abc", "5") is None

def test_sentence():
    assert symbolic_check("I don't know", "5") is None

def test_expression_with_variable_vs_number():
    # student types "x" when answer is a number — should be None (parse error)
    assert symbolic_check("x", "5") is None


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_zero():
    assert symbolic_check("0", "0") is True

def test_zero_vs_nonzero():
    assert symbolic_check("0", "1") is False

def test_whitespace_trimmed():
    assert symbolic_check("  5  ", "5") is True

def test_large_integer():
    assert symbolic_check("1000", "1000") is True

def test_fraction_with_spaces():
    # "3 / 4" — SymPy should parse this
    assert symbolic_check("3 / 4", "3/4") is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
