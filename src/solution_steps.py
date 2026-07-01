"""SymPy-computed solution derivations.

The LLM never writes math steps: every derivation shown to a student is
generated here, and every displayed equality is verified symbolically before
it is emitted. Every derivation leads with a bare restatement of the problem,
then applies one operation per step. If a structured builder can't decompose
the input, the generic fallback shows restatement + result — minimal, but
provably correct.
"""
from __future__ import annotations

from sympy import Eq, Integer, Mul, Pow, Rational, lcm, latex, simplify, solve, sympify

__all__ = [
    "generate_solution_steps",
    "slope_solution_steps",
    "linear_eval_solution_steps",
    "poly_eval_solution_steps",
    "polynomial_latex",
]


class _StepCheckError(Exception):
    """An internally generated step failed symbolic verification."""


def _frac(n, d) -> str:
    """LaTeX fraction from raw numerator/denominator (no simplification)."""
    return rf"\frac{{{latex(n)}}}{{{latex(d)}}}"


def _running_steps(*pairs) -> list[str]:
    """One numbered step per (latex, value) state, in order.

    The first step is the bare restatement of the problem; every later step
    applies a single operation to the previous one. Each consecutive pair of
    concrete numeric values is verified equal before display, preserving the
    "every emitted equality is symbolically true" guarantee. A state with value
    ``None`` (a formula or templated restatement that does not evaluate to a
    number — e.g. ``m = (y_2 - y_1)/(x_2 - x_1)``) is emitted verbatim and skips
    the check."""
    steps: list[str] = []
    prev = None
    for text, value in pairs:
        if prev is not None and value is not None and simplify(prev - value) != 0:
            raise _StepCheckError(f"{prev} != {value}")
        steps.append(f"${text}$")
        if value is not None:
            prev = value
    return steps


def _rational_latex(r: Rational) -> str:
    """LaTeX for a rational: integers plain, fractions as \\frac (sign in front)."""
    if r.q == 1:
        return latex(r)
    sign = "-" if r < 0 else ""
    return sign + _frac(abs(r.p), r.q)


def _terms_latex(terms: list[Rational]) -> str:
    """'\\frac{3}{4} - \\frac{1}{6}' from signed rational terms."""
    parts = [_rational_latex(terms[0])]
    for t in terms[1:]:
        parts.append(("- " if t < 0 else "+ ") + _rational_latex(abs(t)))
    return " ".join(parts)


# ── Linear equations ──────────────────────────────────────────────────────────

def _linear_equation_steps(eq: Eq) -> list[str]:
    free = eq.free_symbols
    if len(free) != 1:
        raise _StepCheckError("not a single-variable equation")
    x = free.pop()

    poly = (eq.lhs - eq.rhs).expand()
    if poly.has(x) and poly.diff(x).has(x):
        raise _StepCheckError("not linear")
    a = poly.coeff(x, 1)
    b = poly.coeff(x, 0)
    if a == 0:
        raise _StepCheckError("variable cancels out")
    if a < 0:  # keep the working form positive: -2x = -6  →  2x = 6
        a, b = -a, -b

    solution = simplify(-b / a)
    solutions = solve(eq, x)
    if solutions != [solution]:
        raise _StepCheckError("solution mismatch")

    def check_equation(display_eq: Eq) -> None:
        if solve(display_eq, x) != [solution]:
            raise _StepCheckError(f"step equation {display_eq} changes the solution")

    steps = [f"${latex(eq.lhs)} = {latex(eq.rhs)}$"]

    isolated_rhs = simplify(-b)  # value of a*x after moving the constant
    moved = Eq(a * x, isolated_rhs, evaluate=False)
    if moved != eq:  # skip when the equation already has the form a·x = c
        check_equation(moved)
        # show the arithmetic when the original is the common  a·x + b = c  shape
        lhs_const = eq.lhs.coeff(x, 0) if eq.lhs.has(x) else None
        if (
            lhs_const is not None
            and lhs_const != 0
            and not eq.rhs.has(x)
            and simplify(eq.rhs - lhs_const - isolated_rhs) == 0
        ):
            sign = "-" if lhs_const > 0 else "+"
            arithmetic = f"{latex(eq.rhs)} {sign} {latex(abs(lhs_const))}"
            steps.append(f"${latex(a * x)} = {arithmetic} = {latex(isolated_rhs)}$")
        else:
            steps.append(f"${latex(moved.lhs)} = {latex(moved.rhs)}$")

    if a != 1:
        division = _frac(isolated_rhs, a)
        if simplify(Rational(1) * isolated_rhs / a - solution) != 0:
            raise _StepCheckError("division arithmetic is wrong")
        if division == latex(solution):
            steps.append(f"${latex(x)} = {division}$")
        else:
            steps.append(f"${latex(x)} = {division} = {latex(solution)}$")

    return steps


# ── Linear relationships (slope, evaluate y) ─────────────────────────────────

def _signed_sub(minuend: Integer, subtrahend: Integer) -> str:
    """LaTeX for 'minuend - subtrahend', parenthesizing a negative subtrahend
    so the substitution reads '5 - (-2)' rather than the ambiguous '5 - -2'."""
    if subtrahend < 0:
        return rf"{latex(minuend)} - ({latex(subtrahend)})"
    return rf"{latex(minuend)} - {latex(subtrahend)}"


def slope_solution_steps(p1: tuple[int, int], p2: tuple[int, int]) -> list[str]:
    """Worked derivation for the slope through two points; every equality verified.

    Renders the formula, the substitution, the raw fraction, and the reduced
    value:  m = (y2 - y1)/(x2 - x1) = (d - b)/(c - a) = num/den = reduced.
    """
    (a, b), (c, d) = p1, p2
    a, b, c, d = Integer(a), Integer(b), Integer(c), Integer(d)
    num_raw, den_raw = d - b, c - a
    if den_raw == 0:
        raise _StepCheckError("vertical line — slope undefined")
    slope = Rational(num_raw, den_raw)

    substituted = rf"\frac{{{_signed_sub(d, b)}}}{{{_signed_sub(c, a)}}}"
    raw = _frac(num_raw, den_raw)
    reduced = _rational_latex(slope)

    # Step 1 restates the formula (the math setup); each later step applies one
    # operation: substitute the points, evaluate, then reduce.
    states = [
        (r"m = \frac{y_2 - y_1}{x_2 - x_1}", None),
        (rf"m = {substituted}", slope),
        (rf"m = {raw}", slope),
    ]
    if raw != reduced:  # skip the reduction step when already in lowest terms
        states.append((rf"m = {reduced}", slope))
    return _running_steps(*states)


def linear_eval_solution_steps(m: int, k: int, v: int) -> list[str]:
    """Worked derivation for y = m·x + k evaluated at x = v; every equality verified.

    Renders  y = m x + k = m(v) + k = product + k = answer.
    """
    m, k, v = Integer(m), Integer(k), Integer(v)
    answer = m * v + k

    def _plus(term: Integer) -> str:
        return rf"+ {latex(term)}" if term >= 0 else rf"- {latex(abs(term))}"

    coef = "" if m == 1 else "-" if m == -1 else latex(m)
    formula = rf"{coef}x {_plus(k)}"
    substituted = rf"{latex(m)}({latex(v)}) {_plus(k)}"
    arithmetic = rf"{latex(m * v)} {_plus(k)}"

    # Step 1 restates the given equation; each later step applies one operation:
    # substitute x, multiply, then add.
    return _running_steps(
        (rf"y = {formula}", None),
        (rf"y = {substituted}", answer),
        (rf"y = {arithmetic}", answer),
        (rf"y = {latex(answer)}", answer),
    )


# ── Evaluating expressions (substitute a value into a polynomial) ─────────────

def polynomial_latex(coeffs: list[int], base: str = "x") -> str:
    """Render a polynomial 'a·base^d + … + c' (coeffs highest-degree-first) as
    signed LaTeX. ``base`` is the variable rendering — 'x' for the problem
    statement, or a parenthesized value like '(-8)' for the substitution step.

    e.g. polynomial_latex([-2, 3, 1]) -> '-2x^{2} + 3x + 1'
         polynomial_latex([-2, 3, 1], '(-8)') -> '-2(-8)^{2} + 3(-8) + 1'
    """
    d = len(coeffs) - 1
    parts: list[str] = []
    for i, c in enumerate(coeffs):
        c, p = int(c), d - i
        if c == 0:
            continue
        mag = abs(c)
        mono = "" if p == 0 else base if p == 1 else f"{base}^{{{p}}}"
        coef = "" if (mag == 1 and p != 0) else str(mag)
        body = f"{coef}{mono}"
        if not parts:
            parts.append(f"-{body}" if c < 0 else body)
        else:
            parts.append(("- " if c < 0 else "+ ") + body)
    return " ".join(parts) if parts else "0"


def poly_eval_solution_steps(coeffs: list[int], v: int) -> list[str]:
    """Worked derivation for evaluating a polynomial (coeffs highest-degree-first)
    at x = v; every numeric equality verified. Starts from the problem expression,
    substitutes, evaluates each term, then totals:

        8x^2 - 14x + 7 = 8(23)^2 - 14(23) + 7 = 4232 - 322 + 7 = 3917
    """
    ints = [Integer(c) for c in coeffs]
    v = Integer(v)
    d = len(ints) - 1
    term_values = [c * v ** (d - i) for i, c in enumerate(ints) if c != 0]
    answer = sum(term_values, Integer(0))

    states = [
        (polynomial_latex(coeffs, "x"), None),            # restate the problem
        (polynomial_latex(coeffs, f"({latex(v)})"), answer),  # substitute x = v
    ]
    term_sum = _terms_latex(term_values)                  # evaluate each term
    if term_sum != latex(answer):
        states.append((term_sum, answer))
    states.append((latex(answer), answer))                # total
    return _running_steps(*states)


# ── Fraction arithmetic ───────────────────────────────────────────────────────

def _as_signed_rationals(expr) -> list[Rational]:
    """Flatten an unevaluated Add into signed Rational terms, in input order."""
    terms = []
    for arg in expr.args:
        value = simplify(arg)
        if not isinstance(value, Rational):
            raise _StepCheckError(f"non-rational term: {arg}")
        terms.append(value)
    return terms


def _fraction_add_steps(expr) -> list[str]:
    terms = _as_signed_rationals(expr)
    if len(terms) != 2:
        raise _StepCheckError("only two-term sums get granular steps")
    r1, r2 = terms
    total = r1 + r2
    original = _terms_latex(terms)

    # Step 1 restates the sum; each later step applies one operation: rewrite
    # over a common denominator, add numerators, then reduce.
    states = [(original, total)]
    if r1.q == r2.q:
        d = r1.q
        combined_n = r1.p + r2.p
    else:
        d = lcm(r1.q, r2.q)
        c1, c2 = r1.p * (d // r1.q), r2.p * (d // r2.q)
        sign1 = "-" if c1 < 0 else ""
        sign2 = "- " if c2 < 0 else "+ "
        converted = f"{sign1}{_frac(abs(c1), d)} {sign2}{_frac(abs(c2), d)}"
        states.append((converted, Rational(c1, d) + Rational(c2, d)))
        combined_n = c1 + c2

    states.append((_frac(combined_n, d), Rational(combined_n, d)))
    if (total.p, total.q) != (combined_n, d):
        states.append((_rational_latex(total), total))
    return _running_steps(*states)


def _fraction_mul_steps(expr) -> list[str]:
    factors = []
    for arg in expr.args:
        if isinstance(arg, Pow) and arg.exp == -1:
            base = simplify(arg.base)
            if not isinstance(base, Rational):
                raise _StepCheckError("non-rational divisor")
            factors.append(("div", base))
        else:
            value = simplify(arg)
            if not isinstance(value, Rational):
                raise _StepCheckError(f"non-rational factor: {arg}")
            factors.append(("mul", value))
    if len(factors) != 2:
        raise _StepCheckError("only two-factor products get granular steps")

    (k1, f1), (k2, f2) = factors
    if k1 == "div":
        raise _StepCheckError("unsupported shape")
    if f1.q == 1 and f2.q == 1:
        raise _StepCheckError("integer product — generic step reads better")
    total = simplify(expr)

    # Step 1 restates the product/quotient; each later step applies one
    # operation: (for division) invert and multiply, cross-multiply, then reduce.
    if k2 == "div":
        states = [(rf"{_rational_latex(f1)} \div {_rational_latex(f2)}", total)]
        f2 = Rational(f2.q, f2.p)  # reciprocal (sign carried by p)
        states.append((rf"{_rational_latex(f1)} \times {_rational_latex(f2)}", f1 * f2))
    else:
        states = [(rf"{_rational_latex(f1)} \times {_rational_latex(f2)}", total)]

    raw_n = f1.p * f2.p
    raw_d = f1.q * f2.q
    cross = (
        rf"\frac{{{latex(Integer(f1.p))} \times {latex(Integer(f2.p))}}}"
        rf"{{{latex(Integer(f1.q))} \times {latex(Integer(f2.q))}}}"
    )
    states.append((cross, Rational(raw_n, raw_d)))
    states.append((_frac(raw_n, raw_d), Rational(raw_n, raw_d)))
    if (total.p, total.q) != (raw_n, raw_d):
        states.append((_rational_latex(total), total))
    return _running_steps(*states)


# ── Generic fallback ──────────────────────────────────────────────────────────

def _generic_steps(expression: str) -> list[str]:
    """Restatement + answer (two steps). Used when the structured builders
    cannot decompose the input further: step 1 restates the problem, step 2
    gives the result. Collapses to a single step only when the input is already
    fully simplified (nothing to derive)."""
    try:
        value = sympify(expression, locals={"Rational": Rational, "Eq": Eq}, rational=True)
        if isinstance(value, list):
            if len(value) != 1:
                return []
            value = value[0]
        if isinstance(value, Eq):
            free = value.free_symbols
            if len(free) != 1:
                return []
            x = free.pop()
            solutions = solve(value, x)
            if len(solutions) != 1:
                return []
            return [
                f"${latex(value.lhs)} = {latex(value.rhs)}$",
                f"${latex(x)} = {latex(solutions[0])}$",
            ]
        if value.free_symbols or not value.is_number:
            return []
        try:
            unevaluated = sympify(expression, locals={"Rational": Rational}, evaluate=False)
            display = latex(unevaluated)
        except Exception:
            display = latex(value)
        result = simplify(value)
        if display == latex(result):
            return [f"${display}$"]
        return _running_steps((display, value), (latex(result), result))
    except Exception:
        return []


def generate_solution_steps(sympy_expression: str) -> list[str]:
    """LaTeX derivation steps for a problem's sympy expression.

    Every emitted equality is symbolically verified; on any failure the
    generic single-step fallback is used instead.
    """
    try:
        expr = sympify(
            sympy_expression,
            locals={"Rational": Rational, "Eq": Eq},
            evaluate=False,
        )
        if isinstance(expr, Eq):
            return _linear_equation_steps(expr)
        if expr.is_Add:
            return _fraction_add_steps(expr)
        if expr.is_Mul:
            return _fraction_mul_steps(expr)
        raise _StepCheckError("unrecognized structure")
    except Exception:
        return _generic_steps(sympy_expression)
