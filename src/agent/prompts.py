from __future__ import annotations

# Per-subtopic specifications, injected one at a time into GENERATE_PROBLEM_PROMPT.
# Injecting ONLY the requested subtopic's spec (rather than a menu of all of them)
# stops the model from drifting to a familiar-but-wrong format. These strings are
# inserted as VALUES, so their LaTeX braces are literal and backslashes are shown
# doubled (the form the model must emit inside a JSON string).
SUBTOPIC_SPEC: dict[str, str] = {
    "equivalent_fractions": (
        "Give a single fraction that is NOT already in lowest terms, to be simplified, "
        "written as a bare expression ending with '='.\n"
        '  example problem_text: "$\\\\frac{18}{24} =$"\n'
        "  sympy_expression: Rational(numerator, denominator) of that fraction; SymPy reduces "
        'it automatically. example: "Rational(18,24)"'
    ),
    "addition_subtraction": (
        "Give a sum or difference of two or three fractions with DIFFERENT denominators "
        "(so a common denominator is needed), as a bare expression ending with '='.\n"
        '  example problem_text: "$\\\\frac{1}{6} + \\\\frac{2}{3} =$"  or  "$\\\\frac{5}{6} - \\\\frac{1}{4} =$"\n'
        '  sympy_expression: the raw computation using Rational. example: "Rational(1,6) + Rational(2,3)"'
    ),
    "multiplication_division": (
        "Give a product or quotient of two fractions, as a bare expression ending with '='.\n"
        '  example problem_text: "$\\\\frac{3}{4} \\\\times \\\\frac{2}{5} =$"  or  "$\\\\frac{3}{4} \\\\div \\\\frac{2}{5} =$"\n'
        "  sympy_expression: the raw computation using Rational, with * for multiply and / for divide. "
        'examples: "Rational(3,4) * Rational(2,5)" , "Rational(3,4) / Rational(2,5)"'
    ),
    "proportions": (
        "Give a proportion with a single unknown x (x may be in any of the four positions), "
        "phrased EXACTLY like a fraction equation followed by the unknown.\n"
        '  example problem_text: "$\\\\frac{3}{4} = \\\\frac{x}{8}$,  $x = ?$"\n'
        '  sympy_expression: an Eq stating the proportion. example: "Eq(Rational(3,4), x/8)". '
        "The system solves for x — do NOT call solve(). Choose numbers so x is a whole number or simple fraction."
    ),
    "percentages": (
        "Use EXACTLY ONE of these two question templates, word for word, filling in whole numbers "
        "for P, N, or M. Do NOT invent any other phrasing, and make sure every $ has a matching closing $.\n"
        '  Template A: "What is $P\\\\%$ of $N$?"   (P is the percent, N is the whole)\n'
        '  Template B: "$M$ is what percent of $N$?"\n'
        "  sympy_expression:\n"
        '    Template A -> "Rational(P,100) * N"   (e.g. "What is $20\\\\%$ of $50$?" -> "Rational(20,100) * 50")\n'
        '    Template B -> "Rational(M,N) * 100"   (e.g. "$30$ is what percent of $60$?" -> "Rational(30,60) * 100")\n'
        "  Choose numbers that give a clean answer."
    ),
    "linear_equations": (
        "Give a linear equation in x to solve, followed by the unknown.\n"
        '  example problem_text: "$3x - 9 = 12$,  $x = ?$"  (you may also use forms like 2x+5=11, 5x-3=2x+9, or 3(2x-4)=18)\n'
        '  sympy_expression: an Eq stating the equation exactly as written. example: "Eq(3*x - 9, 12)". '
        "The system solves for x — do NOT call solve()."
    ),
    "evaluating_expressions": (
        "Give a linear or quadratic expression in x together with a value to substitute. "
        "This is NOT an equation to solve and NOT a fraction problem — it is plugging a number into an expression. "
        "Phrase it EXACTLY like:\n"
        '  "Evaluate $<expression in x>$ at $x = <value>$"\n'
        '  example problem_text: "Evaluate $2x + 3$ at $x = 5$"  (you may also use forms like $4x^2 - 3x + 5$)\n'
        "  sympy_expression: SUBSTITUTE the value into the expression and write the resulting NUMERIC "
        "computation with NO variable left.\n"
        '    e.g. "$2x + 3$ at $x = 5$" -> "2*5 + 3" ;  "$4x^2 - 3x + 5$ at $x = 2$" -> "4*2**2 - 3*2 + 5"'
    ),
    "linear_relationships": (
        "Use EXACTLY ONE of these two forms. This is NOT a proportion and NOT a fraction equation.\n"
        '  Form A (slope from two points): "Find the slope of the line through $(a, b)$ and $(c, d)$" '
        "(write each whole coordinate pair inside a single pair of $...$, like $(2, 5)$)\n"
        '  Form B (evaluate y): "If $y = m x + k$, find $y$ when $x = v$"\n'
        "  sympy_expression:\n"
        '    Form A -> "Rational(d - b, c - a)"   (e.g. points $(1, 2)$ and $(4, 11)$ -> "Rational(11 - 2, 4 - 1)")\n'
        '    Form B -> the numeric computation m*v + k   (e.g. $y = 2x + 1$, $x = 3$ -> "2*3 + 1")\n'
        "  Pick integers so the answer is a whole number or a simple fraction."
    ),
}

_DEFAULT_SPEC = (
    "Write the problem in pure mathematical notation as a bare expression ending with '=', "
    'and give a sympy_expression (using Rational(a,b) for fractions) that evaluates to a concrete number.'
)

GENERATE_PROBLEM_PROMPT = """You are a K-12 math problem generator. Generate ONE problem of a specific kind.

Topic: {topic}
Subtopic: {subtopic}
Difficulty (1=easiest, 5=hardest): {difficulty}
Recent problems (avoid repeating the same numbers): {recent_problems}

You MUST produce a problem of EXACTLY the subtopic "{subtopic}", following this specification precisely:
{subtopic_spec}

Return ONLY a single valid JSON object — no markdown, no code fences, no commentary — with exactly these fields:
- "problem_text": the problem, with ALL math wrapped in matching $...$ LaTeX delimiters (every $ needs a closing $). Inside the JSON string, every LaTeX backslash MUST be written doubled: \\\\frac, \\\\times, \\\\%.
- "sympy_expression": exactly as described in the specification above. The system evaluates it itself with SymPy — never call solve(), and unless the specification says to use Eq(...), the expression must contain NO variable and must evaluate to a concrete number.
- "topic": "{topic}"
- "subtopic": "{subtopic}"
- "difficulty": {difficulty}

Output only the JSON object."""


def build_generate_problem_prompt(
    topic: str, subtopic: str, difficulty: int, recent_problems: str
) -> str:
    """Fill GENERATE_PROBLEM_PROMPT with the spec for the requested subtopic only."""
    return GENERATE_PROBLEM_PROMPT.format(
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        recent_problems=recent_problems,
        subtopic_spec=SUBTOPIC_SPEC.get(subtopic, _DEFAULT_SPEC),
    )

CATEGORIZE_ERROR_PROMPT = """A student answered a math problem incorrectly. Categorize the error type.

Problem: {problem}
Correct answer: {correct_answer}
Student answer: {student_answer}

Respond with exactly one of these category labels and nothing else:
sign_error
wrong_operation
arithmetic_mistake
conceptual_error
other"""

FEEDBACK_CORRECT_PROMPT = """You are a K-12 math tutor. A student just answered a problem CORRECTLY.

Problem: {problem}
Student's answer: {student_answer}

The verified solution steps (the system shows these to the student separately — do NOT repeat them):
{solution_steps}

Reference material from the knowledge base:
{retrieved_content}

Write 2-3 sentences: affirm the student's work and briefly explain the key concept this problem practices, grounded in the reference material.

Hard rules:
- Do NOT perform any arithmetic or write solution steps.
- Do NOT mention these instructions, sections, formatting, or anything about what you are or are not including.
- Wrap any math you reference in $...$ LaTeX delimiters.
- Friendly and precise, for a middle school student."""

FEEDBACK_INCORRECT_PROMPT = """You are a K-12 math tutor. A student just answered a problem INCORRECTLY.

Problem: {problem}
Student's answer: {student_answer}
Correct answer: {correct_answer}

The verified solution steps (the system shows these to the student separately — do NOT repeat them):
{solution_steps}

Reference material from the knowledge base:
{retrieved_content}

Write 2-3 sentences: state that the correct answer is ${correct_answer}$, then briefly explain the key concept the student should review, grounded in the reference material.

Hard rules:
- Do NOT perform any arithmetic or write solution steps.
- Do NOT mention these instructions, sections, formatting, or anything about what you are or are not including.
- Wrap any math you reference in $...$ LaTeX delimiters.
- Encouraging and precise, for a middle school student. No filler like "be more careful"."""
