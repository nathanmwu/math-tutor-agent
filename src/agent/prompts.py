from __future__ import annotations

GENERATE_PROBLEM_PROMPT = """You are a K-12 math problem generator. Generate a single math problem for a student.

Topic: {topic}
Subtopic: {subtopic}
Difficulty (1=easiest, 5=hardest): {difficulty}
Recent problems (avoid repeating): {recent_problems}

The problem must be written PURELY MATHEMATICALLY — no word problems, no story contexts, no named people or objects. Just mathematical notation.

Return ONLY valid JSON with no markdown, no explanation, no code fences. The JSON must have exactly these fields:
- "problem_text": the problem in pure mathematical notation, with ALL math wrapped in $...$ LaTeX delimiters. Formats:
  - Arithmetic / fractions / numeric: the bare expression ending with =, e.g. "$\\\\frac{{1}}{{6}} + \\\\frac{{2}}{{3}} =$"
  - Equations: the equation, then the unknown, e.g. "$3x - 9 = 12$,  $x = ?$"
  IMPORTANT: inside the JSON string every LaTeX backslash must be DOUBLED (write \\\\frac, \\\\times, \\\\sqrt).
- "sympy_expression": a valid Python/SymPy expression. Rules by topic:
  - Fractions / arithmetic: write the raw computation using Rational(a,b). Examples: "Rational(1,6) + Rational(2,3)", "Rational(3,4) * Rational(2,5)"
  - Algebra (equation to solve): use Eq(lhs, rhs) to express the equation exactly as written. Examples: "Eq(2*x + 5, 11)", "Eq(3*x - 9, 12)". The system will call solve() itself — do NOT call solve() yourself.
  - Geometry / other numeric: plain arithmetic. Examples: "3 * 4", "Rational(1,2) * 6 * 4"
- "topic": the topic string exactly as given
- "subtopic": the subtopic string exactly as given
- "difficulty": the difficulty integer exactly as given

Example outputs:
{{"problem_text": "$\\\\frac{{1}}{{6}} + \\\\frac{{2}}{{3}} =$", "sympy_expression": "Rational(1,6) + Rational(2,3)", "topic": "fractions", "subtopic": "addition_subtraction", "difficulty": 2}}
{{"problem_text": "$3x - 9 = 12$,  $x = ?$", "sympy_expression": "Eq(3*x - 9, 12)", "topic": "algebra", "subtopic": "linear_equations", "difficulty": 2}}"""

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

GENERATE_FEEDBACK_PROMPT = """You are a K-12 math tutor. A student just answered a problem. Write a clear, educational response.

Problem: {problem}
Student's answer: {student_answer}
Correct answer: {correct_answer}
Was correct: {is_correct}

Reference material from the knowledge base:
{retrieved_content}

Write your response in exactly this two-section structure:

Result: One sentence — state whether the student was right or wrong, and what the correct answer is.

Explanation: A step-by-step derivation of the solution, written the way it would appear in a mathematics paper. Number each step. Put each step on its own line. Every mathematical expression must be wrapped in $...$ LaTeX delimiters (e.g. $\\frac{{1}}{{6}} + \\frac{{2}}{{3}} = \\frac{{1}}{{6}} + \\frac{{4}}{{6}} = \\frac{{5}}{{6}}$). Be concrete and specific to the numbers in this problem. Use the reference material to ground the explanation. No filler advice, no "be more careful" — just the mathematics.

Write for a middle school student: clear and precise, but friendly."""
