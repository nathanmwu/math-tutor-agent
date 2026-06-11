from __future__ import annotations

GENERATE_PROBLEM_PROMPT = """You are a K-12 math problem generator. Generate a single math problem for a student.

Topic: {topic}
Subtopic: {subtopic}
Difficulty (1=easiest, 5=hardest): {difficulty}
Recent problems (avoid repeating): {recent_problems}

Return ONLY valid JSON with no markdown, no explanation, no code fences. The JSON must have exactly these fields:
- "problem_text": the problem as a plain string the student will read
- "sympy_expression": a valid Python/SymPy expression that evaluates to the correct numeric answer. Never pre-compute it yourself — write the raw computation so the system can evaluate it. Rules by topic:
  - Fractions / arithmetic: use Rational(a,b). Examples: "Rational(1,6) + Rational(2,3)", "Rational(3,4) * Rational(2,5)"
  - Algebra (solve for x): use solve(). Examples: "solve(2*x + 5 - 11, x)", "solve(3*x - 9, x)"
  - Geometry / other numeric: plain arithmetic. Examples: "3 * 4", "Rational(1,2) * 6 * 4"
- "topic": the topic string exactly as given
- "subtopic": the subtopic string exactly as given
- "difficulty": the difficulty integer exactly as given

Example outputs:
{{"problem_text": "What is 1/6 + 2/3?", "sympy_expression": "Rational(1,6) + Rational(2,3)", "topic": "fractions", "subtopic": "addition_subtraction", "difficulty": 2}}
{{"problem_text": "Solve for x: 2x + 5 = 11", "sympy_expression": "solve(2*x + 5 - 11, x)", "topic": "algebra", "subtopic": "linear_equations", "difficulty": 2}}"""

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
Error type (if wrong): {error_category}

Reference material from the knowledge base:
{retrieved_content}

Write your response in the following structure. Use plain text, no markdown headers or bullet symbols.

Result: One sentence — state whether the student was right or wrong, and what the correct answer is.

Explanation: Using the reference material above, walk through how to solve this specific problem step by step. Be concrete and specific to the numbers in this problem, not generic. Write 3-5 sentences.

If the student was WRONG, add this section:
What went wrong: Based on the error type "{error_category}", explain in 1-2 sentences exactly what mistake the student likely made and how to avoid it next time.

Be encouraging. Write for a middle school student."""
