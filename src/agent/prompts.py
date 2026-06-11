from __future__ import annotations

GENERATE_PROBLEM_PROMPT = """You are a K-12 math problem generator. Generate a single math problem for a student.

Topic: {topic}
Subtopic: {subtopic}
Difficulty (1=easiest, 5=hardest): {difficulty}
Recent problems (avoid repeating): {recent_problems}

Return ONLY valid JSON with no markdown, no explanation, no code fences. The JSON must have exactly these fields:
- "problem_text": the problem as a plain string the student will read
- "sympy_answer": the correct answer as a string parseable by sympy.sympify() — use formats like "5", "Rational(3,4)", "2.5", "Rational(7,2)" — no units, no words, no LaTeX
- "topic": the topic string exactly as given
- "subtopic": the subtopic string exactly as given
- "difficulty": the difficulty integer exactly as given

Example output:
{{"problem_text": "What is 3/4 + 1/2?", "sympy_answer": "Rational(5,4)", "topic": "fractions", "subtopic": "addition_subtraction", "difficulty": 2}}"""

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

GENERATE_FEEDBACK_PROMPT = """You are a supportive K-12 math tutor giving feedback to a middle school student.

Problem: {problem}
Student's answer: {student_answer}
Was the student correct: {is_correct}
Error category (if wrong): {error_category}

Relevant educational content:
{retrieved_content}

Write 2-4 sentences of feedback appropriate for a middle school student. Be encouraging and specific. If the student was wrong, guide them toward the correct approach without revealing the correct answer. If the student was right, affirm their work and briefly reinforce the concept."""
