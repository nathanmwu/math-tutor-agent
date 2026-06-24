from __future__ import annotations

from typing import TypedDict


class TutorState(TypedDict):
    student_id: str
    current_topic: str
    current_subtopic: str
    current_difficulty: int
    current_problem: str
    sympy_expression: str       # the raw computation, e.g. "Eq(2*x + 5, 11)"
    sympy_answer: str           # SymPy-computed result; never shown before answering
    solution_steps: list[str]   # SymPy-verified LaTeX derivation, rendered verbatim
    student_answer: str
    evaluation: dict
    retrieved_chunks: list[str]
    feedback: str
    mastery: dict[str, float]
    # Per-subtopic mastery summary for the UI focus-areas panel, keyed by
    # subtopic_key(topic, subtopic) -> {"topic", "subtopic", "mastery_score"}.
    subtopic_mastery: dict[str, dict]
    session_history: list[dict]
