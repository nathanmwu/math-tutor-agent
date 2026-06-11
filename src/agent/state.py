from __future__ import annotations

from typing import TypedDict


class TutorState(TypedDict):
    student_id: str
    current_topic: str
    current_subtopic: str
    current_difficulty: int
    current_problem: str
    sympy_answer: str
    student_answer: str
    evaluation: dict
    retrieved_chunks: list[str]
    feedback: str
    mastery: dict[str, float]
    session_history: list[dict]
