from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .models import AttemptRecord, StudentState, TopicMastery


def load_student(student_id: str, base_dir: Path) -> StudentState:
    """Load existing state or create new if none exists."""
    try:
        return StudentState.load(student_id, base_dir)
    except FileNotFoundError:
        return StudentState.new(student_id)


def save_student(state: StudentState, base_dir: Path) -> None:
    """Persist state to disk."""
    state.save(base_dir)


def record_attempt(
    state: StudentState,
    topic: str,
    subtopic: str,
    difficulty: int,
    problem_text: str,
    student_answer: str,
    is_correct: bool,
    error_category: str | None,
    parse_error: bool,
) -> StudentState:
    """Append an AttemptRecord and update TopicMastery via EMA. Return updated state (does not save to disk)."""
    record = AttemptRecord(
        timestamp=datetime.now(),
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        problem_text=problem_text,
        student_answer=student_answer,
        is_correct=is_correct,
        error_category=error_category,
        parse_error=parse_error,
    )
    state.attempt_history.append(record)

    mastery = state.topic_mastery.get(topic) or TopicMastery(topic=topic)
    mastery = update_mastery_ema(mastery, is_correct)
    mastery.attempts += 1
    if is_correct:
        mastery.correct_attempts += 1
    elif error_category is not None:
        mastery.error_pattern_counts[error_category] = (
            mastery.error_pattern_counts.get(error_category, 0) + 1
        )
    mastery.last_updated = datetime.now()

    state.topic_mastery[topic] = mastery
    state.last_active = datetime.now()
    return state


def update_mastery_ema(mastery: TopicMastery, is_correct: bool) -> TopicMastery:
    """Apply EMA update: new_score = 0.8 * old + 0.2 * (1.0 if correct else 0.0).
    Also updates current_difficulty: +1 if correct, -1 if wrong, clamped [1,5]."""
    mastery.mastery_score = 0.8 * mastery.mastery_score + 0.2 * (1.0 if is_correct else 0.0)
    mastery.current_difficulty = max(1, min(5, mastery.current_difficulty + (1 if is_correct else -1)))
    return mastery
