from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .models import (
    AttemptRecord,
    StudentState,
    SubtopicMastery,
    TopicMastery,
    subtopic_key,
)

# Difficulty ramp tuning. Difficulty rises only after sustained success at the
# current level (not per-attempt jitter) and steps down immediately on a miss.
RAMP_STREAK_THRESHOLD = 3
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5

# How many recent problem-shape signatures to retain per subtopic.
MAX_RECENT_SIGNATURES = 10


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

    # Topic-level roll-up (unchanged — the UI mastery bars read this).
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

    # Subtopic-level record — the granularity personalization operates at.
    key = subtopic_key(topic, subtopic)
    sm = state.subtopic_mastery.get(key) or SubtopicMastery(topic=topic, subtopic=subtopic)
    sm = update_subtopic_mastery(sm, is_correct)
    sm.attempts += 1
    if is_correct:
        sm.correct_attempts += 1
    elif error_category is not None:
        sm.error_pattern_counts[error_category] = (
            sm.error_pattern_counts.get(error_category, 0) + 1
        )
    sm.last_updated = datetime.now()
    state.subtopic_mastery[key] = sm

    # Persist a problem-shape signature so future generation can avoid repeats.
    sigs = state.recent_signatures.setdefault(key, [])
    sigs.append(problem_signature(problem_text))
    del sigs[:-MAX_RECENT_SIGNATURES]

    state.last_active = datetime.now()
    return state


def update_mastery_ema(mastery: TopicMastery, is_correct: bool) -> TopicMastery:
    """Apply EMA update: new_score = 0.8 * old + 0.2 * (1.0 if correct else 0.0).
    Also updates current_difficulty: +1 if correct, -1 if wrong, clamped [1,5]."""
    mastery.mastery_score = 0.8 * mastery.mastery_score + 0.2 * (1.0 if is_correct else 0.0)
    mastery.current_difficulty = max(1, min(5, mastery.current_difficulty + (1 if is_correct else -1)))
    return mastery


def update_subtopic_mastery(sm: SubtopicMastery, is_correct: bool) -> SubtopicMastery:
    """EMA score plus a streak-gated difficulty ramp.

    Difficulty rises only after RAMP_STREAK_THRESHOLD correct answers in a row at
    the current level (then the streak resets, so each step up must be re-earned),
    and steps down by one on any miss. This gives smooth, sustained progression on
    weak subtopics instead of per-attempt jitter.
    """
    sm.mastery_score = 0.8 * sm.mastery_score + 0.2 * (1.0 if is_correct else 0.0)
    if is_correct:
        sm.consecutive_correct += 1
        if sm.consecutive_correct >= RAMP_STREAK_THRESHOLD and sm.current_difficulty < MAX_DIFFICULTY:
            sm.current_difficulty += 1
            sm.consecutive_correct = 0
    else:
        sm.consecutive_correct = 0
        sm.current_difficulty = max(MIN_DIFFICULTY, sm.current_difficulty - 1)
    return sm


def problem_signature(problem_text: str) -> str:
    """Structural signature of a problem: its text with all digits removed.

    Captures problem *shape* rather than exact numbers, so "same structure,
    different numbers" repetition (which exact-match dedup misses) is detectable.
    """
    return re.sub(r"\d+", "#", problem_text).strip()
