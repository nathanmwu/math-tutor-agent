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

# Mastery EMA tuning. A correct answer moves the score LEARNING_RATE of the way
# toward 1.0; a wrong answer the same fraction toward 0.0. Once a correct answer
# brings the score within (1 - MASTERY_SNAP_THRESHOLD) of full it snaps to 1.0,
# so sustained success actually fills the bar instead of crawling asymptotically
# (…0.97, 0.976, 0.981…) and stranding the student a few percent short forever.
LEARNING_RATE = 0.3
MASTERY_SNAP_THRESHOLD = 0.95

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


def update_mastery_ema_score(old: float, is_correct: bool) -> float:
    """One EMA step toward 1.0 (correct) or 0.0 (wrong), with a snap-to-full near
    the top so a mastered skill reaches exactly 100% rather than approaching it
    asymptotically."""
    target = 1.0 if is_correct else 0.0
    new = old + LEARNING_RATE * (target - old)
    if is_correct and new >= MASTERY_SNAP_THRESHOLD:
        new = 1.0
    return max(0.0, min(1.0, new))


def difficulty_for_mastery(score: float) -> int:
    """Map a 0..1 mastery score onto the 1..5 difficulty band, so the problems a
    student is served track the mastery they've demonstrated — the same signal as
    the progress bar — instead of lagging behind it on a separate counter."""
    band = int(score * 5) + 1  # [0,.2)->1, [.2,.4)->2, … [.8,1]->5
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, band))


def update_mastery_ema(mastery: TopicMastery, is_correct: bool) -> TopicMastery:
    """Apply the mastery EMA and re-derive difficulty from the new score."""
    mastery.mastery_score = update_mastery_ema_score(mastery.mastery_score, is_correct)
    mastery.current_difficulty = difficulty_for_mastery(mastery.mastery_score)
    return mastery


def update_subtopic_mastery(sm: SubtopicMastery, is_correct: bool) -> SubtopicMastery:
    """Apply the mastery EMA and re-derive this subtopic's difficulty from it, so
    mastery and difficulty move together. The running correct-streak is retained
    for analytics but no longer gates difficulty."""
    sm.mastery_score = update_mastery_ema_score(sm.mastery_score, is_correct)
    sm.consecutive_correct = sm.consecutive_correct + 1 if is_correct else 0
    sm.current_difficulty = difficulty_for_mastery(sm.mastery_score)
    return sm


def problem_signature(problem_text: str) -> str:
    """Structural signature of a problem: its text with all digits removed.

    Captures problem *shape* rather than exact numbers, so "same structure,
    different numbers" repetition (which exact-match dedup misses) is detectable.
    """
    return re.sub(r"\d+", "#", problem_text).strip()
