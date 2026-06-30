from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class AttemptRecord(BaseModel):
    timestamp: datetime
    topic: str
    subtopic: str
    difficulty: int
    problem_text: str
    student_answer: str
    is_correct: bool
    error_category: str | None
    parse_error: bool


class TopicMastery(BaseModel):
    topic: str
    mastery_score: float = Field(0.0, ge=0.0, le=1.0)
    current_difficulty: int = 2
    attempts: int = 0
    error_pattern_counts: dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


class SubtopicMastery(BaseModel):
    """Per-subtopic mastery — the granularity at which personalization operates.

    Topic-level TopicMastery is the coarse roll-up the UI bars read; this is the
    fine-grained record that drives weakness-targeted selection and the
    per-subtopic difficulty ramp.
    """

    topic: str
    subtopic: str
    mastery_score: float = Field(0.0, ge=0.0, le=1.0)
    current_difficulty: int = 2
    attempts: int = 0
    error_pattern_counts: dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


def subtopic_key(topic: str, subtopic: str) -> str:
    """Composite key for SubtopicMastery / recent_signatures dicts."""
    return f"{topic}::{subtopic}"


class StudentState(BaseModel):
    student_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    topic_mastery: dict[str, TopicMastery] = Field(default_factory=dict)
    # Keyed by subtopic_key(topic, subtopic). default_factory keeps pre-Phase-3
    # student JSON loadable — these populate on the next recorded attempt.
    subtopic_mastery: dict[str, SubtopicMastery] = Field(default_factory=dict)
    # Last ~10 problem-shape signatures per subtopic key, used as a generation
    # avoid-list so problems vary in structure across sessions.
    recent_signatures: dict[str, list[str]] = Field(default_factory=dict)
    attempt_history: list[AttemptRecord] = Field(default_factory=list)

    def save(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / f"{self.student_id}.json").write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, student_id: str, base_dir: Path) -> StudentState:
        path = base_dir / f"{student_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No state file found for student '{student_id}' at {path}")
        return cls.model_validate_json(path.read_text())

    @classmethod
    def new(cls, student_id: str) -> StudentState:
        return cls(student_id=student_id)

    @classmethod
    def load_or_new(cls, student_id: str, base_dir: Path) -> StudentState:
        """Load existing state, or start a fresh one if no file exists yet."""
        try:
            return cls.load(student_id, base_dir)
        except FileNotFoundError:
            return cls.new(student_id)


# ── Mastery tracking ──────────────────────────────────────────────────────────
# A correct answer moves the score LEARNING_RATE of the way toward 1.0; a wrong
# answer the same fraction toward 0.0. Once a correct answer brings the score
# within (1 - MASTERY_SNAP_THRESHOLD) of full it snaps to 1.0, so sustained
# success fills the bar instead of crawling asymptotically (…0.97, 0.976…).
LEARNING_RATE = 0.3
MASTERY_SNAP_THRESHOLD = 0.95

MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 5

# How many recent problem-shape signatures to retain per subtopic.
MAX_RECENT_SIGNATURES = 10


def update_mastery_ema_score(old: float, is_correct: bool) -> float:
    """One EMA step toward 1.0 (correct) or 0.0 (wrong), snapping to exactly 1.0
    near the top so a mastered skill reaches 100% rather than approaching it."""
    target = 1.0 if is_correct else 0.0
    new = old + LEARNING_RATE * (target - old)
    if is_correct and new >= MASTERY_SNAP_THRESHOLD:
        new = 1.0
    return max(0.0, min(1.0, new))


def difficulty_for_mastery(score: float) -> int:
    """Map a 0..1 mastery score onto the 1..5 difficulty band."""
    band = int(score * 5) + 1  # [0,.2)->1, [.2,.4)->2, … [.8,1]->5
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, band))


def update_mastery_ema(mastery: TopicMastery, is_correct: bool) -> TopicMastery:
    """Apply the mastery EMA and re-derive difficulty from the new score."""
    mastery.mastery_score = update_mastery_ema_score(mastery.mastery_score, is_correct)
    mastery.current_difficulty = difficulty_for_mastery(mastery.mastery_score)
    return mastery


def update_subtopic_mastery(sm: SubtopicMastery, is_correct: bool) -> SubtopicMastery:
    """Apply the mastery EMA and re-derive this subtopic's difficulty, so mastery
    and difficulty move together."""
    sm.mastery_score = update_mastery_ema_score(sm.mastery_score, is_correct)
    sm.current_difficulty = difficulty_for_mastery(sm.mastery_score)
    return sm


def problem_signature(problem_text: str) -> str:
    """Structural signature of a problem: its text with all digits removed, so
    'same structure, different numbers' repetition is detectable."""
    return re.sub(r"\d+", "#", problem_text).strip()


def _bump(counts: dict[str, int], category: str | None) -> None:
    """Increment the error-pattern tally for a category (no-op when None)."""
    if category:
        counts[category] = counts.get(category, 0) + 1


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
    """Append an AttemptRecord and update topic + subtopic mastery via EMA.
    Returns the updated state (does not save to disk)."""
    state.attempt_history.append(
        AttemptRecord(
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
    )

    # Topic-level roll-up — the UI mastery bars read this.
    mastery = update_mastery_ema(state.topic_mastery.get(topic) or TopicMastery(topic=topic), is_correct)
    mastery.attempts += 1
    _bump(mastery.error_pattern_counts, None if is_correct else error_category)
    mastery.last_updated = datetime.now()
    state.topic_mastery[topic] = mastery

    # Subtopic-level record — the granularity personalization operates at.
    key = subtopic_key(topic, subtopic)
    sm = update_subtopic_mastery(
        state.subtopic_mastery.get(key) or SubtopicMastery(topic=topic, subtopic=subtopic), is_correct
    )
    sm.attempts += 1
    _bump(sm.error_pattern_counts, None if is_correct else error_category)
    sm.last_updated = datetime.now()
    state.subtopic_mastery[key] = sm

    # Persist a problem-shape signature so future generation can avoid repeats.
    sigs = state.recent_signatures.setdefault(key, [])
    sigs.append(problem_signature(problem_text))
    del sigs[:-MAX_RECENT_SIGNATURES]

    state.last_active = datetime.now()
    return state
