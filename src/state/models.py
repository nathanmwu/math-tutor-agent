from __future__ import annotations

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
    correct_attempts: int = 0
    error_pattern_counts: dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


class SubtopicMastery(BaseModel):
    """Per-subtopic mastery — the granularity at which personalization operates.

    Topic-level TopicMastery is the coarse roll-up the UI bars read; this is the
    fine-grained record that drives weakness-targeted selection and the
    sustained, per-subtopic difficulty ramp.
    """

    topic: str
    subtopic: str
    mastery_score: float = Field(0.0, ge=0.0, le=1.0)
    current_difficulty: int = 2
    attempts: int = 0
    correct_attempts: int = 0
    consecutive_correct: int = 0  # streak at the current difficulty; resets to 0 on any miss
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
