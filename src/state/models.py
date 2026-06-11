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


class StudentState(BaseModel):
    student_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    topic_mastery: dict[str, TopicMastery] = Field(default_factory=dict)
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
