"""Tests for Phase 3 personalization: subtopic mastery, sustained difficulty
ramp, weakness-targeted selection, signature dedup, and backward-compat (no LLM).

Run with: .venv/bin/python -m pytest tests/test_personalization.py -v
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.prompts import build_generate_problem_prompt
from src.state import store
from src.state.models import (
    StudentState,
    SubtopicMastery,
    subtopic_key,
)


# ── Variety: prompt carries the variant hint and structural avoid-list ───────

def test_prompt_includes_variant_and_avoid_signatures():
    prompt = build_generate_problem_prompt(
        topic="fractions_ratios",
        subtopic="addition_subtraction",
        difficulty=4,
        recent_problems="none",
        variant_hint="three fractions with unlike denominators",
        avoid_signatures=r"$\frac{#}{#} + \frac{#}{#} =$",
    )
    assert "three fractions with unlike denominators" in prompt
    assert r"\frac{#}{#}" in prompt
    # difficulty guidance for level 4 is injected
    assert "two-digit" in prompt


# ── Sustained difficulty ramp ────────────────────────────────────────────────

def test_difficulty_does_not_rise_before_streak_threshold():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    # Two correct in a row — below RAMP_STREAK_THRESHOLD (3): difficulty unchanged.
    sm = store.update_subtopic_mastery(sm, True)
    sm = store.update_subtopic_mastery(sm, True)
    assert sm.current_difficulty == 2
    assert sm.consecutive_correct == 2


def test_difficulty_rises_at_streak_threshold_and_resets_streak():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    for _ in range(store.RAMP_STREAK_THRESHOLD):
        sm = store.update_subtopic_mastery(sm, True)
    assert sm.current_difficulty == 3       # stepped up once
    assert sm.consecutive_correct == 0       # streak re-earned from scratch


def test_difficulty_steps_down_and_resets_on_miss():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations", current_difficulty=3)
    sm.consecutive_correct = 2
    sm = store.update_subtopic_mastery(sm, False)
    assert sm.current_difficulty == 2
    assert sm.consecutive_correct == 0


def test_difficulty_clamps_at_bounds():
    # Never below MIN
    low = SubtopicMastery(topic="algebra", subtopic="linear_equations", current_difficulty=1)
    low = store.update_subtopic_mastery(low, False)
    assert low.current_difficulty == store.MIN_DIFFICULTY

    # Never above MAX even with a long streak
    high = SubtopicMastery(topic="algebra", subtopic="linear_equations", current_difficulty=5)
    for _ in range(10):
        high = store.update_subtopic_mastery(high, True)
    assert high.current_difficulty == store.MAX_DIFFICULTY


def test_ema_moves_toward_one_on_success():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    sm = store.update_subtopic_mastery(sm, True)
    assert sm.mastery_score == 0.2  # 0.8*0 + 0.2*1


# ── record_attempt updates both topic and subtopic ───────────────────────────

def test_record_attempt_updates_topic_and_subtopic():
    state = StudentState.new("unit_test_student")
    state = store.record_attempt(
        state=state,
        topic="algebra",
        subtopic="linear_equations",
        difficulty=2,
        problem_text="$2x + 5 = 11$",
        student_answer="3",
        is_correct=True,
        error_category=None,
        parse_error=False,
    )
    assert "algebra" in state.topic_mastery
    key = subtopic_key("algebra", "linear_equations")
    assert key in state.subtopic_mastery
    assert state.subtopic_mastery[key].attempts == 1
    assert state.subtopic_mastery[key].correct_attempts == 1


def test_record_attempt_tracks_subtopic_errors():
    state = StudentState.new("unit_test_student")
    state = store.record_attempt(
        state=state,
        topic="algebra",
        subtopic="linear_equations",
        difficulty=2,
        problem_text="$2x + 5 = 11$",
        student_answer="4",
        is_correct=False,
        error_category="sign_error",
        parse_error=False,
    )
    key = subtopic_key("algebra", "linear_equations")
    assert state.subtopic_mastery[key].error_pattern_counts["sign_error"] == 1


# ── Signature dedup ──────────────────────────────────────────────────────────

def test_problem_signature_strips_numbers():
    a = store.problem_signature(r"$\frac{1}{6} + \frac{2}{3} =$")
    b = store.problem_signature(r"$\frac{5}{8} + \frac{1}{4} =$")
    assert a == b  # same shape, different numbers


def test_recent_signatures_capped():
    state = StudentState.new("unit_test_student")
    for i in range(store.MAX_RECENT_SIGNATURES + 5):
        state = store.record_attempt(
            state=state,
            topic="algebra",
            subtopic="linear_equations",
            difficulty=2,
            problem_text=f"${i}x + {i} = {i}$",
            student_answer="0",
            is_correct=True,
            error_category=None,
            parse_error=False,
        )
    key = subtopic_key("algebra", "linear_equations")
    assert len(state.recent_signatures[key]) == store.MAX_RECENT_SIGNATURES


# ── Backward-compat: legacy JSON without Phase 3 fields ───────────────────────

def test_legacy_student_json_loads_with_defaults():
    legacy = """
    {
      "student_id": "legacy_kid",
      "created_at": "2026-06-01T10:00:00",
      "last_active": "2026-06-01T10:05:00",
      "topic_mastery": {
        "algebra": {
          "topic": "algebra",
          "mastery_score": 0.4,
          "current_difficulty": 3,
          "attempts": 5,
          "correct_attempts": 2,
          "error_pattern_counts": {"sign_error": 1},
          "last_updated": "2026-06-01T10:05:00"
        }
      },
      "attempt_history": []
    }
    """
    state = StudentState.model_validate_json(legacy)
    assert state.subtopic_mastery == {}
    assert state.recent_signatures == {}
    assert state.topic_mastery["algebra"].mastery_score == 0.4


# ── Weakness-targeted selection ──────────────────────────────────────────────

def test_selection_favors_weak_subtopic(monkeypatch, tmp_path):
    from src.agent import nodes

    monkeypatch.setattr(nodes, "STUDENT_STATE_DIR", tmp_path)

    # Build a student who has attempted every subtopic (no cold-start path), with
    # one clearly-weak subtopic (low mastery + errors) and the rest near-mastered.
    state = StudentState.new("weak_kid")
    for topic, subs in nodes.SUBTOPICS.items():
        for sub in subs:
            key = subtopic_key(topic, sub)
            weak = (topic, sub) == ("algebra", "proportions") or sub == "proportions"
            state.subtopic_mastery[key] = SubtopicMastery(
                topic=topic,
                subtopic=sub,
                mastery_score=0.1 if weak else 0.95,
                attempts=10,
                correct_attempts=1 if weak else 9,
                error_pattern_counts={"conceptual_error": 5} if weak else {},
            )
    store.save_student(state, tmp_path)

    random.seed(7)
    picks = []
    tutor_state = {"student_id": "weak_kid"}
    for _ in range(400):
        result = nodes.select_topic_node(tutor_state)
        picks.append(result["current_subtopic"])

    weak_count = picks.count("proportions")
    # Weak subtopic should be over-represented relative to a uniform 1/8 share,
    # while exploration still surfaces others (so it's not 100%).
    assert weak_count > len(picks) / 8
    assert len(set(picks)) > 1


def test_selection_difficulty_comes_from_subtopic(monkeypatch, tmp_path):
    from src.agent import nodes

    monkeypatch.setattr(nodes, "STUDENT_STATE_DIR", tmp_path)

    state = StudentState.new("ramp_kid")
    # Attempt every subtopic so cold-start/exploration won't pick a fresh one,
    # and give the target a known elevated difficulty.
    for topic, subs in nodes.SUBTOPICS.items():
        for sub in subs:
            state.subtopic_mastery[subtopic_key(topic, sub)] = SubtopicMastery(
                topic=topic, subtopic=sub, attempts=5, current_difficulty=2
            )
    state.subtopic_mastery[subtopic_key("algebra", "linear_equations")].current_difficulty = 5
    store.save_student(state, tmp_path)

    random.seed(1)
    for _ in range(200):
        result = nodes.select_topic_node({"student_id": "ramp_kid"})
        if result["current_subtopic"] == "linear_equations":
            assert result["current_difficulty"] == 5
            return
    raise AssertionError("linear_equations was never selected in 200 draws")
