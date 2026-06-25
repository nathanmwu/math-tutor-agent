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
    TopicMastery,
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


# ── Difficulty tracks mastery ─────────────────────────────────────────────────

def test_difficulty_for_mastery_bands():
    assert store.difficulty_for_mastery(0.0) == 1
    assert store.difficulty_for_mastery(0.3) == 2
    assert store.difficulty_for_mastery(0.5) == 3
    assert store.difficulty_for_mastery(0.7) == 4
    assert store.difficulty_for_mastery(1.0) == store.MAX_DIFFICULTY


def test_difficulty_rises_as_mastery_rises():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    # A fresh subtopic (mastery 0) sits in the easiest band, and difficulty
    # climbs in lockstep with the EMA as the student succeeds.
    sm = store.update_subtopic_mastery(sm, True)  # 0 -> 0.3
    assert sm.current_difficulty == store.difficulty_for_mastery(sm.mastery_score) == 2
    for _ in range(10):
        sm = store.update_subtopic_mastery(sm, True)
    assert sm.mastery_score == 1.0                 # snaps to full
    assert sm.current_difficulty == store.MAX_DIFFICULTY


def test_difficulty_eases_after_misses():
    sm = SubtopicMastery(
        topic="algebra", subtopic="linear_equations", mastery_score=1.0, current_difficulty=5
    )
    sm = store.update_subtopic_mastery(sm, False)  # 1.0 -> 0.7 -> band 4
    assert sm.current_difficulty == 4
    assert sm.consecutive_correct == 0


def test_difficulty_clamps_at_bounds():
    low = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    for _ in range(10):
        low = store.update_subtopic_mastery(low, False)
    assert low.current_difficulty == store.MIN_DIFFICULTY

    high = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    for _ in range(15):
        high = store.update_subtopic_mastery(high, True)
    assert high.current_difficulty == store.MAX_DIFFICULTY


def test_ema_moves_toward_one_on_success():
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations")
    sm = store.update_subtopic_mastery(sm, True)
    assert sm.mastery_score == 0.3  # 0 + 0.3*(1-0)


def test_mastery_snaps_to_full_near_the_top():
    # The asymptote fix: once a correct answer brings the score within the snap
    # threshold, it reaches exactly 100% instead of crawling (…0.97, 0.976…).
    sm = SubtopicMastery(topic="algebra", subtopic="linear_equations", mastery_score=0.94)
    sm = store.update_subtopic_mastery(sm, True)  # 0.94 -> 0.958 -> snap 1.0
    assert sm.mastery_score == 1.0


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


def test_selection_difficulty_tracks_topic_mastery(monkeypatch, tmp_path):
    from src.agent import nodes

    monkeypatch.setattr(nodes, "STUDENT_STATE_DIR", tmp_path)

    state = StudentState.new("ramp_kid")
    # Attempt every subtopic so cold-start/exploration won't pick a fresh one.
    for topic, subs in nodes.SUBTOPICS.items():
        for sub in subs:
            state.subtopic_mastery[subtopic_key(topic, sub)] = SubtopicMastery(
                topic=topic, subtopic=sub, attempts=5
            )
    # Near-mastered on algebra: every algebra problem must be served at the top
    # difficulty band, regardless of which subtopic is selected — so the student
    # never drops back to "Easy" while the bar reads ~100%.
    state.topic_mastery["algebra"] = TopicMastery(topic="algebra", mastery_score=0.99)
    store.save_student(state, tmp_path)

    random.seed(1)
    saw_algebra = False
    for _ in range(200):
        result = nodes.select_topic_node({"student_id": "ramp_kid"})
        if result["current_topic"] == "algebra":
            saw_algebra = True
            assert result["current_difficulty"] == store.MAX_DIFFICULTY
    assert saw_algebra, "algebra was never selected in 200 draws"


# ── Deterministic on-topic fallback (no LLM) ─────────────────────────────────

def test_fallback_problem_is_on_topic_and_valid_for_every_subtopic():
    from src.agent import nodes

    random.seed(3)
    for topic, subs in nodes.SUBTOPICS.items():
        for sub in subs:
            for difficulty in (1, 3, 5):
                result = nodes._fallback_problem(sub, difficulty, set())
                assert result is not None, f"no fallback for {sub} @ d{difficulty}"
                # On-topic (never the old off-topic "2 + 2") and answerable.
                assert result["current_problem"]
                assert result["current_problem"] != "$2 + 2 =$"
                assert result["sympy_answer"] not in ("", None)
                assert result["solution_steps"]  # a derivation was produced
