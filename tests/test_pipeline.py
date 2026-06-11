"""
End-to-end pipeline test.
Runs the full graph for one turn and verifies the state at each checkpoint.
Prints every field so failures are easy to diagnose.

Run with: .venv/bin/python -m pytest tests/test_pipeline.py -v -s
"""
import sys
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import graph
from src.agent.state import TutorState


def _fresh_config():
    return {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}


def _bootstrap(config) -> TutorState:
    initial: TutorState = {
        "student_id": "test_student",
        "current_topic": "",
        "current_subtopic": "",
        "current_difficulty": 2,
        "current_problem": "",
        "sympy_answer": "",
        "student_answer": "",
        "evaluation": {},
        "retrieved_chunks": [],
        "feedback": "",
        "mastery": {},
        "session_history": [],
    }
    for chunk in graph.stream(initial, config=config, stream_mode="updates"):
        node = list(chunk.keys())[0]
        print(f"  node: {node}  ->  {list(chunk[node].keys())}")
    return graph.get_state(config).values


def test_bootstrap_produces_problem():
    config = _fresh_config()
    state = _bootstrap(config)

    print(f"\n--- State after bootstrap ---")
    print(f"  topic:        {state.get('current_topic')}")
    print(f"  subtopic:     {state.get('current_subtopic')}")
    print(f"  difficulty:   {state.get('current_difficulty')}")
    print(f"  problem_text: {state.get('current_problem')}")
    print(f"  sympy_answer: {state.get('sympy_answer')}")

    assert state.get("current_problem"), "No problem was generated"
    assert state.get("sympy_answer"), "sympy_answer is empty after bootstrap"


def test_sympy_answer_is_numeric():
    """sympy_answer must be a concrete number with no free symbols."""
    from sympy import sympify

    config = _fresh_config()
    state = _bootstrap(config)

    sa = state.get("sympy_answer", "")
    print(f"\n  sympy_answer raw: {sa!r}")

    expr = sympify(sa, rational=True)
    print(f"  sympified:        {expr}  free_symbols={expr.free_symbols}")

    assert not expr.free_symbols, (
        f"sympy_answer '{sa}' contains free symbols {expr.free_symbols} — "
        f"LLM is not computing a concrete answer"
    )
    assert expr.is_number, f"sympy_answer '{sa}' is not a number: {expr}"


def test_correct_answer_marked_correct():
    """Submit the exact sympy_answer and expect is_correct=True."""
    from sympy import sympify

    config = _fresh_config()
    state = _bootstrap(config)

    sympy_answer = state.get("sympy_answer", "")
    problem = state.get("current_problem", "")
    print(f"\n  problem:      {problem}")
    print(f"  sympy_answer: {sympy_answer}")

    # Submit the exact stored answer as the student's answer
    student_answer = str(sympify(sympy_answer))
    print(f"  submitting:   {student_answer}")

    graph.update_state(config, {"student_answer": student_answer})
    for chunk in graph.stream(None, config=config, stream_mode="updates"):
        node = list(chunk.keys())[0]
        values = chunk[node]
        print(f"  node: {node}  ->  {values}")

    final = graph.get_state(config).values
    evaluation = final.get("evaluation", {})
    print(f"\n  evaluation: {evaluation}")
    print(f"  feedback:   {final.get('feedback', '')[:200]}")

    assert evaluation.get("is_correct") is True, (
        f"Expected correct but got: {evaluation}  "
        f"(problem={problem!r}, sympy_answer={sympy_answer!r}, submitted={student_answer!r})"
    )


def test_wrong_answer_marked_incorrect():
    """Submit a clearly wrong answer and expect is_correct=False."""
    config = _fresh_config()
    state = _bootstrap(config)

    problem = state.get("current_problem", "")
    sympy_answer = state.get("sympy_answer", "")
    print(f"\n  problem:      {problem}")
    print(f"  sympy_answer: {sympy_answer}")

    # Submit 99999 — extremely unlikely to be correct
    graph.update_state(config, {"student_answer": "99999"})
    for chunk in graph.stream(None, config=config, stream_mode="updates"):
        pass

    final = graph.get_state(config).values
    evaluation = final.get("evaluation", {})
    print(f"  evaluation: {evaluation}")

    assert evaluation.get("is_correct") is False
