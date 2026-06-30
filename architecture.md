# System Architecture — Math Tutor Agent

## Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  NiceGUI UI  (src/ui.py) — single process, event-driven             │
│  Problem (KaTeX) · Answer input · Mastery dashboard · Live op feed   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ graph.stream(TutorState)  (in-process)
┌─────────────────────────────▼───────────────────────────────────────┐
│  LangGraph StateGraph  (src/pipeline.py)                            │
│                                                                     │
│   setup ─▶ generate_problem ─▶ [PAUSE] ─▶ evaluate_answer ─▶        │
│                                              explain ─▶ update ─▶ END │
│                                                                     │
│   setup            load student, pick weakest subtopic, set diff     │
│   generate_problem Ollama → LaTeX problem + SymPy expr (verified)    │
│   [PAUSE]          interrupt_before evaluate_answer; UI submits      │
│   evaluate_answer  SymPy check (+ Ollama error category if wrong)    │
│   explain          ChromaDB RAG + Ollama concept note               │
│   update_state     EMA mastery + difficulty, write JSON             │
└──────────────┬───────────────────────────────┬─────────────────────┘
               │                               │
      ┌────────▼─────────┐          ┌──────────▼──────────┐
      │  ChromaDB (RAG)  │          │  Student State      │
      │  data/chromadb/  │          │  data/students/*.json│
      │  src/knowledge.py│          │  src/student.py      │
      └────────┬─────────┘          └─────────────────────┘
               │
      ┌────────▼─────────┐
      │  Ollama          │
      │  llama3.1:8b      │
      └──────────────────┘
```

The pipeline is five nodes with a single human-in-the-loop pause. Next-turn
difficulty is recomputed in `setup` from the freshly-saved mastery, so there is
no separate "adapt" node.

---

## Data Flow — One Complete Turn

```
1. Student opens app; UI reads/asks for a name → student_id.

2. setup
   └─ StudentState.load_or_new(student_id) reads data/students/{id}.json
   └─ Picks a (topic, subtopic): cold-start a never-attempted one (60%), else a
      uniform exploration pick (25%), else weighted-random by weakness priority.
   └─ Difficulty = difficulty_for_mastery(topic mastery)  → band in [1,5]
   └─ Exposes mastery + subtopic_mastery to the UI sidebar.

3. generate_problem
   └─ Ollama → JSON {problem_text (LaTeX), sympy_expression, ...}
   └─ _parse_problem_json repairs LaTeX-in-JSON escaping
   └─ SymPy evaluates the expression itself (Eq solved, lists unwrapped); retries
      up to 3x, then a deterministic on-topic fallback. The LLM's arithmetic is
      never trusted. sympy_answer is stored (never shown). solution_steps are
      precomputed by generate_solution_steps() — a SymPy-verified derivation.

4. [PAUSE] UI renders the problem (KaTeX), student submits an answer.

5. evaluate_answer
   └─ symbolic_check(student_input, sympy_answer): simplify(a - b) == 0 → T/F/None
   └─ equivalent_fractions also requires lowest terms (gcd check)
   └─ if wrong: Ollama categorizes the error (internal only)
   └─ evaluation = {is_correct, error_category, parse_error}

6. explain
   └─ ChromaDB query: problem text as the semantic query, filtered by
      topic/subtopic/difficulty (+ misconception_tag when an error category exists)
   └─ Ollama writes a 2-3 sentence concept note grounded in the retrieved chunks.
      The derivation is solution_steps (SymPy, from step 3) — the LLM never writes
      math. error_category sharpens retrieval and accrues in statistics, never shown.
   └─ UI shows: result banner + concept note + verified steps (KaTeX).

7. update_state
   └─ Append an AttemptRecord; EMA-update topic + subtopic mastery; persist a
      problem-shape signature (avoid-list); StudentState.save() writes JSON.

8. Graph reaches END (state preserved by the MemorySaver checkpointer). UI enters
   the reviewing phase; "Next problem" starts a fresh turn at setup.
```

---

## LangGraph State (`src/pipeline.py`)

```python
class TutorState(TypedDict):
    student_id: str
    current_topic: str
    current_subtopic: str
    current_difficulty: int          # 1–5
    current_problem: str             # LaTeX problem text
    sympy_expression: str            # raw computation, e.g. "Eq(2*x + 5, 11)"
    sympy_answer: str                # canonical answer, never exposed before answering
    solution_steps: list[str]        # SymPy-verified LaTeX derivation
    student_answer: str
    evaluation: dict                 # {is_correct, error_category, parse_error}
    feedback: str                    # LLM concept note (2-3 sentences)
    mastery: dict[str, float]        # {topic: 0.0–1.0}, for the UI bars
    subtopic_mastery: dict[str, dict]# per-subtopic summary, for the focus-areas panel
    session_history: list[dict]      # lightweight turn log
```

---

## Student State File (`data/students/{id}.json`, `src/student.py`)

```json
{
  "student_id": "alice",
  "created_at": "2026-06-10T10:00:00",
  "last_active": "2026-06-10T11:30:00",
  "topic_mastery": {
    "algebra": {
      "topic": "algebra", "mastery_score": 0.73, "current_difficulty": 3,
      "attempts": 15, "error_pattern_counts": {"sign_error": 3},
      "last_updated": "2026-06-10T11:30:00"
    }
  },
  "subtopic_mastery": {
    "algebra::linear_equations": {
      "topic": "algebra", "subtopic": "linear_equations", "mastery_score": 0.8,
      "current_difficulty": 4, "attempts": 9, "error_pattern_counts": {},
      "last_updated": "2026-06-10T11:30:00"
    }
  },
  "recent_signatures": {"algebra::linear_equations": ["$#x + # = #$, $x = ?$"]},
  "attempt_history": [ /* AttemptRecord entries */ ]
}
```

`topic_mastery` drives the UI bars; `subtopic_mastery` is the granularity
personalization operates at; `recent_signatures` is a per-subtopic avoid-list so
problems vary in shape across sessions. Pydantic ignores unknown fields, so older
state files load cleanly.

---

## Adaptive Logic

**Subtopic selection** (`setup_node`) — over all (topic, subtopic) pairs:
```
if a never-attempted subtopic exists and random() < 0.60:  pick one (cold start)
elif random() < 0.25:                                       pick uniformly (explore)
else:                                                       weighted-random by priority
    priority = max(0.05, (1 - mastery_score) + 0.5 * error_rate)
```

**Difficulty** tracks demonstrated topic mastery (the same signal as the bar):
```
difficulty_for_mastery(score) = clamp(int(score * 5) + 1, 1, 5)
```

**Mastery EMA** (per topic and per subtopic), updated after each attempt:
```
new = 0.7 * old + 0.3 * (1.0 if correct else 0.0)
# snaps to exactly 1.0 once a correct answer brings it within 0.05 of full
```

---

## Knowledge Base (`data/knowledge_base/*.json` → ChromaDB)

Collection `math_knowledge`; embeddings via `sentence-transformers/all-MiniLM-L6-v2`
(384-dim, CPU). Each chunk:

```json
{
  "id": "algebra_linear_eq_misconception_sign",
  "text": "Common error: when moving a term across the equals sign ...",
  "metadata": {
    "topic": "algebra", "subtopic": "linear_equations",
    "content_type": "common_misconception", "difficulty": 2,
    "misconception_tag": "sign_error"
  }
}
```

`content_type` is one of `concept_explanation`, `worked_example`,
`common_misconception`. `misconception_tag` matches the error categories from
answer evaluation, so a wrong answer can retrieve the matching misconception.

---

## Module Dependencies

```
ui.py ──▶ pipeline.py ──┬──▶ prompts.py
                        ├──▶ solution_steps.py
                        ├──▶ knowledge.py ──▶ data/knowledge_base/*.json
                        ├──▶ student.py   ──▶ data/students/*.json
                        └──▶ Ollama (langchain-ollama)

scripts/ingest_kb.py ──▶ knowledge.py   (one-time KB ingestion)
```

No circular dependencies. The UI imports only the compiled `graph` (and
`TutorState`) from `pipeline.py`; all LLM calls live in `pipeline.py`; all state
I/O lives in `student.py`.
```
