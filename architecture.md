# System Architecture — Math Tutor Agent

## Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│  NiceGUI UI  (src/ui/app.py) — single process, event-driven         │
│                                                                     │
│  ┌──────────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ Problem (KaTeX)  │  │  Answer Input │  │  Mastery Dashboard  │  │
│  └──────────────────┘  └───────────────┘  └─────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Live operation feed (per-node labels during graph.stream)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ graph.stream(TutorState)  (in-process)
┌─────────────────────────────▼───────────────────────────────────────┐
│  LangGraph Agent  (src/agent/)                                      │
│                                                                     │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────────────┐   │
│  │ load_state  │──▶│ select_topic   │──▶│  generate_problem    │   │
│  └─────────────┘   └────────────────┘   └──────────┬───────────┘   │
│                                                     │ Ollama LLM    │
│                                                     ▼               │
│                                          ┌──────────────────────┐   │
│                                          │  present_problem     │   │
│                                          │  (UI pause point)    │   │
│                                          └──────────┬───────────┘   │
│                                                     │ student_answer│
│                                                     ▼               │
│                                          ┌──────────────────────┐   │
│                                          │  evaluate_answer     │   │
│                                          │  SymPy → bool        │   │
│                                          │  Ollama → error cat  │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│                                                     ▼               │
│                                          ┌──────────────────────┐   │
│                                          │ retrieve_explanation │   │
│                                          │  ChromaDB RAG        │   │
│                                          └──────────┬───────────┘   │
│                                                     │               │
│                                                     ▼               │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────────────┐   │
│  │ update_state│◀──│  adapt_next    │◀──│  generate_feedback   │   │
│  └──────┬──────┘   └───────┬────────┘   │  Ollama + chunks     │   │
│         │                 END           └──────────────────────┘   │
│         │  (UI shows feedback; student clicks Next to start again)  │
└─────────────────────────────────────────────────────────────────────┘
         │                          │
┌────────▼─────────┐    ┌──────────▼──────────┐
│  ChromaDB        │    │  Student State      │
│  Knowledge Base  │    │  JSON Store         │
│  data/chromadb/  │    │  data/students/     │
└──────────────────┘    └─────────────────────┘
         │
┌────────▼─────────┐
│  Ollama          │
│  llama3:8b       │
│  localhost:11434 │
└──────────────────┘
```

---

## Data Flow — One Complete Tutoring Turn

```
1. Student opens app
   └─ UI reads student_id from session or prompts for name
   └─ StudentState.load() reads data/students/{id}.json (or creates new)

2. Start of turn: load_state → select_topic → generate_problem
   └─ select_topic:
        mastery scores → sorted topic list
        70% chance: pick lowest-mastery topic
        30% chance: pick first unattempted topic (exploration)
        → current_topic, current_difficulty set in TutorState
   └─ generate_problem:
        Prompt: "Generate a difficulty-{n} problem on {topic}/{subtopic} in pure
                 mathematical notation with $...$ LaTeX. Return JSON."
        Ollama response → _parse_problem_json (repairs LaTeX-in-JSON escaping)
        → sympy_expression evaluated by SymPy itself (Eq solved, lists unwrapped)
        → retry up to 2x if malformed or symbolic
        sympy_answer stored in TutorState (never shown to student)
        → current_problem (LaTeX text) passed to UI

3. UI renders problem (KaTeX typesets the $...$ math), student types answer, submits

4. evaluate_answer
   └─ symbolic_check(student_input, sympy_answer):
        sympify(student_input) vs sympify(sympy_answer)
        simplify(a - b) == 0 → True/False/None
   └─ if False:
        categorize_error(problem, student_answer, correct):
          Ollama call → error_category string
   └─ evaluation = {is_correct, error_category, parse_error}

5. retrieve_explanation
   └─ ChromaDB query:
        where = {topic: current_topic, difficulty: {$lte: current_difficulty + 1}}
        if error_category: prefer chunks where misconception_tag matches
        n_results = 3 → list of chunk text strings

6. generate_feedback
   └─ Prompt: system context + problem + student_answer + correct_answer +
              is_correct + retrieved_chunks (injected)
        Ollama → "Result:" + "Explanation:" (numbered LaTeX derivation)
   └─ error_category is internal only: it filters retrieval (step 5) and
      accumulates in error_pattern_counts (step 7) but is never displayed
   └─ Feedback rendered in UI with KaTeX

7. update_state
   └─ AttemptRecord created and appended to StudentState.attempt_history
   └─ TopicMastery updated:
        mastery_score = 0.8 * old + 0.2 * (1.0 if correct else 0.0)
        current_difficulty += 1 if correct else -1  (clamped to [1,5])
        error_pattern_counts[error_category] += 1 if not correct
   └─ StudentState.save() → writes data/students/{id}.json

8. adapt_next
   └─ Sets next topic and difficulty in TutorState
   └─ Graph reaches END — state is preserved by MemorySaver checkpointer

9. UI enters reviewing phase
   └─ Displays the answered problem, student's answer, and full LLM explanation
   └─ Student clicks "Next problem →"
   └─ UI triggers a new turn: graph.stream({"student_answer": ""}, config)
   └─ Graph resumes from START: load_state → select_topic → generate_problem → pause
```

---

## LangGraph State

```python
class TutorState(TypedDict):
    # Identity
    student_id: str

    # Current turn
    current_topic: str
    current_subtopic: str
    current_difficulty: int          # 1–5
    current_problem: str             # human-readable problem text
    sympy_answer: str                # canonical answer, never exposed to UI
    student_answer: str              # populated when student submits

    # Evaluation results
    evaluation: dict                 # {is_correct, error_category, parse_error}

    # RAG
    retrieved_chunks: list[str]      # top-3 chunk bodies from ChromaDB

    # Output
    feedback: str                    # LLM-generated explanation

    # Persistent state (read at start, written at end of each turn)
    mastery: dict[str, float]        # {topic: 0.0–1.0}
    session_history: list[dict]      # lightweight turn log (not full AttemptRecord)
```

---

## Knowledge Base Structure

### Source files (data/knowledge_base/)

Each topic has a JSON file containing an array of chunks:

```json
[
  {
    "id": "algebra_linear_eq_concept_001",
    "text": "A linear equation is an equation where the variable appears with exponent 1. The standard form is ax + b = c. To solve, isolate x by performing the same inverse operation on both sides...",
    "metadata": {
      "topic": "algebra",
      "subtopic": "linear_equations",
      "content_type": "concept_explanation",
      "difficulty": 2,
      "misconception_tag": null
    }
  },
  {
    "id": "algebra_linear_eq_misconception_sign",
    "text": "Common error: when moving a term across the equals sign, students forget to change its sign. For example, in 2x + 3 = 7, subtracting 3 from both sides gives 2x = 4, not 2x = 10...",
    "metadata": {
      "topic": "algebra",
      "subtopic": "linear_equations",
      "content_type": "common_misconception",
      "difficulty": 2,
      "misconception_tag": "sign_error"
    }
  }
]
```

### ChromaDB collection schema

Collection name: `math_knowledge`

Each document is stored with:
- **document**: chunk `text` field
- **id**: chunk `id` field
- **metadata**: all fields from `metadata` object

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU-compatible)

---

## Student State File Format

```
data/students/{student_id}.json
```

```json
{
  "student_id": "alice_123",
  "created_at": "2026-06-10T10:00:00",
  "last_active": "2026-06-10T11:30:00",
  "topic_mastery": {
    "algebra": {
      "topic": "algebra",
      "mastery_score": 0.73,
      "current_difficulty": 3,
      "attempts": 15,
      "correct_attempts": 11,
      "error_pattern_counts": {
        "sign_error": 3,
        "wrong_operation": 1
      },
      "last_updated": "2026-06-10T11:30:00"
    }
  },
  "attempt_history": [
    {
      "timestamp": "2026-06-10T10:05:00",
      "topic": "algebra",
      "subtopic": "linear_equations",
      "difficulty": 2,
      "problem_text": "Solve for x: 2x + 3 = 7",
      "student_answer": "2",
      "is_correct": true,
      "error_category": null,
      "parse_error": false
    }
  ]
}
```

---

## Adaptive Logic

### Topic Selection

```
Topics ordered by mastery_score ascending.

With probability 0.70:
    Pick the topic with the lowest mastery_score (exploitation)

With probability 0.30:
    Pick the first topic with zero attempts (exploration)
    If all topics have been attempted, fall back to exploitation

On first session (no state):
    Start with topic[0] in curriculum order: fractions → ratios → algebra → geometry
```

### Difficulty Adaptation

```
Per topic, after each attempt:
    if is_correct:
        new_difficulty = min(current_difficulty + 1, 5)
    else:
        new_difficulty = max(current_difficulty - 1, 1)

New students start at difficulty = 2 for each topic.
```

### Mastery Score Update

```
EMA with α = 0.2:
    new_mastery = 0.8 * old_mastery + 0.2 * outcome
    where outcome = 1.0 if correct, 0.0 if wrong

Initial mastery = 0.0 for all topics.
A mastery score ≥ 0.8 after ≥ 5 attempts indicates strong performance on that topic.
```

---

## Technology Choices

| Component | Choice | Alternatives Considered |
|---|---|---|
| Agent framework | LangGraph | LangChain AgentExecutor (open-ended ReAct, less suitable for stateful turn-based loop) |
| Vector store | ChromaDB | Qdrant (used in `network-recommendation-engine`; ChromaDB simpler for local-only, no Docker) |
| LLM | Ollama / llama3:8b | OpenAI API (requires API key; Ollama keeps project fully local) |
| Answer evaluation | SymPy | LLM-based (non-deterministic; failure modes unacceptable in tutoring context) |
| Student state | Pydantic + JSON | SQLite (adequate for demo scale; JSON is human-readable for debugging) |
| Mastery algorithm | EMA (α=0.2) | Bayesian Knowledge Tracing (requires per-skill parameter estimation; overkill for v1) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | OpenAI text-embedding-3-small (requires API; MiniLM runs CPU-locally) |
| UI | NiceGUI | Streamlit (rerun-per-interaction model too slow; replaced); React + shadcn over FastAPI (rejected: two processes + Node toolchain conflicts with one-command setup) |
| Math rendering | KaTeX (CDN, auto-render + MutationObserver) | MathJax (heavier); markdown2 latex extra (MathML output, weaker typography) |

---

## Dependency Graph

```
app.py (NiceGUI UI)
    └── graph.py (LangGraph StateGraph)
            ├── nodes.py
            │       ├── prompts.py          (prompt templates)
            │       ├── retriever.py        (ChromaDB queries)
            │       └── store.py            (student state I/O)
            └── (Ollama via langchain-ollama)

retriever.py
    └── loader.py                           (ingest_kb.py calls this once)
            └── data/knowledge_base/*.json  (source content)

store.py
    └── models.py                           (Pydantic StudentState)
            └── data/students/*.json        (persisted state)
```

No circular dependencies. UI layer only touches `graph.py`. State I/O only touches `store.py`. LLM calls only in `nodes.py`.
