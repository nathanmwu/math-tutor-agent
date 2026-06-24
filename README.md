# Math Tutor Agent

An adaptive K-12 **arithmetic and algebra** tutoring system that generates problems, evaluates answers with a symbolic math solver, and retrieves targeted explanations from a curated knowledge base. It covers two topics — **Fractions & Ratios** (equivalent fractions, the four operations, proportions, percentages) and **Algebra** (linear equations, evaluating expressions, linear relationships). Difficulty and topic selection adjust automatically based on each student's performance history.

Built as a portfolio project demonstrating core AI engineering techniques: retrieval-augmented generation (RAG), stateful agentic loops with LangGraph, deterministic answer evaluation with SymPy, and persistent student modeling.

---

## What It Does

1. **Generates a problem** appropriate for the student's current level on their weakest topic — written in pure mathematical notation and rendered with LaTeX (e.g. $\frac{1}{6} + \frac{2}{3} =$, no word problems)
2. **Student submits an answer** — any equivalent form is accepted (`1/2`, `0.5`, `2/4` are all correct for the same answer)
3. **Evaluates the answer symbolically** — right/wrong is determined by math, not an AI guess
4. **Retrieves an explanation** from a knowledge base of concept notes and worked examples
5. **Explains the solution** — a step-by-step LaTeX derivation grounded in the retrieved content, shown after every answer
6. **Updates the student's mastery profile** and adjusts difficulty for the next problem
7. **Loops** — presents the next problem, now at an adjusted difficulty on the most appropriate topic

All computation runs locally. No API keys required.

---

## Quick Start

**Prerequisites**: Python 3.12, [Ollama](https://ollama.com) installed and running

```bash
# Pull the language model
ollama pull llama3.1:8b

# Clone and set up
git clone https://github.com/nathanmwu/math-tutor-agent.git
cd math-tutor-agent
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Populate the knowledge base (run once)
python scripts/ingest_kb.py

# Launch
python src/ui/app.py
```

Open http://localhost:8501, enter your name, and start practicing.

---

## Project Structure

```
math-tutor-agent/
├── src/
│   ├── agent/
│   │   ├── graph.py        # LangGraph StateGraph — defines the tutoring loop
│   │   ├── nodes.py        # All node functions + symbolic_check()
│   │   ├── solution_steps.py  # SymPy-verified derivations (LLM never writes math)
│   │   ├── prompts.py      # All LLM prompt templates
│   │   └── state.py        # TutorState TypedDict
│   ├── knowledge/
│   │   ├── loader.py       # Ingests knowledge base JSON → ChromaDB
│   │   └── retriever.py    # ChromaDB query with topic/difficulty filters
│   ├── state/
│   │   ├── models.py       # Pydantic models: StudentState, TopicMastery, AttemptRecord
│   │   └── store.py        # load_student(), save_student(), record_attempt()
│   └── ui/
│       └── app.py          # NiceGUI app (problem display, answer input, mastery dashboard)
├── data/
│   ├── knowledge_base/     # Source JSON chunks (version-controlled)
│   │   ├── fractions_ratios.json
│   │   └── algebra.json
│   ├── chromadb/           # ChromaDB vector store (generated, gitignored)
│   └── students/           # Per-student state files (generated, gitignored)
├── tests/
│   ├── test_symbolic_check.py      # Unit tests for answer evaluation logic
│   ├── test_answer_evaluation.py   # Integration tests for evaluation pipeline
│   ├── test_problem_generation.py  # LLM output validation tests
│   └── test_pipeline.py            # End-to-end graph tests
└── scripts/
    └── ingest_kb.py        # One-time knowledge base ingestion script
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  NiceGUI UI  (src/ui/app.py)                    │
│  LaTeX problems (KaTeX) · Answer input ·        │
│  Mastery bars · Live operation feed             │
└──────────────────────┬──────────────────────────┘
                       │ graph.stream(TutorState)
┌──────────────────────▼──────────────────────────┐
│  LangGraph StateGraph  (src/agent/graph.py)     │
│                                                 │
│  load_state → select_topic → generate_problem  │
│                                    ↓            │
│                         [PAUSE — student reads] │
│                                    ↓            │
│                           evaluate_answer       │
│                                    ↓            │
│                         retrieve_explanation    │
│                                    ↓            │
│                          generate_feedback      │
│                                    ↓            │
│                    update_state → adapt_next → END
└─────────────────┬───────────────────────────────┘
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
┌─────────────┐       ┌────────────────────┐
│  ChromaDB   │       │  Student State     │
│  (RAG)      │       │  JSON + Pydantic   │
└──────┬──────┘       └────────────────────┘
       │
┌──────▼──────┐
│   Ollama    │
│ llama3.1:8b │
└─────────────┘
```

### How a single turn flows

1. `load_state_node` reads the student's JSON file, loading mastery scores into the graph state
2. `select_topic_node` picks the topic with the lowest mastery (70% of the time) or an unexplored topic (30%)
3. `generate_problem_node` calls Ollama with a structured prompt, asking for a pure-notation, LaTeX-formatted problem (`$\frac{1}{6} + \frac{2}{3} =$` — no word problems) plus a SymPy expression representing the same computation. The node evaluates the expression itself — the LLM's arithmetic is never trusted. A repair layer fixes the LaTeX-in-JSON escaping mistakes LLMs commonly make (`\frac` silently decoding as a formfeed character, `\sqrt` breaking the parse)
4. **Graph pauses** (`interrupt_before=["evaluate_answer_node"]`) — the UI renders the problem and the student submits an answer
5. `evaluate_answer_node` runs `symbolic_check()`: parses the student's input with SymPy and checks `simplify(student - correct) == 0`. If wrong, calls Ollama to categorize the error type
6. `retrieve_explanation_node` queries ChromaDB using the **problem text** as the semantic query (with topic/subtopic/difficulty/error-category metadata filters), returning the 3 most relevant knowledge chunks
7. `generate_feedback_node` calls Ollama with the retrieved chunks injected — produces a 2–3 sentence concept note (separate prompts for correct vs. incorrect answers). The derivation itself comes from `solution_steps`, computed by SymPy in step 3 and rendered verbatim — the LLM never writes math steps. The error category from step 5 is never shown to the student — it only sharpens retrieval and accumulates in the student's error-pattern statistics
8. `update_state_node` updates the mastery score with an EMA, adjusts difficulty, and saves to disk
9. `adapt_next_node` sets up the next topic and difficulty in state, then the graph reaches **END**
10. **UI enters reviewing phase** — the problem, the student's answer, and the full explanation are displayed together. The student clicks "Next problem →" to trigger a new turn from step 1

---

## Technology Deep Dive

### LangGraph — stateful agentic loop

LangGraph is used to define the tutoring session as an explicit state machine rather than an open-ended agent. Each step is a typed node that reads from and writes to `TutorState` (a `TypedDict`). The graph is compiled with:

- **`MemorySaver` checkpointer**: snapshots the full state after each node, keyed by `thread_id` (the student's name). Sessions resume from exactly where they left off.
- **`interrupt_before=["evaluate_answer_node"]`**: the graph pauses before evaluation and hands control back to the UI. When the student submits, the UI injects the answer via `graph.update_state()` and calls `graph.stream(None)` to resume.

This is the pattern that enables human-in-the-loop interaction without polling or threading. The graph reaches `END` after `adapt_next_node` — it does **not** loop back automatically. This keeps the feedback and the answered problem on screen until the student explicitly requests the next one.

```python
# Turn start: graph runs load → select → generate, pauses before evaluate
graph.stream(initial_state, config={"configurable": {"thread_id": "alice"}})

# Student submits: inject answer, run evaluate → retrieve → feedback → update → adapt → END
graph.update_state(config, {"student_answer": "3/4"})
graph.stream(None, config=config)

# Student clicks "Next problem": start a fresh turn
graph.stream({"student_answer": ""}, config=config)
```

Why LangGraph over LangChain's `AgentExecutor`: the tutoring session has a fixed, well-defined sequence of steps. An explicit graph is more predictable, easier to debug, and clearer to reason about than an open-ended ReAct loop that decides at runtime what to do next.

### ChromaDB — retrieval-augmented generation (RAG)

ChromaDB stores a curated knowledge base of math explanations, worked examples, and common misconception notes. Each chunk is tagged with `topic`, `subtopic`, `difficulty`, and `misconception_tag` metadata.

When a student answers a problem, `retrieve_explanation_node` queries ChromaDB with metadata filters before performing semantic search. This ensures that retrieved chunks are:
- On the right topic and subtopic
- At an appropriate difficulty level
- Matched to the student's specific error type when possible (e.g., a `sign_error` retrieves chunks tagged `misconception_tag: sign_error`)

The retrieved text is then injected directly into the feedback generation prompt. This is the RAG pattern: the LLM generates explanations grounded in verified content, not just trained intuition.

```
Semantic query: the full problem text (e.g. "Solve for x: 2x + 5 = 11")
Metadata filters: topic=algebra, subtopic=linear_equations, misconception_tag=sign_error
→ Returns: 3 chunks semantically close to this specific problem
→ Injected into: feedback prompt as reference material
```

Using the problem text as the semantic query (rather than just `"algebra linear_equations"`) means the embedding search finds chunks that are relevant to the specific numbers and structure of the problem, not just the topic in general.

Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU-compatible). ChromaDB persists to `data/chromadb/` as a local file store — no server or Docker required.

### SymPy — deterministic answer evaluation

Every problem has a `sympy_answer` field stored in the graph state that is never shown to the student. When the student submits, `symbolic_check()` compares their input to the stored answer mathematically:

```python
simplify(sympify(student_input) - sympify(sympy_answer)) == 0
```

This accepts any mathematically equivalent form:
- `1/2`, `0.5`, `2/4`, `4/8` — all correct for the same answer
- `5/6`, `10/12` — equivalent fractions accepted
- `8` — correct answer to `solve(3x - 9 = 15)`

The correct answer is **computed by SymPy from a raw expression**, not taken from the LLM. The LLM provides a `sympy_expression` (e.g., `"Eq(3*x - 9, 15)"` or `"Rational(1,6) + Rational(2,3)"`), and the node evaluates it independently. This eliminates the most common failure mode: an LLM getting the arithmetic wrong on its own output.

Using SymPy instead of an LLM for answer evaluation is a deliberate architectural choice. LLM-based math checking has two unacceptable failure modes in a tutoring context: marking correct answers wrong, and accepting subtly wrong answers.

**Verified solution steps** (`src/agent/solution_steps.py`) extend the same principle to explanations. The numbered derivation shown after every answer is generated by SymPy from the problem's expression — moving the constant, dividing by the coefficient, converting to a common denominator — with every displayed equality verified symbolically (`simplify(a - b) == 0`) before it is emitted. If any internal check fails, the builder falls back to a single `expression = result` step. The LLM never writes math steps, because an 8B model narrating multi-step algebra reliably drifts on numbers (it once explained $2x+5=11$ with "subtract 5 from both sides → $2x-5=6$"). Its role in feedback is reduced to a 2–3 sentence concept note grounded in the retrieved knowledge-base chunks.

### Ollama + llama3.1:8b — local language model

Ollama runs `llama3.1:8b` locally at `localhost:11434`. The LLM is used for three tasks where natural language generation is appropriate:

| Task | Why LLM is appropriate |
|---|---|
| Problem generation | Requires natural language and variety (the math is verified by SymPy afterward) |
| Error categorization | Qualitative judgment about what kind of mistake was made (internal only — drives retrieval and statistics) |
| Feedback concept note | A short, encouraging note naming the concept the problem practices, grounded in retrieved content |

Answer evaluation (right/wrong) and the solution derivation are explicitly **not** delegated to the LLM — both are handled by SymPy.

### Pydantic + JSON — persistent student state

Each student's state is stored as a JSON file at `data/students/{student_id}.json`, validated by Pydantic v2 models:

- `StudentState` — top-level container (student ID, creation time, mastery map, attempt history)
- `TopicMastery` — per-topic model (mastery score, current difficulty, attempt counts, error pattern tallies)
- `AttemptRecord` — immutable record of one problem attempt (timestamp, problem text, answer, result, error category)

Pydantic handles serialization (`model_dump_json`) and deserialization (`model_validate_json`). JSON files are human-readable, easy to inspect during development, and require no database infrastructure.

### Mastery algorithm — Exponential Moving Average

The student's mastery on each topic is a single float from 0.0 to 1.0, updated after every attempt:

```
new_mastery = 0.8 × old_mastery + 0.2 × outcome
```

where `outcome = 1.0` for correct, `0.0` for wrong. This is an EMA with α = 0.2 — recent performance has more influence, but a single wrong answer on a well-mastered topic does not collapse the score.

Difficulty adapts separately: +1 on a correct answer, −1 on wrong, clamped to [1, 5].

The production path for mastery modeling would be Bayesian Knowledge Tracing (BKT), which maintains per-skill estimates of learning rate, guess probability, and slip probability. EMA is used here because it demonstrates the concept clearly without requiring training data for parameter estimation.

### NiceGUI + KaTeX — the interface

The UI is a [NiceGUI](https://nicegui.io) app: a single Python process that serves a modern, event-driven web interface and calls the LangGraph `graph` singleton in-process — no separate backend, no Node toolchain, one command to run (`python src/ui/app.py`).

Why NiceGUI over Streamlit (which this project originally used): Streamlit re-runs the entire script on every interaction, which makes each click feel slow and forces awkward `session_state` bookkeeping. NiceGUI is event-driven — a button click invokes one async handler, and only the elements that change are updated over a persistent websocket. The result is an interface that responds instantly and code that reads like ordinary event handlers.

Two implementation details worth knowing:

- **LaTeX rendering**: problems and explanations contain `$...$` LaTeX. KaTeX (loaded via CDN with its auto-render extension) typesets them in the browser, scoped strictly to the `.math-content` containers whose innerHTML Vue treats as opaque. This scoping is load-bearing: running KaTeX's auto-render on `document.body` merges adjacent text nodes during its scan, which destroys the empty-text-node anchors Vue 3 uses for fragments — the next Vue patch then crashes and the UI freezes. The server triggers typesetting (`typesetMath()`) after each phase change, via a client handle captured at page build (the ambient client context is lost after `await` in NiceGUI handlers).
- **Responsiveness during LLM calls**: `graph.stream()` blocks on Ollama for seconds at a time. The UI consumes the stream one chunk at a time on a worker thread (`run.io_bound`), appending each completed node ("Checking answer (SymPy symbolic solver)", "Searching knowledge base (ChromaDB RAG)", …) to a live operation feed — the same under-the-hood transparency a `graph.stream` loop gives in a terminal.

---

## Knowledge Base

The knowledge base lives in `data/knowledge_base/` as version-controlled JSON files. Each chunk has:

```json
{
  "id": "algebra_linear_eq_misconception_sign",
  "text": "Common error: when moving a term across the equals sign...",
  "metadata": {
    "topic": "algebra",
    "subtopic": "linear_equations",
    "content_type": "common_misconception",
    "difficulty": 2,
    "misconception_tag": "sign_error"
  }
}
```

**Content types**:
- `concept_explanation` — explains what a concept is and how to approach it
- `worked_example` — walks through a sample problem step by step
- `common_misconception` — describes a specific error pattern and why it's wrong

**Topics covered**:
- `fractions_ratios` ("Fractions & Ratios"): equivalent fractions, addition/subtraction, multiplication/division, proportions, percentages
- `algebra`: linear equations, evaluating expressions, linear relationships (slope, y = mx + b)

Run `python scripts/ingest_kb.py` to load all chunks into ChromaDB. The script is idempotent — re-running it skips chunks that are already indexed. (Because the loader only *adds* new chunk ids, delete `data/chromadb/` and re-run after renaming or removing chunks.)

---

## Running Tests

```bash
# Fast unit tests — no LLM required
.venv/bin/python -m pytest tests/test_symbolic_check.py tests/test_answer_evaluation.py -v

# LLM integration tests — requires Ollama running (~3 min)
.venv/bin/python -m pytest tests/test_problem_generation.py -v -s

# Full end-to-end pipeline tests — requires Ollama running (~5 min)
.venv/bin/python -m pytest tests/test_pipeline.py -v -s

# All tests
.venv/bin/python -m pytest tests/ -v
```

---

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
CHROMADB_PATH=data/chromadb
STUDENT_STATE_DIR=data/students
KNOWLEDGE_BASE_DIR=data/knowledge_base
```

---

## Design Decisions

**Why not evaluate answers with an LLM?** LLM math evaluation is non-deterministic and fails in both directions: it can mark correct answers wrong and accept subtly wrong answers. SymPy compares symbolic expressions and eliminates both failure modes. In a tutoring context, a wrong grade is worse than no grade.

**Why LangGraph instead of a simple loop?** LangGraph's checkpointer enables the pause/resume pattern that allows the UI to wait for human input without threading. It also provides typed state that makes each node's inputs and outputs explicit, which makes the system easier to test and extend.

**Why ChromaDB instead of a simple list of strings?** Metadata filtering means retrieved chunks are relevant to the specific topic, difficulty, and error type — not just semantically similar. A sign error in algebra should retrieve content about sign errors, not general algebra explanations. This is the difference between useful feedback and generic feedback.

**Why local models (Ollama) instead of an API?** The project is fully self-contained: no API keys, no cost per run, no data leaving the machine. `llama3.1:8b` is capable enough for problem generation and explanation. The model is parameterized in `.env` and can be swapped without code changes.

**Why NiceGUI instead of Streamlit or a React frontend?** Streamlit's rerun-per-interaction model made every click slow. A React/shadcn split was considered but rejected: it would require a separate backend API, a Node toolchain, and two processes — at odds with a demo that should run with one command. NiceGUI is the middle path: a real event-driven UI, modern look, LaTeX support via KaTeX, and still a single pip-installed Python process calling the graph directly.

**Why pure-notation problems instead of word problems?** Two reasons. Reliability: word problems require the LLM to keep the story and the underlying math consistent, which is where answer-mismatch hallucinations crept in. Clarity: the SymPy expression and the rendered problem are now two views of exactly the same mathematical object, which makes verification airtight.

**Why doesn't the LLM write the explanations?** It was tried, and it hallucinated arithmetic mid-derivation (e.g. "subtract 5 from both sides of $2x+5=11$ → $2x-5=6$"). No prompt fixes that reliably in an 8B model, and the knowledge base can't either — RAG grounds concepts, not per-problem arithmetic. So derivations are generated by SymPy (`src/agent/solution_steps.py`), with every displayed equality verified symbolically before it's shown; the LLM contributes only a short concept note grounded in the retrieved chunks. Wrong math in an explanation is now structurally impossible, not just unlikely.
