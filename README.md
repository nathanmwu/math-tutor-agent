# Math Tutor Agent

An adaptive K-12 arithmetic and algebra tutor that generates problems, checks answers with a symbolic math solver (SymPy), and grounds its explanations in a curated knowledge base (RAG). It covers **Fractions & Ratios** (equivalent fractions, the four operations, proportions, percentages) and **Algebra** (linear equations, evaluating expressions, linear relationships). Difficulty and topic selection adapt to each student's history.

Built as a portfolio project demonstrating RAG, a stateful LangGraph agent loop, deterministic SymPy evaluation, and persistent student modeling. Runs fully locally — no API keys.

---

## What It Does

Each turn: pick the student's weakest subtopic → generate a problem in pure LaTeX notation (e.g. $\frac{1}{6} + \frac{2}{3} =$) → the student answers (any equivalent form accepted: `1/2`, `0.5`, `2/4`) → SymPy checks it → retrieve a relevant explanation → show a SymPy-verified solution plus a short concept note → update mastery and difficulty → next problem.

---

## Quick Start

**Prerequisites**: Python 3.12, [Ollama](https://ollama.com) running.

```bash
ollama pull llama3.1:8b
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/ingest_kb.py   # populate the knowledge base (run once)
python src/ui.py              # serves http://localhost:8501
```

Open http://localhost:8501, enter your name, and start practicing.

---

## How It Works

A LangGraph state machine drives each turn as five nodes, with one pause for the student's answer:

```
setup → generate_problem → [PAUSE: student answers] → evaluate_answer → explain → update_state
```

- **setup** — load the student, pick the weakest subtopic, set difficulty from mastery
- **generate_problem** — Ollama writes a LaTeX problem plus a SymPy expression; the expression is evaluated locally (the LLM's arithmetic is never trusted), and a SymPy-verified derivation is precomputed
- **evaluate_answer** — `symbolic_check()` compares answers with `simplify(student - correct) == 0`; wrong answers are categorized by the LLM (internal, used to sharpen retrieval)
- **explain** — ChromaDB RAG retrieves relevant chunks; Ollama writes a 2–3 sentence concept note grounded in them (the math derivation comes from SymPy, never the LLM)
- **update_state** — EMA mastery update and difficulty adjustment, saved to per-student JSON

The graph pauses via `interrupt_before=["evaluate_answer_node"]` and resumes when the UI submits the answer. Mastery is tracked per subtopic with an exponential moving average; right/wrong and the step-by-step solution are SymPy's job, not the LLM's.

---

## Project Structure

```
src/
├── pipeline.py        # the 5-node LangGraph loop + TutorState + symbolic_check()
├── prompts.py         # all LLM prompt templates
├── solution_steps.py  # SymPy-verified derivations (the LLM never writes math)
├── student.py         # Pydantic state models + persistence + EMA mastery
├── knowledge.py       # ChromaDB ingest + retrieval (RAG)
└── ui.py              # NiceGUI app (LaTeX via KaTeX, mastery dashboard)
data/
├── knowledge_base/    # source JSON chunks (version-controlled)
├── chromadb/          # vector store (generated, gitignored)
└── students/          # per-student state (generated, gitignored)
scripts/ingest_kb.py   # one-time knowledge-base ingestion
tests/                 # symbolic_check, solution_steps, personalization, generation, pipeline
```

The knowledge base in `data/knowledge_base/` is the source of truth — re-run `python scripts/ingest_kb.py` after editing it (delete `data/chromadb/` first to drop removed chunks).

---

## Running Tests

```bash
# Fast unit tests — no LLM required
.venv/bin/python -m pytest tests/test_symbolic_check.py tests/test_solution_steps.py tests/test_personalization.py -v

# LLM integration tests — requires Ollama running
.venv/bin/python -m pytest tests/test_problem_generation.py tests/test_pipeline.py -v -s

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
