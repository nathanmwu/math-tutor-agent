# Math Tutor Agent — Claude Code Guide

## Project Purpose

A portfolio project demonstrating core AI-engineering techniques in a K-12 math tutor. The goal is to be *correct*, *clear*, and *complete* as a demonstration of technique — not production-scale. Each component deliberately exposes one technique (RAG, an agentic loop, symbolic verification, adaptive state) so the implementation is legible and defensible in an interview context.

---

## AI Techniques Demonstrated

| Technique | Where in the codebase | Why it's here |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | `src/knowledge.py` + `src/pipeline.py:explain_node` | Grounds LLM feedback in verified educational content rather than parametric memory |
| **Agentic loop** | `src/pipeline.py` | A LangGraph StateGraph drives a fixed sequence: setup → generate → pause → evaluate → explain → update → END. The UI advances to the next turn explicitly. |
| **Adaptive behavior** | `src/pipeline.py:setup_node` + `src/student.py` | Mastery tracked per subtopic via EMA; difficulty and topic selection adjust each turn |
| **Persistent student state** | `src/student.py` + `data/students/` | Pydantic models serialized to per-student JSON; survives session restarts |
| **Symbolic answer verification** | `src/pipeline.py:evaluate_answer_node` via SymPy | Deterministic math evaluation — right/wrong is never delegated to an LLM |
| **Verified solution steps** | `src/solution_steps.py` | The displayed derivation is SymPy-generated; every emitted equality is checked symbolically before display. The LLM only writes a short concept note |
| **Structured LLM output** | `src/prompts.py` + `src/pipeline.py:_parse_problem_json` | Problem generation returns validated JSON; a repair layer fixes LaTeX-in-JSON escaping corruption |

---

## Architecture at a Glance

```
NiceGUI UI (single process, event-driven; KaTeX renders $...$ LaTeX)
    ↕ graph.stream (in-process, consumed via run.io_bound)
LangGraph StateGraph (TutorState) — src/pipeline.py
    ├── setup            → load student, pick weakest subtopic, set difficulty from mastery
    ├── generate_problem → Ollama (llama3.1:8b) → pure-notation LaTeX problem JSON;
    │                       _parse_problem_json repairs escaping; SymPy evaluates the
    │                       expression and precomputes the verified derivation
    ├── [PAUSE]          → UI renders problem, student submits answer
    ├── evaluate_answer  → SymPy symbolic check (deterministic)
    │                       + Ollama error categorization (wrong only, internal)
    ├── explain          → ChromaDB RAG (problem text as semantic query) + Ollama;
    │                       2-3 sentence concept note grounded in the retrieved chunks
    │                       (the derivation comes from solution_steps, not the LLM)
    └── update_state     → EMA mastery update + write JSON → END
                            UI shows feedback; "Next problem" starts a fresh turn at setup
```

The pipeline is five nodes with one human-in-the-loop pause. Next-turn difficulty is recomputed in `setup` from the freshly-saved mastery, so there is no separate "adapt" step.

Full system design: see [architecture.md](architecture.md). Requirements/API contracts: see [project_spec.md](project_spec.md).

---

## Key Design Decisions

- **SymPy for answer evaluation and derivations** — LLM math checking marks correct answers wrong and accepts wrong ones; SymPy compares symbolic expressions deterministically. Derivations are SymPy-generated with every equality verified before display. The LLM is used only where natural language fits: problem generation, error categorization, and the concept note.
- **LangGraph for the loop** — The session is a fixed state machine, not an open-ended ReAct loop. `MemorySaver` + `interrupt_before=["evaluate_answer_node"]` give the pause/resume the UI needs without threading.
- **ChromaDB for RAG** — Local `PersistentClient`, no server. Metadata filters (topic/subtopic/difficulty/misconception) keep retrieval relevant to the specific problem and error.
- **Pydantic + JSON state** — Human-readable, git-friendly, one file per student.
- **NiceGUI + KaTeX** — Single event-driven process calling the graph in-process. KaTeX is scoped strictly to `.math-content` divs — NEVER run it on `document.body`: its text-node scan destroys Vue's empty-text-node fragment anchors and freezes the UI. Typesetting is triggered via the `page_client` handle captured at page build (ambient client context is lost after `await` in handlers).
- **Pure-notation problems** — Problems are written mathematically (`$\frac{1}{6} + \frac{2}{3} =$`), so `problem_text` and `sympy_expression` are two views of one object. The generator only emits problems with a concrete numeric answer (the `is_number` gate).

---

## Local Setup

```bash
# Prerequisites: Python 3.12, Ollama running with llama3.1:8b pulled
ollama pull llama3.1:8b
pip install -r requirements.txt
python scripts/ingest_kb.py     # populate the knowledge base (run once)
python src/ui.py                # single process — serves http://localhost:8501
```

---

## Project Structure

```
Tutor-Agent/
├── CLAUDE.md / README.md / architecture.md / project_spec.md
├── requirements.txt / .env.example
├── scripts/ingest_kb.py        # one-time KB ingestion
├── data/
│   ├── knowledge_base/         # source JSON chunks (version-controlled)
│   ├── chromadb/               # ChromaDB persistence (gitignored)
│   └── students/               # per-student state JSON (gitignored)
└── src/
    ├── pipeline.py             # TutorState + the 5-node graph + symbolic_check()
    ├── prompts.py              # all LLM prompt templates
    ├── solution_steps.py       # SymPy-verified derivations
    ├── student.py              # Pydantic models + persistence + EMA mastery
    ├── knowledge.py            # ChromaDB ingest + retrieval (RAG)
    └── ui.py                   # NiceGUI app (KaTeX rendering, live op feed)
```

---

## Conventions

- All LLM calls go through `src/pipeline.py` (via `_get_llm()`) — never call Ollama directly from the UI or knowledge layer.
- All prompt templates live in `src/prompts.py` — never embed prompt strings inline.
- All student state mutations go through `src/student.py` — never write JSON directly from a node.
- SymPy evaluation lives in `src/pipeline.py:symbolic_check()` — keep it isolated so it's easy to swap or extend.
- Knowledge base source files in `data/knowledge_base/` are the source of truth — re-run `ingest_kb.py` if ChromaDB is deleted.
- The UI (`src/ui.py`) only imports the compiled `graph` (and `TutorState`) from `src/pipeline.py` — never import the nodes, student, or knowledge modules directly.
- All math shown to the student is LaTeX wrapped in `$...$`; problems are pure notation; the error category is never displayed.
- Solution derivations are SymPy-generated (`src/solution_steps.py`) and rendered verbatim — the LLM never writes math steps. Unverifiable shapes fall back to a single `expression = result` step.
- The UI must stay responsive during LLM calls: consume `graph.stream` via `run.io_bound` chunk-by-chunk, never block the event loop.
```
