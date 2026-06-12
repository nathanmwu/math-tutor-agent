# Math Tutor Agent — Claude Code Guide

## Project Purpose

This is a portfolio project demonstrating core AI engineering techniques in the context of K-12 math tutoring. It is designed to be:

1. **Educational for the builder** — each component deliberately exposes one AI technique so the implementation teaches as much as the outcome.
2. **Legible to EdTech engineering teams** (e.g., IXL Learning) — architectural decisions are explicit, documented, and defensible in an interview context.

The system is not trying to be production-scale. It is trying to be *correct*, *clear*, and *complete* as a demonstration of technique.

---

## AI Techniques Demonstrated

| Technique | Where in the codebase | Why it's here |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | `src/knowledge/` + `src/agent/nodes.py:retrieve_explanation` | Grounds LLM feedback in verified educational content rather than parametric memory |
| **Agentic loop with tool use** | `src/agent/graph.py` | LangGraph StateGraph drives a deterministic sequence: generate → pause → evaluate → retrieve → feedback → adapt → END. The UI advances to the next turn explicitly. |
| **Adaptive behavior** | `src/agent/nodes.py:adapt_next` + `src/state/` | Student mastery tracked per topic via EMA; difficulty and topic selection adjust each turn |
| **Persistent student state** | `src/state/models.py` + `data/students/` | Pydantic models serialized to per-student JSON; survives session restarts |
| **Symbolic answer verification** | `src/agent/nodes.py:evaluate_answer` via SymPy | Deterministic math evaluation — right/wrong is never delegated to an LLM |
| **Structured LLM output** | `src/agent/prompts.py` + `nodes.py:_parse_problem_json` | Problem generation returns validated JSON (`problem_text` in LaTeX, `sympy_expression`, `topic`, `subtopic`); a repair layer fixes LaTeX-in-JSON escaping corruption |

---

## Architecture at a Glance

```
NiceGUI UI (single process, event-driven; KaTeX renders $...$ LaTeX)
    ↕ graph.stream (in-process, consumed via run.io_bound)
LangGraph StateGraph (TutorState)
    ├── load_state         → reads data/students/{id}.json
    ├── select_topic       → weakest topic or next curriculum slot
    ├── generate_problem   → Ollama (llama3.1:8b) → pure-notation LaTeX problem JSON;
    │                         _parse_problem_json repairs escaping; SymPy evaluates expression
    ├── [PAUSE]            → UI renders problem, student submits answer
    ├── evaluate_answer    → SymPy symbolic check (deterministic)
    │                         + Ollama error categorization (wrong only, internal)
    ├── retrieve_explanation → ChromaDB RAG (problem text as semantic query)
    ├── generate_feedback  → Ollama with retrieved chunks; Result + step-by-step
    │                         LaTeX Explanation (error category never displayed)
    ├── update_state       → EMA mastery update + write JSON
    └── adapt_next         → sets next difficulty/topic → END
                              UI shows feedback; "Next problem" triggers a new turn from load_state
```

Full system design: see [architecture.md](architecture.md).
Full requirements and API contracts: see [project_spec.md](project_spec.md).

---

## Key Design Decisions

**LangGraph over LangChain AgentExecutor** — The tutoring loop is a well-defined state machine, not an open-ended ReAct loop. An explicit graph is easier to reason about, test, and explain. Contrast with the `network-recommendation-engine` project which uses `create_tool_calling_agent` — that project has an open-ended query pattern; this one does not.

**SymPy for answer evaluation** — LLM evaluation of math answers has unacceptable failure modes (marking correct answers wrong; accepting wrong answers). SymPy compares symbolic expressions deterministically. LLM is still used for error categorization, which is a qualitative task it handles well.

**ChromaDB for local vector store** — No Docker, no server, `PersistentClient(path=...)` in-process. Adequate for a knowledge base of ~300 chunks.

**EMA mastery over Bayesian Knowledge Tracing** — BKT requires per-skill parameter estimation and training data. EMA (`0.8 * old + 0.2 * new`) is transparent, correct in direction, and sufficient for demonstrating adaptive behavior. BKT is noted in the README as the production path.

**Pydantic + JSON over SQLite** — Human-readable, git-friendly, zero infrastructure. One file per student eliminates locking. SQLite is a trivial upgrade if cross-student querying becomes needed.

**NiceGUI over Streamlit / React+shadcn** — Streamlit's rerun-per-interaction model was too slow; a React frontend would force a backend API split, Node toolchain, and two processes. NiceGUI is event-driven, single-process, pip-only, and calls the graph in-process. LaTeX is rendered by KaTeX (`src/ui/app.py:KATEX_HEAD`), scoped strictly to `.math-content` divs — NEVER run KaTeX on `document.body`: its text-node scan destroys Vue's empty-text-node fragment anchors and freezes the UI. Trigger typesetting via the `page_client` handle captured at page build (ambient client context is lost after `await` in handlers).

**Pure-notation problems over word problems** — Problems are written purely mathematically (`$\frac{1}{6} + \frac{2}{3} =$`, `$3x - 9 = 12$, $x = ?$`). Word problems let the story and the math drift apart (the source of past answer-mismatch bugs); pure notation keeps `problem_text` and `sympy_expression` two views of the same object.

---

## Local Setup

```bash
# Prerequisites: Python 3.12, Ollama running with llama3.1:8b pulled
ollama pull llama3.1:8b

# Install dependencies
pip install -r requirements.txt

# Populate the knowledge base (run once)
python scripts/ingest_kb.py

# Start the app (single process — serves http://localhost:8501)
python src/ui/app.py
```

---

## Project Structure

```
Tutor-Agent/
├── CLAUDE.md              ← this file
├── README.md
├── project_spec.md        ← full requirements and API contracts
├── architecture.md        ← system design and data flow
├── requirements.txt
├── .env.example
├── scripts/
│   └── ingest_kb.py       # one-time KB ingestion
├── data/
│   ├── knowledge_base/    # source JSON chunks (version-controlled)
│   │   ├── fractions.json
│   │   ├── ratios.json
│   │   ├── algebra.json
│   │   └── geometry.json
│   ├── chromadb/          # ChromaDB persistence (gitignored)
│   └── students/          # per-student state JSON (gitignored)
└── src/
    ├── knowledge/
    │   ├── loader.py       # ingest data/knowledge_base/ → ChromaDB
    │   └── retriever.py    # retrieve(topic, subtopic, difficulty, error_category)
    ├── state/
    │   ├── models.py       # Pydantic: AttemptRecord, TopicMastery, StudentState
    │   └── store.py        # load_student(), save_student()
    ├── agent/
    │   ├── graph.py        # StateGraph definition
    │   ├── nodes.py        # all node functions
    │   └── prompts.py      # all prompt templates
    └── ui/
        └── app.py          # NiceGUI app (KaTeX rendering, live op feed)
```

---

## Conventions

- All LLM calls go through `src/agent/nodes.py` — never call Ollama directly from the UI or retriever.
- All prompt templates live in `src/agent/prompts.py` — never embed prompt strings inline.
- All student state mutations go through `src/state/store.py` — never write JSON directly from a node.
- SymPy evaluation lives in `src/agent/nodes.py:symbolic_check()` — keep it isolated so it's easy to swap or extend.
- Knowledge base source files in `data/knowledge_base/` are the source of truth — re-run `ingest_kb.py` if ChromaDB is deleted.
- The UI (`src/ui/app.py`) only touches `src/agent/graph.py` — never import nodes, store, or retriever from the UI.
- All math shown to the student is LaTeX wrapped in `$...$` — problems are pure notation (no word problems); feedback is `Result:` + `Explanation:` only, the error category is never displayed.
- The UI must stay responsive during LLM calls: consume `graph.stream` via `run.io_bound` chunk-by-chunk, never block the event loop.
