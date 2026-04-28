# Tectika Multi-Agent Research System

A Python backend service that uses a 5-agent pipeline — built on **LangChain primitives** + **Azure OpenAI (GPT-4o)** — to research any topic and produce a structured professional report.

## Live Deployment

A working instance is hosted on **Azure App Service** so reviewers can test the pipeline without running anything locally.

**Base URL:** `https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net`

> 👉 **Easiest way to try it: open the base URL in a browser.** The root path (`/`) serves a built-in web UI — just type a topic, press Research, and watch the agent trace and final report render in place.

| Endpoint | URL |
|---|---|
| **Web UI** | [`/`](https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/) |
| Interactive Swagger UI | [`/docs`](https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/docs) |
| Liveness probe | [`/health`](https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/health) |
| OpenAPI schema | [`/openapi.json`](https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/openapi.json) |
| Run pipeline (single response) | `POST /run` |
| Run pipeline (SSE stream) | `POST /run/stream` |

### One-liner test (no setup required)

```bash
# Liveness — should return {"status":"ok"} immediately
curl https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/health

# Full pipeline against the live deployment (~50–60 s end-to-end against Azure OpenAI)
curl -X POST https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/run \
  -H "Content-Type: application/json" \
  -d '{"topic":"The impact of LLMs on backend engineering workflows"}'
```

For a live progress feed, use the streaming endpoint — each agent emits a `trace` event the moment it finishes, with researchers arriving in completion order:

```bash
curl -N -X POST https://tectika-agent-api-atg2c4acg5dmaydm.eastus-01.azurewebsites.net/run/stream \
  -H "Content-Type: application/json" \
  -d '{"topic":"The impact of LLMs on backend engineering workflows"}'
```

The deployed instance is the same image this repository's `Dockerfile` produces, with the four `AZURE_OPENAI_*` values configured as App Service **Application Settings** (which the runtime exposes to the container as environment variables — `pydantic-settings` picks them up directly from `os.environ`, no `.env` file in production).

## Architecture

```
POST /run         (topic)  ─►  RunResponse  (report + agent_trace + meta{duration_ms,tokens})
POST /run/stream  (topic)  ─►  SSE stream of trace events, then a final 'complete' event
      │
      ▼
 ManagerAgent.stream()  ──►  yields events as each agent finishes
      │
      │ 1. decompose
      ▼
 PlannerAgent  →  [sub_question_1, sub_question_2, ..., sub_question_N]
      │
      │ 2. research (concurrent via asyncio.create_task + as_completed)
      ├──► ResearcherAgent(sub_question_1)  [bind_tools loop with web_search]
      ├──► ResearcherAgent(sub_question_2)  [bind_tools loop with web_search]
      └──► ResearcherAgent(sub_question_N)  [bind_tools loop with web_search]
      │
      │ 3. aggregate
      ▼
 AggregatorAgent  →  consolidated, deduplicated knowledge document
      │
      │ 4. write
      ▼
  WriterAgent  →  final polished report
```

| Agent | Role | LangChain feature |
|---|---|---|
| Manager | Orchestrates the pipeline; assembles `agent_trace` and `RunResponse` | None (pure orchestration with `asyncio.create_task` + `as_completed`) |
| Planner | Decomposes research topic into 3–6 focused sub-questions | `AzureChatOpenAI.with_structured_output(PlannerOutput)` |
| Researcher | Gathers facts per sub-question; runs concurrently | `AzureChatOpenAI.bind_tools([web_search])` + manual message loop |
| Aggregator | Merges and deduplicates all researcher outputs | `AzureChatOpenAI.ainvoke([SystemMessage, HumanMessage])` |
| Writer | Formats consolidated findings into a professional report | Same as Aggregator |

## Prerequisites

- **Python 3.10+** (only required for running locally without Docker)
- **Docker + Docker Compose** (only required for the Docker path)
- **Azure OpenAI deployment** (GPT-4o via Azure AI Foundry) — credentials provided separately
- **DuckDuckGo search** for real web search — no extra API key is required, and the Researcher falls back to GPT-4o's training knowledge if search is unavailable

## Setup — without Docker

```bash
# 1. Clone and enter the project
git clone https://github.com/NechemiaR/tectika-task.git
cd tectika-task

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your Azure credentials (see Environment Variables below)

# 5. Start the server
uvicorn main:app --reload
```

The API will be available at **http://localhost:8000**. Interactive Swagger docs: **http://localhost:8000/docs**.

## Setup — with Docker

```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env with your Azure credentials

# 2. Build and start the container
docker-compose up --build

# Or run detached
docker-compose up --build -d

# Stop
docker-compose down
```

The API is exposed on **http://localhost:8001** when running with Docker. JSON-structured logs stream to the container's stdout (`docker-compose logs -f`).

Note: the app does not define a `/` route, so opening `http://localhost:8001/` will return `404 Not Found`. Use `http://localhost:8001/docs` for the Swagger UI, or call `POST /run` directly.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure endpoint URL (e.g. `https://your-resource.openai.azure.com/`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | GPT-4o deployment name in Azure AI Foundry |
| `AZURE_OPENAI_API_VERSION` | Yes | API version (default: `2024-02-01`) |

## API Usage

### `POST /run`

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "The impact of LLMs on backend engineering workflows"}'
```

If you are using Docker, send the request to `http://localhost:8001/run` instead.

**Response shape:**

```json
{
  "report": "# The Impact of LLMs on Backend Engineering Workflows\n\n...",
  "agent_trace": [
    {
      "agent": "Planner",
      "action": "decompose_topic",
      "input_summary": "The impact of LLMs on backend engineering workflows",
      "output_summary": "Generated 5 sub-questions from topic",
      "output": "1. What AI-based coding tools are most commonly used...\n2. ...",
      "duration_ms": 1230.1,
      "tokens": { "input_tokens": 142, "output_tokens": 88, "total_tokens": 230 }
    },
    {
      "agent": "Researcher",
      "action": "research_sub_question",
      "input_summary": "What AI-based coding tools are most commonly used by backend engineers?",
      "output_summary": "- GitHub Copilot adoption: 55% of developers...",
      "output": "<full multi-paragraph factual findings>",
      "duration_ms": 4210.8,
      "tokens": { "input_tokens": 1820, "output_tokens": 412, "total_tokens": 2232 }
    }
  ],
  "meta": {
    "duration_ms": 18420.5,
    "tokens": { "input_tokens": 12480, "output_tokens": 3120, "total_tokens": 15600 }
  }
}
```

The PDF spec shows each trace entry as `{agent, action, output}`. We return a strict superset: every entry includes those three keys plus `input_summary`, `output_summary`, `duration_ms`, and `tokens` so consumers can audit timing and cost per agent without re-running the pipeline.

The `agent_trace` contains one entry per agent action. Each entry carries:

- `output_summary` — short human-readable description of what the agent produced.
- `output` — the **full** content the agent produced (Researcher findings, consolidated document, final report, …) so consumers can audit every step.
- `tokens` — per-agent input/output/total token counts captured from the Azure OpenAI response. Researcher entries sum tokens across all tool-loop iterations.

Total token usage and total wall-clock duration are surfaced under `meta`.

Researcher entries run concurrently — their individual `duration_ms` values will be similar to each other but far less than `meta.duration_ms`, which proves the parallel execution.

### `POST /run/stream` (Server-Sent Events)

Same input as `/run`, but the server streams agent traces as each agent completes. Useful for long pipelines where you want progressive feedback instead of waiting 15–30 s for a single JSON blob.

```bash
curl -N -X POST http://localhost:8000/run/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "The impact of LLMs on backend engineering workflows"}'
```

Each line on the wire is `data: <json>\n\n`. Event types:

| `type` | When emitted | `data` payload |
|---|---|---|
| `trace` | After each agent finishes (Researchers in **completion order**) | A single `TraceEntry` |
| `complete` | After the Writer finishes | The full `RunResponse` |
| `error` | If the pipeline raises | `{"message": "Internal pipeline error"}` |

This makes the parallelism observable: you'll see all Researcher `trace` events arrive close together, well before the Aggregator and Writer events.

### Web UI (`GET /`)

Opening the base URL in a browser serves a small built-in frontend that drives the same `POST /run` endpoint:

- A topic input + **Research** button.
- A loading state with a live elapsed-time counter (the pipeline takes 45–60 s).
- A **Research Process** panel that renders one card per `agent_trace` entry — colour-coded badge per agent (Planner / Researcher / Aggregator / Writer), the sub-question or input, a snippet of the output, plus per-agent `duration_ms` and total tokens.
- A **Final Report** panel that renders the Writer's markdown into proper headings, lists, and emphasis.
- Inline error banner if the call fails (surfaces FastAPI's structured `detail` for validation errors).

The frontend is a single self-contained `static/index.html` (~430 lines, embedded CSS + vanilla ES2020) — no build step, no framework, no bundler. It is mounted from `main.py` via `FileResponse` and is hidden from the OpenAPI schema (`include_in_schema=False`) so `/docs` stays focused on the JSON API.

## Design Decisions

**LangChain primitives, not the agent framework** — All agents use LangChain's typed message and model classes (`AzureChatOpenAI`, `SystemMessage`, `HumanMessage`, `ToolMessage`, `@tool`, `bind_tools`, `with_structured_output`). The `AgentExecutor` / LangGraph stack is intentionally *not* used — the orchestration loop in `ManagerAgent` and the tool-calling loop in `ResearcherAgent` are written explicitly. This keeps the code transparent: every step the system takes is visible in the source.

**DuckDuckGo for web search** — No extra API key is required and the tool returns compact result snippets that fit the agent's tool loop. The code wraps the blocking search call in `asyncio.to_thread()` so the async pipeline stays responsive.

**`asyncio.create_task` + `as_completed` for concurrent research** — All Researcher coroutines are scheduled simultaneously, and we consume them with `asyncio.as_completed` so the SSE stream emits each researcher's trace **as it finishes** rather than in submission order. `ResearcherAgent.run()` keeps state in local variables only, making concurrent calls on a single instance safe.

**`pydantic-settings` for config** — Missing required environment variables raise a clear `ValidationError` at startup with field names listed, rather than surfacing as a `None`-dereference deep in an LLM call.

**Structured logging** — JSON-formatted log lines via `python-json-logger`. Each agent has its own logger (`tectika.planner`, `tectika.researcher`, etc.) so log lines are easy to grep by agent. Token counts are logged alongside `duration_ms` for every agent call.

**Single pipeline, two transports** — `ManagerAgent.stream()` is an async generator that yields events as agents finish. `POST /run` consumes the generator and returns the final `complete` event as a single JSON response; `POST /run/stream` forwards every event over SSE. There is no duplicated orchestration logic between the two endpoints.

## Project Structure

```
tectika/
├── agents/
│   ├── manager.py       # Orchestrator — asyncio.create_task + as_completed, trace assembly
│   ├── planner.py       # Topic decomposition (with_structured_output)
│   ├── researcher.py    # Concurrent research (bind_tools + tool loop)
│   ├── aggregator.py    # Deduplication and merging
│   └── writer.py        # Final report generation
├── tools/
│   └── web_search.py    # DuckDuckGo search wrapper
├── models/
│   └── schemas.py       # Pydantic v2 models for API I/O
├── core/
│   ├── config.py        # pydantic-settings BaseSettings
│   └── logging_config.py # JSON structured logging
└── api/
    └── routes.py        # GET /health, POST /run, POST /run/stream
main.py                  # FastAPI entry point — also serves GET / (frontend)
static/
└── index.html           # Single-file vanilla-JS frontend (embedded CSS, no build step)
docs/
└── agent-architecture.md # Original design document
```
