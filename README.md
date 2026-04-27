# Tectika Multi-Agent Research System

A Python backend service that uses a 5-agent pipeline — built on **LangChain primitives** + **Azure OpenAI (GPT-4o)** — to research any topic and produce a structured professional report.

## Architecture

```
POST /run (topic)
      │
      ▼
 ManagerAgent  ──────────────────────────────────────────────────────►  RunResponse
      │                                                                   (report +
      │ 1. decompose                                                       agent_trace +
      ▼                                                                    duration_ms)
 PlannerAgent  →  [sub_question_1, sub_question_2, ..., sub_question_N]
      │
      │ 2. research (concurrent via asyncio.gather)
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
| Manager | Orchestrates the pipeline; assembles `agent_trace` and `RunResponse` | None (pure orchestration with `asyncio.gather`) |
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
      "agent": "Manager",
      "action": "orchestrate_pipeline",
      "input_summary": "The impact of LLMs on backend engineering workflows",
      "output_summary": "Pipeline complete: 5 sub-questions researched, results aggregated and written into final report",
      "output": "Pipeline complete: 5 sub-questions researched, results aggregated and written into final report",
      "duration_ms": 18420.5
    },
    {
      "agent": "Planner",
      "action": "decompose_topic",
      "input_summary": "The impact of LLMs on backend engineering workflows",
      "output_summary": "Generated 5 sub-questions from topic",
      "output": "Generated 5 sub-questions from topic",
      "duration_ms": 1230.1
    },
    {
      "agent": "Researcher",
      "action": "research_sub_question",
      "input_summary": "What AI-based coding tools are most commonly used by backend engineers?",
      "output_summary": "- GitHub Copilot adoption: 55% of developers...",
      "output": "- GitHub Copilot adoption: 55% of developers...",
      "duration_ms": 4210.8
    }
  ],
  "duration_ms": 18420.5,
  "meta": { "duration_ms": 18420.5 }
}
```

The `agent_trace` contains one entry per agent action. Researcher entries run concurrently — their individual `duration_ms` values will be similar to each other but far less than the total `duration_ms`, which proves the parallel execution.

## Design Decisions

**LangChain primitives, not the agent framework** — All agents use LangChain's typed message and model classes (`AzureChatOpenAI`, `SystemMessage`, `HumanMessage`, `ToolMessage`, `@tool`, `bind_tools`, `with_structured_output`). The `AgentExecutor` / LangGraph stack is intentionally *not* used — the orchestration loop in `ManagerAgent` and the tool-calling loop in `ResearcherAgent` are written explicitly. This keeps the code transparent: every step the system takes is visible in the source.

**DuckDuckGo for web search** — No extra API key is required and the tool returns compact result snippets that fit the agent's tool loop. The code wraps the blocking search call in `asyncio.to_thread()` so the async pipeline stays responsive.

**`asyncio.gather` for concurrent research** — All Researcher coroutines are scheduled simultaneously. `ResearcherAgent.run()` keeps state in local variables only, making concurrent calls on a single instance safe.

**`pydantic-settings` for config** — Missing required environment variables raise a clear `ValidationError` at startup with field names listed, rather than surfacing as a `None`-dereference deep in an LLM call.

**Structured logging** — JSON-formatted log lines via `python-json-logger`. Each agent has its own logger (`tectika.planner`, `tectika.researcher`, etc.) so log lines are easy to grep by agent.

## Project Structure

```
tectika/
├── agents/
│   ├── manager.py       # Orchestrator — asyncio.gather + trace assembly
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
    └── routes.py        # POST /run
main.py                  # FastAPI entry point
docs/
└── agent-architecture.md # Original design document
```
