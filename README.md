# Tectika Multi-Agent Research System

A Python backend service that uses a 5-agent pipeline to research any topic and produce a structured professional report.

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
      ├──► ResearcherAgent(sub_question_1)  [tool-calling agentic loop]
      ├──► ResearcherAgent(sub_question_2)  [tool-calling agentic loop]
      └──► ResearcherAgent(sub_question_N)  [tool-calling agentic loop]
      │
      │ 3. aggregate
      ▼
 AggregatorAgent  →  consolidated, deduplicated knowledge document
      │
      │ 4. write
      ▼
  WriterAgent  →  final polished report
```

| Agent | Role | LLM Mode |
|---|---|---|
| Manager | Orchestrates the pipeline; assembles `agent_trace` and `RunResponse` | No direct LLM call |
| Planner | Decomposes research topic into 3–6 focused sub-questions | JSON structured output |
| Researcher | Gathers facts per sub-question; runs concurrently | Tool-calling agentic loop (web search) |
| Aggregator | Merges and deduplicates all researcher outputs | Single completion |
| Writer | Formats consolidated findings into a professional report | Single completion |

## Prerequisites

- Python 3.10+
- Azure OpenAI deployment (GPT-4o via Azure AI Foundry)
- (Optional) [Tavily API key](https://tavily.com/) for real web search — without it the Researcher falls back to GPT-4o's training knowledge

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd tectika-task

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your Azure credentials (see Environment Variables below)

# 5. Start the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

## Docker

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up --build -d
```

Logs are JSON-structured and stream to stdout.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure endpoint URL (e.g. `https://your-resource.openai.azure.com/`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | GPT-4o deployment name in Azure AI Foundry |
| `AZURE_OPENAI_API_VERSION` | Yes | API version (default: `2024-02-01`) |
| `TAVILY_API_KEY` | No | Enables real web search in ResearcherAgent |

## API Usage

### `POST /run`

**Request:**

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "The impact of LLMs on backend engineering workflows"}'
```

**Response:**

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
  "meta": {
    "duration_ms": 18420.5
  }
}
```

The `agent_trace` contains one entry per agent action. Researcher entries run concurrently — their individual `duration_ms` values will be similar to each other but far less than the total `duration_ms`, which proves the parallel execution.

## Design Decisions

**No LangChain** — Direct Azure OpenAI SDK (`openai` package, `AsyncAzureOpenAI`) keeps the code readable and makes the orchestration logic explicit. LangChain's abstractions would obscure the agentic loop in the Researcher.

**Tavily for web search** — Purpose-built for LLM use: returns pre-chunked, relevant excerpts rather than raw HTML. One REST call, no Azure-specific setup required.

**`asyncio.gather` for concurrent research** — All Researcher coroutines are scheduled simultaneously. `ResearcherAgent.run()` is fully stateless (no instance state), making concurrent calls on one instance safe.

**`pydantic-settings` for config** — Missing required environment variables raise a clear `ValidationError` at startup with field names listed, rather than surfacing as a `None`-dereference deep in an LLM call.
