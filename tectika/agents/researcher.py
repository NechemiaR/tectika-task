import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from tectika.core.config import settings
from tectika.models.schemas import TraceEntry
from tectika.tools.web_search import tavily_search

logger = logging.getLogger("tectika.researcher")

_SYSTEM_PROMPT = (
    "You are a research assistant. Your goal is to answer the given research question "
    "with specific, factual information. Use the web_search tool to gather current data "
    "before formulating your answer. Present findings as clear bullet points or short "
    "paragraphs containing only facts, data, and key points — no filler."
)

_MAX_ITERATIONS = 5


@tool
async def web_search(query: str) -> str:
    """Search the web for current, factual information on a topic."""
    try:
        results = await tavily_search(query)
        if not results:
            return f"No results found for: {query}"
        return "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in results
        )
    except RuntimeError:
        return (
            f"[Web search unavailable — no TAVILY_API_KEY set. "
            f"Provide your best answer from training knowledge for: {query}]"
        )
    except Exception as exc:
        return f"[Search failed: {exc}. Continue with available knowledge.]"


class ResearcherAgent:
    def __init__(self) -> None:
        llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment_name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key,
            temperature=0.2,
        )
        self._llm = llm.bind_tools([web_search])

    async def run(self, sub_question: str) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("researcher_start", extra={"sub_question": sub_question})

        messages: list = [SystemMessage(_SYSTEM_PROMPT), HumanMessage(sub_question)]

        for _ in range(_MAX_ITERATIONS):
            response: AIMessage = await self._llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                logger.info("researcher_tool_call", extra={"query": tc["args"].get("query")})
                result = await web_search.ainvoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        findings: str = response.content or ""  # type: ignore[union-attr]
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info("researcher_complete", extra={"duration_ms": duration_ms})

        summary = findings[:400]
        trace = TraceEntry(
            agent="Researcher",
            action="research_sub_question",
            input_summary=sub_question[:300],
            output_summary=summary,
            output=summary,
            duration_ms=duration_ms,
        )
        return findings, trace
