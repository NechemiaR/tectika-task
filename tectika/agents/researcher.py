import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from tectika.core.llm import coerce_text, extract_tokens, make_chat_model
from tectika.models.schemas import TokenUsage, TraceEntry
from tectika.tools.web_search import duckduckgo_search

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
        results = await duckduckgo_search(query)
        if not results:
            return f"No results found for: {query}"
        return "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in results
        )
    except Exception as exc:
        return f"[Search failed: {exc}. Continue with available knowledge.]"


class ResearcherAgent:
    def __init__(self) -> None:
        self._llm = make_chat_model(temperature=0.2).bind_tools([web_search])

    async def run(self, sub_question: str) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("researcher_start", extra={"sub_question": sub_question})

        messages: list[BaseMessage] = [
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(sub_question),
        ]
        total_tokens = TokenUsage()
        response: AIMessage | None = None

        for iteration in range(_MAX_ITERATIONS):
            response = await self._llm.ainvoke(messages)
            total_tokens += extract_tokens(response)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                logger.info("researcher_tool_call", extra={"query": tc["args"].get("query")})
                result = await web_search.ainvoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # Tool budget exhausted — force one final answer turn.
            messages.append(
                SystemMessage(
                    "Tool budget exhausted. Answer the original question now using only "
                    "the search results already returned and your existing knowledge."
                )
            )
            response = await make_chat_model(temperature=0.2).ainvoke(messages)
            total_tokens += extract_tokens(response)

        findings = coerce_text(response.content) if response else ""
        if not findings:
            findings = "[No findings produced — research budget exhausted.]"

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "researcher_complete",
            extra={
                "duration_ms": duration_ms,
                "input_tokens": total_tokens.input_tokens,
                "output_tokens": total_tokens.output_tokens,
            },
        )

        trace = TraceEntry(
            agent="Researcher",
            action="research_sub_question",
            input_summary=sub_question[:300],
            output_summary=findings[:400],
            output=findings,
            duration_ms=duration_ms,
            tokens=total_tokens,
        )
        return findings, trace
