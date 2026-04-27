import json
import logging
import time

from openai import AsyncAzureOpenAI

from tectika.core.config import settings
from tectika.models.schemas import TraceEntry
from tectika.tools.web_search import WEB_SEARCH_TOOL_DEFINITION, tavily_search

logger = logging.getLogger("tectika.researcher")

_SYSTEM_PROMPT = (
    "You are a research assistant. Your goal is to answer the given research question "
    "with specific, factual information. Use the web_search tool to gather current data "
    "before formulating your answer. Present findings as clear bullet points or short "
    "paragraphs containing only facts, data, and key points — no filler."
)

_MAX_TOOL_LOOPS = 5


class ResearcherAgent:
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client

    async def run(self, sub_question: str) -> tuple[str, TraceEntry]:
        """
        Executes an agentic loop: the model autonomously decides when to call the
        web_search tool, retrieves results, and iterates until it produces a final answer.
        All state is local — this method is safe to call concurrently on a single instance.
        """
        start = time.monotonic()
        logger.info("researcher_start", extra={"sub_question": sub_question})

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": sub_question},
        ]

        findings = ""
        loops = 0

        while loops < _MAX_TOOL_LOOPS:
            loops += 1
            response = await self._client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=messages,
                tools=[WEB_SEARCH_TOOL_DEFINITION],
                tool_choice="auto",
                temperature=0.2,
            )

            msg = response.choices[0].message
            # Append assistant turn (may include tool_calls)
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                # Model produced a final answer — exit loop
                findings = msg.content or ""
                break

            # Execute each requested tool call and feed results back
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                query: str = args.get("query", sub_question)

                logger.info("researcher_tool_call", extra={"query": query, "loop": loops})
                result_text = await self._execute_search(query)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )
        else:
            # Safety fallback: ask for a final answer if loop limit hit
            messages.append(
                {"role": "user", "content": "Summarise what you know so far in bullet points."}
            )
            final = await self._client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=messages,
                temperature=0.2,
            )
            findings = final.choices[0].message.content or ""

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "researcher_complete",
            extra={"sub_question": sub_question, "loops": loops, "duration_ms": duration_ms},
        )

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

    async def _execute_search(self, query: str) -> str:
        """Run a Tavily search and format results for the LLM, or return a fallback note."""
        try:
            results = await tavily_search(query)
            if not results:
                return f"No results found for: {query}"
            lines = [f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in results]
            return "\n".join(lines)
        except RuntimeError:
            # TAVILY_API_KEY not set — model will reason from training data
            return (
                f"[Web search unavailable — no TAVILY_API_KEY set. "
                f"Provide your best answer from training knowledge for: {query}]"
            )
        except Exception as exc:
            logger.warning("researcher_search_error", extra={"error": str(exc), "query": query})
            return f"[Search failed: {exc}. Continue with available knowledge.]"
