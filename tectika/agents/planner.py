import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from tectika.core.config import settings
from tectika.models.schemas import PlannerOutput, TraceEntry

logger = logging.getLogger("tectika.planner")

_SYSTEM_PROMPT = (
    "You are a research strategist. Given a research topic, produce a structured list "
    "of focused, non-overlapping research questions. Aim for 3 to 6 questions that "
    "together provide comprehensive coverage of the topic. Each question should be "
    "specific enough to guide targeted research."
)


class PlannerAgent:
    def __init__(self) -> None:
        llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment_name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_api_key,
            temperature=0.3,
        )
        self._llm = llm.with_structured_output(PlannerOutput)

    async def run(self, topic: str) -> tuple[list[str], TraceEntry]:
        start = time.monotonic()
        logger.info("planner_start", extra={"topic": topic})

        result: PlannerOutput = await self._llm.ainvoke([  # type: ignore[assignment]
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(topic),
        ])
        sub_questions = result.sub_questions

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "planner_complete",
            extra={"sub_question_count": len(sub_questions), "duration_ms": duration_ms},
        )

        summary = f"Generated {len(sub_questions)} sub-questions from topic"
        trace = TraceEntry(
            agent="Planner",
            action="decompose_topic",
            input_summary=topic[:300],
            output_summary=summary,
            output=summary,
            duration_ms=duration_ms,
        )
        return sub_questions, trace
