import json
import logging
import time

from openai import AsyncAzureOpenAI

from tectika.core.config import settings
from tectika.models.schemas import PlannerOutput, TraceEntry

logger = logging.getLogger("tectika.planner")

_SYSTEM_PROMPT = (
    "You are a research strategist. Given a research topic, produce a JSON object "
    "with a single key 'sub_questions' containing a list of focused, non-overlapping "
    "research questions. Aim for 3 to 6 questions that together provide comprehensive "
    "coverage of the topic. Each question should be specific enough to guide targeted research."
)


class PlannerAgent:
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client

    async def run(self, topic: str) -> tuple[list[str], TraceEntry]:
        start = time.monotonic()
        logger.info("planner_start", extra={"topic": topic})

        response = await self._client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": topic},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = PlannerOutput.model_validate(json.loads(raw))
        sub_questions = parsed.sub_questions

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
