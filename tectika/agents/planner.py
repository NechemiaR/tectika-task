import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from tectika.core.llm import extract_tokens, make_chat_model
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
        self._llm = make_chat_model(temperature=0.3).with_structured_output(
            PlannerOutput, include_raw=True
        )

    async def run(self, topic: str) -> tuple[list[str], TraceEntry]:
        start = time.monotonic()
        logger.info("planner_start", extra={"topic": topic})

        result = await self._llm.ainvoke([
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(topic),
        ])
        parsed: PlannerOutput = result["parsed"]
        tokens = extract_tokens(result["raw"])
        sub_questions = parsed.sub_questions

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "planner_complete",
            extra={
                "sub_question_count": len(sub_questions),
                "duration_ms": duration_ms,
                "input_tokens": tokens.input_tokens,
                "output_tokens": tokens.output_tokens,
            },
        )

        full_output = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
        trace = TraceEntry(
            agent="Planner",
            action="decompose_topic",
            input_summary=topic[:300],
            output_summary=f"Generated {len(sub_questions)} sub-questions from topic",
            output=full_output,
            duration_ms=duration_ms,
            tokens=tokens,
        )
        return sub_questions, trace
