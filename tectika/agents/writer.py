import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from tectika.core.llm import coerce_text, extract_tokens, make_chat_model
from tectika.models.schemas import TraceEntry

logger = logging.getLogger("tectika.writer")

_SYSTEM_PROMPT = (
    "You are a professional content editor producing a research report. Given a set of "
    "consolidated research notes, write a polished, structured report that includes:\n"
    "- A clear, descriptive title\n"
    "- An executive summary (2-3 sentences)\n"
    "- Thematic body sections with bold headers\n"
    "- A conclusion with key takeaways\n"
    "Write in a professional, authoritative tone. Use the facts provided — do not add "
    "information not present in the input."
)


class WriterAgent:
    def __init__(self) -> None:
        self._llm = make_chat_model(temperature=0.5)

    async def run(self, consolidated_findings: str) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("writer_start")

        response = await self._llm.ainvoke([
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(consolidated_findings),
        ])
        report = coerce_text(response.content)
        tokens = extract_tokens(response)

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "writer_complete",
            extra={
                "duration_ms": duration_ms,
                "input_tokens": tokens.input_tokens,
                "output_tokens": tokens.output_tokens,
            },
        )

        trace = TraceEntry(
            agent="Writer",
            action="generate_report",
            input_summary="Consolidated research document",
            output_summary="Produced structured professional report from consolidated research",
            output=report,
            duration_ms=duration_ms,
            tokens=tokens,
        )
        return report, trace
