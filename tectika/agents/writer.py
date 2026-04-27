import logging
import time

from openai import AsyncAzureOpenAI

from tectika.core.config import settings
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
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client

    async def run(self, consolidated_findings: str) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("writer_start")

        response = await self._client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": consolidated_findings},
            ],
            temperature=0.5,
        )

        report = response.choices[0].message.content or ""
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info("writer_complete", extra={"duration_ms": duration_ms})

        summary = "Produced structured professional report from consolidated research"
        trace = TraceEntry(
            agent="Writer",
            action="generate_report",
            input_summary="Consolidated research document",
            output_summary=summary,
            output=summary,
            duration_ms=duration_ms,
        )
        return report, trace
