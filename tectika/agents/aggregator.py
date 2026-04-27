import logging
import time

from openai import AsyncAzureOpenAI

from tectika.core.config import settings
from tectika.models.schemas import TraceEntry

logger = logging.getLogger("tectika.aggregator")

_SYSTEM_PROMPT = (
    "You are a research synthesizer. You will receive multiple sets of research findings "
    "on related sub-topics. Your task is to:\n"
    "1. Merge all findings into a single, coherent knowledge document.\n"
    "2. Remove duplicate information — keep only one copy of any repeated fact.\n"
    "3. Identify connections between findings across different sub-topics.\n"
    "4. Organise the result thematically with clear section headings.\n"
    "Preserve every unique fact and data point. Do not summarise away detail."
)


class AggregatorAgent:
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client

    async def run(self, research_results: list[str]) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("aggregator_start", extra={"result_count": len(research_results)})

        numbered = "\n\n".join(
            f"[Research Batch {i + 1}]\n{text}" for i, text in enumerate(research_results)
        )

        response = await self._client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
            temperature=0.2,
        )

        consolidated = response.choices[0].message.content or ""
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info("aggregator_complete", extra={"duration_ms": duration_ms})

        summary = f"Consolidated {len(research_results)} research batches into unified document"
        trace = TraceEntry(
            agent="Aggregator",
            action="merge_and_deduplicate",
            input_summary=f"{len(research_results)} researcher outputs",
            output_summary=summary,
            output=summary,
            duration_ms=duration_ms,
        )
        return consolidated, trace
