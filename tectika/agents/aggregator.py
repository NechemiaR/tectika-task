import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from tectika.core.llm import coerce_text, extract_tokens, make_chat_model
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
    def __init__(self) -> None:
        self._llm = make_chat_model(temperature=0.2)

    async def run(self, research_results: list[str]) -> tuple[str, TraceEntry]:
        start = time.monotonic()
        logger.info("aggregator_start", extra={"result_count": len(research_results)})

        numbered = "\n\n".join(
            f"[Research Batch {i + 1}]\n{text}" for i, text in enumerate(research_results)
        )

        response = await self._llm.ainvoke([
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(numbered),
        ])
        consolidated = coerce_text(response.content)
        tokens = extract_tokens(response)

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "aggregator_complete",
            extra={
                "duration_ms": duration_ms,
                "input_tokens": tokens.input_tokens,
                "output_tokens": tokens.output_tokens,
            },
        )

        trace = TraceEntry(
            agent="Aggregator",
            action="merge_and_deduplicate",
            input_summary=f"{len(research_results)} researcher outputs",
            output_summary=f"Consolidated {len(research_results)} research batches into unified document",
            output=consolidated,
            duration_ms=duration_ms,
            tokens=tokens,
        )
        return consolidated, trace
