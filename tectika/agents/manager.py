import asyncio
import logging
import time

from openai import AsyncAzureOpenAI

from tectika.agents.aggregator import AggregatorAgent
from tectika.agents.planner import PlannerAgent
from tectika.agents.researcher import ResearcherAgent
from tectika.agents.writer import WriterAgent
from tectika.models.schemas import MetaInfo, RunResponse, TraceEntry

logger = logging.getLogger("tectika.manager")


class ManagerAgent:
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client
        self._planner = PlannerAgent(client=client)
        self._researcher = ResearcherAgent(client=client)
        self._aggregator = AggregatorAgent(client=client)
        self._writer = WriterAgent(client=client)

    async def run(self, topic: str) -> RunResponse:
        pipeline_start = time.monotonic()
        logger.info("manager_start", extra={"topic": topic})

        # Step 1 — Decompose topic into sub-questions
        sub_questions, planner_trace = await self._planner.run(topic)
        logger.info(
            "manager_planning_done",
            extra={"sub_question_count": len(sub_questions)},
        )

        # Step 2 — Research all sub-questions concurrently
        # ResearcherAgent.run() is stateless (all state in local variables),
        # so calling it concurrently on one instance is safe.
        research_tasks = [self._researcher.run(q) for q in sub_questions]
        research_outputs: list[tuple[str, TraceEntry]] = await asyncio.gather(*research_tasks)

        findings_texts: list[str] = []
        researcher_traces: list[TraceEntry] = []
        for findings, r_trace in research_outputs:
            findings_texts.append(findings)
            researcher_traces.append(r_trace)

        logger.info(
            "manager_research_done",
            extra={"researcher_count": len(researcher_traces)},
        )

        # Step 3 — Aggregate all findings into one deduplicated document
        consolidated, aggregator_trace = await self._aggregator.run(findings_texts)

        # Step 4 — Write the final report
        report, writer_trace = await self._writer.run(consolidated)

        total_ms = round((time.monotonic() - pipeline_start) * 1000, 1)
        logger.info("manager_complete", extra={"duration_ms": total_ms})

        manager_trace = TraceEntry(
            agent="Manager",
            action="orchestrate_pipeline",
            input_summary=topic[:300],
            output_summary=(
                f"Pipeline complete: {len(sub_questions)} sub-questions researched, "
                f"results aggregated and written into final report"
            ),
            output=(
                f"Pipeline complete: {len(sub_questions)} sub-questions researched, "
                f"results aggregated and written into final report"
            ),
            duration_ms=total_ms,
        )

        # Manager trace first (matches PDF example order), then execution order
        full_trace = [manager_trace, planner_trace, *researcher_traces, aggregator_trace, writer_trace]

        return RunResponse(
            report=report,
            agent_trace=full_trace,
            duration_ms=total_ms,
            meta=MetaInfo(duration_ms=total_ms),
        )
