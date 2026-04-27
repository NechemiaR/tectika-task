import asyncio
import logging
import time
from typing import AsyncIterator

from tectika.agents.aggregator import AggregatorAgent
from tectika.agents.planner import PlannerAgent
from tectika.agents.researcher import ResearcherAgent
from tectika.agents.writer import WriterAgent
from tectika.models.schemas import MetaInfo, RunResponse, TokenUsage, TraceEntry

logger = logging.getLogger("tectika.manager")


class ManagerAgent:
    def __init__(self) -> None:
        self._planner = PlannerAgent()
        self._researcher = ResearcherAgent()
        self._aggregator = AggregatorAgent()
        self._writer = WriterAgent()

    async def stream(self, topic: str) -> AsyncIterator[dict]:
        """Run the pipeline and yield events as agents complete.

        Event shapes:
          {"type": "trace",    "data": <TraceEntry dict>}
          {"type": "complete", "data": <RunResponse dict>}
          {"type": "error",    "data": {"message": str}}
        """
        pipeline_start = time.monotonic()
        logger.info("manager_start", extra={"topic": topic})

        try:
            # Step 1 — Decompose
            sub_questions, planner_trace = await self._planner.run(topic)
            yield {"type": "trace", "data": planner_trace.model_dump()}
            logger.info("manager_planning_done", extra={"sub_question_count": len(sub_questions)})

            # Step 2 — Research concurrently, emit each as it finishes
            research_tasks = [
                asyncio.create_task(self._researcher.run(q)) for q in sub_questions
            ]
            researcher_traces: list[TraceEntry] = []
            findings_texts: list[str] = []
            for coro in asyncio.as_completed(research_tasks):
                findings, r_trace = await coro
                findings_texts.append(findings)
                researcher_traces.append(r_trace)
                yield {"type": "trace", "data": r_trace.model_dump()}
            logger.info("manager_research_done", extra={"researcher_count": len(researcher_traces)})

            # Step 3 — Aggregate
            consolidated, aggregator_trace = await self._aggregator.run(findings_texts)
            yield {"type": "trace", "data": aggregator_trace.model_dump()}

            # Step 4 — Write
            report, writer_trace = await self._writer.run(consolidated)
            yield {"type": "trace", "data": writer_trace.model_dump()}

            total_ms = round((time.monotonic() - pipeline_start) * 1000, 1)
            total_tokens = (
                planner_trace.tokens
                + sum((t.tokens for t in researcher_traces), TokenUsage())
                + aggregator_trace.tokens
                + writer_trace.tokens
            )

            manager_summary = (
                f"Pipeline complete: {len(sub_questions)} sub-questions researched, "
                "results aggregated and written into final report"
            )
            manager_trace = TraceEntry(
                agent="Manager",
                action="orchestrate_pipeline",
                input_summary=topic[:300],
                output_summary=manager_summary,
                output=manager_summary,
                duration_ms=total_ms,
                tokens=TokenUsage(),
            )
            yield {"type": "trace", "data": manager_trace.model_dump()}

            response = RunResponse(
                report=report,
                agent_trace=[
                    manager_trace,
                    planner_trace,
                    *researcher_traces,
                    aggregator_trace,
                    writer_trace,
                ],
                meta=MetaInfo(duration_ms=total_ms, tokens=total_tokens),
            )
            logger.info(
                "manager_complete",
                extra={
                    "duration_ms": total_ms,
                    "input_tokens": total_tokens.input_tokens,
                    "output_tokens": total_tokens.output_tokens,
                },
            )
            yield {"type": "complete", "data": response.model_dump()}

        except Exception as exc:
            logger.error("manager_error", extra={"error": str(exc)}, exc_info=True)
            yield {"type": "error", "data": {"message": "Internal pipeline error"}}
            raise

    async def run(self, topic: str) -> RunResponse:
        async for event in self.stream(topic):
            if event["type"] == "complete":
                return RunResponse(**event["data"])
        raise RuntimeError("pipeline did not produce a final response")
