import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from tectika.agents.manager import ManagerAgent
from tectika.models.schemas import RunRequest, RunResponse

router = APIRouter()
logger = logging.getLogger("tectika.api")


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 with no external dependencies."""
    return {"status": "ok"}


@router.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest) -> RunResponse:
    logger.info("request_received", extra={"topic": request.topic})
    manager = ManagerAgent()
    try:
        result = await manager.run(topic=request.topic)
    except Exception as exc:
        logger.error("pipeline_error", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error") from exc
    logger.info("request_complete", extra={"duration_ms": result.meta.duration_ms})
    return result


@router.post("/run/stream")
async def run_pipeline_stream(request: RunRequest) -> StreamingResponse:
    """Server-Sent Events endpoint.

    Each event is a single JSON object on a `data:` line, terminated by a blank line.
    Event types: `trace` (one per agent completion, in completion order),
    `complete` (final RunResponse), `error` (on failure).
    """
    logger.info("stream_request_received", extra={"topic": request.topic})
    manager = ManagerAgent()

    async def event_source():
        try:
            async for event in manager.stream(topic=request.topic):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            # Manager already logged + emitted an error event; stop the stream cleanly.
            return

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
