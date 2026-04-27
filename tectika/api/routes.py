import logging

from fastapi import APIRouter, HTTPException

from tectika.agents.manager import ManagerAgent
from tectika.core.llm_client import get_async_client
from tectika.models.schemas import RunRequest, RunResponse

router = APIRouter()
logger = logging.getLogger("tectika.api")


@router.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest) -> RunResponse:
    logger.info("request_received", extra={"topic": request.topic})
    client = get_async_client()
    manager = ManagerAgent(client=client)
    try:
        result = await manager.run(topic=request.topic)
    except Exception as exc:
        logger.error("pipeline_error", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    logger.info("request_complete", extra={"duration_ms": result.duration_ms})
    return result
