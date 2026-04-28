from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from tectika.api.routes import router
from tectika.core.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Tectika Multi-Agent Research System",
    description="5-agent pipeline: Planner → Researchers (concurrent) → Aggregator → Writer",
    version="1.0.0",
)
app.include_router(router)

_INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page frontend that drives POST /run."""
    return FileResponse(_INDEX_HTML, media_type="text/html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
