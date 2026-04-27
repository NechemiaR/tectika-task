import uvicorn
from fastapi import FastAPI

from tectika.api.routes import router
from tectika.core.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Tectika Multi-Agent Research System",
    description="5-agent pipeline: Planner → Researchers (concurrent) → Aggregator → Writer",
    version="1.0.0",
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
