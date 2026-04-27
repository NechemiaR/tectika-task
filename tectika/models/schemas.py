from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic to investigate")


class TraceEntry(BaseModel):
    agent: str
    action: str
    input_summary: str
    output_summary: str
    output: str          # mirrors output_summary; satisfies the PDF contract field name
    duration_ms: float


class MetaInfo(BaseModel):
    duration_ms: float


class RunResponse(BaseModel):
    report: str
    agent_trace: list[TraceEntry]
    duration_ms: float   # flat field per agent instructions contract
    meta: MetaInfo       # nested field per PDF contract


class PlannerOutput(BaseModel):
    sub_questions: list[str]
