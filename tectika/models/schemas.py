from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic to investigate")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class TraceEntry(BaseModel):
    agent: str
    action: str
    input_summary: str
    output_summary: str
    output: str
    duration_ms: float
    tokens: TokenUsage = TokenUsage()


class MetaInfo(BaseModel):
    duration_ms: float
    tokens: TokenUsage = TokenUsage()


class RunResponse(BaseModel):
    report: str
    agent_trace: list[TraceEntry]
    meta: MetaInfo


class PlannerOutput(BaseModel):
    sub_questions: list[str]
