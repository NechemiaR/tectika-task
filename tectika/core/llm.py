from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI

from tectika.core.config import settings
from tectika.models.schemas import TokenUsage


def make_chat_model(temperature: float) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
        temperature=temperature,
    )


def extract_tokens(message: AIMessage | Any) -> TokenUsage:
    usage = getattr(message, "usage_metadata", None) or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
    )


def coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return ""
