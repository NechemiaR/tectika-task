from openai import AsyncAzureOpenAI

from tectika.core.config import settings


def get_async_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
