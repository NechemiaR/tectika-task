import httpx

from tectika.core.config import settings


async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Call Tavily Search API and return a list of {title, url, content} dicts."""
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
