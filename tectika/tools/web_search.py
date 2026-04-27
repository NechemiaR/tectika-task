import asyncio
import logging

from ddgs import DDGS

logger = logging.getLogger("tectika.tools.web_search")


def _duckduckgo_search_sync(query: str, max_results: int) -> list[dict]:
    results = DDGS().text(query, max_results=max_results, backend="lite")
    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "content": result.get("body", ""),
        }
        for result in results
    ]


async def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return a list of {title, url, content} dicts."""
    try:
        return await asyncio.to_thread(_duckduckgo_search_sync, query, max_results)
    except Exception as exc:
        logger.warning("duckduckgo_search_failed", extra={"query": query, "error": str(exc)})
        return []
