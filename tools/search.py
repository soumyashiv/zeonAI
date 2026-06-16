"""
JARVIS Search Tool
DuckDuckGo search (zero cost, no API key) + Crawl4AI web extraction.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

log = structlog.get_logger(__name__)


async def quick_search(query: str, max_results: int = 5) -> list[str]:
    """
    DuckDuckGo text search. Returns list of 'Title: snippet' strings.
    Falls back to empty list if duckduckgo-search not installed.
    """
    try:
        from duckduckgo_search import DDGS
        loop = asyncio.get_event_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await loop.run_in_executor(None, _search)
        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")[:200]
            href = r.get("href", "")
            snippets.append(f"{title}: {body} [{href}]")
        return snippets

    except ImportError:
        log.warning("search.duckduckgo_not_installed")
        return [f"[Search unavailable — install duckduckgo-search]"]
    except Exception as e:
        log.error("search.failed", error=str(e))
        return [f"[Search error: {str(e)[:100]}]"]


async def crawl_page(url: str) -> str:
    """
    Extract clean text from a URL using Crawl4AI.
    Falls back to httpx + basic HTML strip if Crawl4AI not available.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown[:3000] if result.success else ""
    except ImportError:
        log.warning("search.crawl4ai_not_installed", fallback="httpx")
        return await _httpx_fetch(url)
    except Exception as e:
        log.error("search.crawl_failed", url=url, error=str(e))
        return ""


async def _httpx_fetch(url: str) -> str:
    """Minimal fallback: fetch URL and strip HTML tags."""
    try:
        import httpx
        import re
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, follow_redirects=True)
            html = resp.text
            clean = re.sub(r"<[^>]+>", " ", html)
            clean = " ".join(clean.split())
            return clean[:3000]
    except Exception as e:
        return f"[Fetch error: {str(e)[:100]}]"


async def deep_research(topic: str, max_pages: int = 3) -> dict[str, Any]:
    """
    Multi-step research: search → fetch top pages → summarise.
    Returns structured research result.
    """
    snippets = await quick_search(topic, max_results=max_pages + 2)
    pages_content = []

    # Extract URLs from snippets and fetch top pages
    import re
    urls = re.findall(r"\[https?://[^\]]+\]", " ".join(snippets))
    urls = [u.strip("[]") for u in urls[:max_pages]]

    for url in urls:
        content = await crawl_page(url)
        if content:
            pages_content.append({"url": url, "content": content[:1500]})

    return {
        "topic": topic,
        "search_snippets": snippets,
        "pages": pages_content,
        "total_sources": len(snippets) + len(pages_content),
    }
