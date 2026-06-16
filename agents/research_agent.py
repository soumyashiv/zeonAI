"""
ZEON Research Agent
Performs multi-step web research: search → fetch → summarise → store.
Uses DuckDuckGo (zero cost) + Crawl4AI fallback.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat
from tools.search import quick_search, deep_research
from brains.memory import get_memory_brain

log = structlog.get_logger(__name__)

SUMMARISE_PROMPT = """You are a research summariser. 
Given raw web search results, extract the key facts relevant to the query.
Be concise. Cite sources. Respond as JSON:
{
  "query": "...",
  "key_facts": ["fact1", "fact2", "..."],
  "summary": "...",
  "confidence": 0.8,
  "sources": ["url1", "url2"]
}"""


class ResearchAgent(BaseAgent):
    name = "research_agent"
    description = "Web research specialist. Searches, fetches, summarises, stores findings."

    async def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        mem = get_memory_brain()
        query = payload.get("task", payload.get("query", ""))
        prior = await mem.recall(query, limit=3)
        return {"prior_knowledge": prior, "query": query}

    async def understand(self, task: str, context: dict) -> dict[str, Any]:
        return {"query": task, "context": context}

    async def plan(self, task: str, understanding: dict, knowledge: dict) -> dict[str, Any]:
        query = understanding.get("query", task)
        prior = understanding.get("context", {}).get("prior_knowledge", {})

        # Check if we already have fresh knowledge
        if prior.get("semantic") and any(
            h.get("score", 0) > 0.85 for h in prior["semantic"]
        ):
            return {"query": query, "use_cache": True, "cached": prior}

        return {
            "query": query,
            "use_cache": False,
            "steps": ["search", "fetch_top", "summarise", "store"],
        }

    async def critique(self, plan: dict) -> tuple[int, str]:
        if plan.get("use_cache"):
            return 9, "Using cached knowledge — high confidence match"
        if plan.get("query"):
            return 8, "Fresh research plan ready"
        return 4, "No query specified"

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        query = plan.get("query", "")

        if plan.get("use_cache"):
            result = {
                "query": query,
                "source": "memory_cache",
                "cached_hits": len(plan.get("cached", {}).get("semantic", [])),
            }
            await self._reply(original_message, result)
            return result

        self._log.info("research.starting", query=query[:60])

        # Step 1: Search
        snippets = await quick_search(query, max_results=5)
        self._log.info("research.searched", results=len(snippets))

        # Step 2: Summarise with LLM
        search_text = "\n".join(snippets[:5])
        messages = [
            {"role": "system", "content": SUMMARISE_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{search_text}"},
        ]

        try:
            raw = await chat(messages, temperature=0.3, max_tokens=600)
            s, e = raw.find("{"), raw.rfind("}") + 1
            summary = json.loads(raw[s:e]) if s >= 0 and e > s else {
                "query": query, "key_facts": snippets[:3],
                "summary": search_text[:300], "confidence": 0.5, "sources": []
            }
        except Exception as ex:
            self._log.warning("research.summarise_failed", error=str(ex))
            summary = {
                "query": query, "key_facts": snippets[:3],
                "summary": search_text[:300], "confidence": 0.4, "sources": []
            }

        # Step 3: Store findings in memory
        mem = get_memory_brain()
        importance = summary.get("confidence", 0.5)
        await mem.remember(
            f"Research: {query} → {summary.get('summary', '')[:200]}",
            agent=self.name,
            importance=importance,
            tags=["research", "web"],
        )

        result = {"query": query, "summary": summary, "raw_snippets": snippets}
        self._log.info("research.complete",
                       facts=len(summary.get("key_facts", [])),
                       confidence=summary.get("confidence", 0))

        await self._reply(original_message, result)
        return result

    async def _reply(self, original: AgentMessage, result: dict) -> None:
        if original.from_agent:
            await self.send(
                original.from_agent,
                MessageType.TASK_RESULT,
                {"research_result": result},
                priority=7,
            )

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        if not isinstance(result, dict):
            return False, "no result"
        has_summary = bool(result.get("summary"))
        return has_summary, f"research complete, confidence={result.get('summary', {}).get('confidence', '?')}"
