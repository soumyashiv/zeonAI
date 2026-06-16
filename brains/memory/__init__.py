"""
JARVIS Memory Brain — Unified API
Single entry point for all 5 memory types.
All agents call this instead of individual memory modules.
"""
from __future__ import annotations

from typing import Any

import structlog

from brains.memory.working import WorkingMemory, get_working_memory
from brains.memory.episodic import EpisodicMemory, get_episodic_memory
from brains.memory.semantic import SemanticMemory, SemanticEntry, get_semantic_memory
from brains.memory.procedural import ProceduralMemory, get_procedural_memory
from brains.memory.knowledge_graph import KnowledgeGraph, GraphNode, get_knowledge_graph

log = structlog.get_logger(__name__)


class MemoryBrain:
    """
    Unified facade for all JARVIS memory systems.

    Memory types:
        working     — Fast, ephemeral session context (Redis/in-memory)
        episodic    — Long-term experience log (SQLite)
        semantic    — Vector knowledge store (Qdrant/in-process)
        procedural  — Compiled skills registry (SQLite)
        graph       — Knowledge graph (Neo4j/NetworkX)
    """

    def __init__(self) -> None:
        self.working: WorkingMemory = get_working_memory()
        self.episodic: EpisodicMemory = get_episodic_memory()
        self.semantic: SemanticMemory = get_semantic_memory()
        self.procedural: ProceduralMemory = get_procedural_memory()
        self.graph: KnowledgeGraph = get_knowledge_graph()
        log.info("memory_brain.initialized")

    # ── Convenience: remember anything ───────────────────────────────────────

    async def remember(
        self,
        text: str,
        *,
        agent: str = "system",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Store a piece of knowledge across all relevant memory types.
        Returns a dict of created IDs.
        """
        ids: dict[str, str] = {}

        # Semantic (vector)
        sem_id = await self.semantic.store(
            text,
            metadata={"agent": agent, "importance": importance, "tags": tags or []},
        )
        ids["semantic_id"] = sem_id

        # Episodic (experience record)
        ep_id = await self.episodic.add(
            agent=agent,
            task_description="memory.remember",
            action_taken="stored knowledge",
            result=text[:300],
            success=True,
            importance_score=importance,
            embedding_id=sem_id,
        )
        ids["episode_id"] = ep_id

        # Knowledge graph (concept node)
        if importance >= 0.7:
            concept_id = await self.graph.add_concept(
                name=text[:80],
                description=text,
                confidence=importance,
            )
            ids["concept_id"] = concept_id

        return ids

    # ── Retrieve context for a query ──────────────────────────────────────────

    async def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        include_skills: bool = True,
    ) -> dict[str, Any]:
        """
        Multi-memory retrieval for building agent context.
        Returns semantic matches, recent episodes, and relevant skills.
        """
        results: dict[str, Any] = {}

        # Semantic search
        semantic_hits = await self.semantic.search(query, limit=limit)
        results["semantic"] = [
            {"text": h.text, "score": round(h.score, 3), "meta": h.metadata}
            for h in semantic_hits
        ]

        # Recent episodes (keyword fallback)
        episodes = await self.episodic.search_by_task(query, limit=limit)
        results["episodes"] = episodes

        # Relevant skills
        if include_skills:
            skills = await self.procedural.search(query, limit=3)
            results["skills"] = skills

        return results

    # ── Working memory shortcuts ──────────────────────────────────────────────

    async def set_context(self, key: str, value: Any, ttl: int = 3600) -> None:
        await self.working.set(key, value, ttl=ttl)

    async def get_context(self, key: str) -> Any:
        return await self.working.get(key)

    async def push_turn(self, role: str, content: str) -> None:
        """Append a conversation turn to working memory."""
        await self.working.append_to_list(
            "conversation_history",
            {"role": role, "content": content},
            max_len=40,
        )

    async def get_history(self) -> list[dict]:
        return await self.working.get_list("conversation_history")

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        ep_stats = await self.episodic.stats()
        proc_stats = await self.procedural.stats()
        kg_stats = await self.graph.stats()
        return {
            "working_memory_keys": self.working.size,
            "episodic": ep_stats,
            "procedural": proc_stats,
            "knowledge_graph": kg_stats,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_brain: MemoryBrain | None = None


def get_memory_brain() -> MemoryBrain:
    global _brain
    if _brain is None:
        _brain = MemoryBrain()
    return _brain
