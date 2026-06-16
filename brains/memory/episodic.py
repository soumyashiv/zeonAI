"""
Episodic Memory — SQLite-backed experience log with semantic search hooks.
Stores: what happened, when, by which agent, with what result.
Supports importance scoring and embedding ID cross-reference to Qdrant.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite
import structlog

from core.config import get_config
from core.audit_log import AuditLog  # reuses the episodes table

log = structlog.get_logger(__name__)
cfg = get_config()


class EpisodicMemory:
    """
    Long-term experience store. Every meaningful agent action creates
    an episode. Episodes can be retrieved by recency, agent, or
    semantic similarity (via embedding_id → Qdrant lookup).
    """

    async def add(
        self,
        *,
        agent: str,
        task_description: str,
        action_taken: str,
        result: str,
        success: bool,
        importance_score: float = 0.5,
        embedding_id: str | None = None,
    ) -> str:
        return await AuditLog.save_episode(
            agent=agent,
            task_description=task_description,
            action_taken=action_taken,
            result=result,
            success=success,
            importance_score=importance_score,
            embedding_id=embedding_id,
        )

    async def recent(
        self,
        agent: str | None = None,
        limit: int = 20,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        episodes = await AuditLog.recent_episodes(agent=agent, limit=limit * 2)
        if success_only:
            episodes = [e for e in episodes if e["success"]]
        return episodes[:limit]

    async def important(self, threshold: float = 0.7, limit: int = 10) -> list[dict]:
        """Retrieve highest-importance episodes (for context injection)."""
        db_path = cfg.sqlite_path_resolved
        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM episodes
                   WHERE importance_score >= ?
                   ORDER BY importance_score DESC, timestamp DESC
                   LIMIT ?""",
                (threshold, limit),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def search_by_task(self, query: str, limit: int = 5) -> list[dict]:
        """Simple keyword search over task descriptions (pre-vector fallback)."""
        db_path = cfg.sqlite_path_resolved
        like = f"%{query}%"
        async with aiosqlite.connect(str(db_path)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM episodes
                   WHERE task_description LIKE ? OR result LIKE ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (like, like, limit),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_embedding_id(self, episode_id: str, embedding_id: str) -> None:
        """Link an episode to its Qdrant vector after embedding is created."""
        db_path = cfg.sqlite_path_resolved
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute(
                "UPDATE episodes SET embedding_id=? WHERE id=?",
                (embedding_id, episode_id),
            )
            await db.commit()

    async def stats(self) -> dict[str, Any]:
        db_path = cfg.sqlite_path_resolved
        async with aiosqlite.connect(str(db_path)) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) as total, AVG(success) as success_rate, "
                "AVG(importance_score) as avg_importance FROM episodes"
            )
            row = await cursor.fetchone()
        return {
            "total_episodes": row[0],
            "success_rate": round(row[1] or 0, 3),
            "avg_importance": round(row[2] or 0, 3),
        }


_episodic: EpisodicMemory | None = None


def get_episodic_memory() -> EpisodicMemory:
    global _episodic
    if _episodic is None:
        _episodic = EpisodicMemory()
    return _episodic
