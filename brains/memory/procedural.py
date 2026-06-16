"""
Procedural Memory — Skills registry backed by SQLite.
Stores reusable skills: Python callables, prompts, or workflow templates
that JARVIS has compiled from successful episode patterns.
"""
from __future__ import annotations

import importlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import aiosqlite
import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


class ProceduralMemory:
    """
    Manages JARVIS skills — reusable, compiled procedures learned from experience.
    Skills are stored in SQLite and optionally as Python files in skills_registry/.
    """

    @property
    def _db_path(self) -> str:
        return str(cfg.sqlite_path_resolved)

    async def register(
        self,
        *,
        name: str,
        description: str,
        trigger_pattern: str,
        code: str | None = None,
        overwrite: bool = False,
    ) -> str:
        skill_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            existing = await db.execute(
                "SELECT id FROM skills WHERE name=?", (name,)
            )
            row = await existing.fetchone()

            if row and not overwrite:
                log.warning("procedural_memory.skill_exists", name=name)
                return row[0]

            if row and overwrite:
                await db.execute(
                    """UPDATE skills SET description=?, trigger_pattern=?, code=?,
                       last_used=? WHERE name=?""",
                    (description, trigger_pattern, code, now, name),
                )
            else:
                await db.execute(
                    """INSERT INTO skills
                       (id, name, description, code, trigger_pattern,
                        success_rate, usage_count, created_at, last_used)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (skill_id, name, description, code,
                     trigger_pattern, 0.0, 0, now, now),
                )
            await db.commit()

        log.info("procedural_memory.skill_registered", name=name)
        return skill_id

    async def get(self, name: str) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM skills WHERE name=?", (name,)
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Find skills whose trigger_pattern or description matches query."""
        like = f"%{query}%"
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM skills
                   WHERE trigger_pattern LIKE ? OR description LIKE ?
                   ORDER BY success_rate DESC, usage_count DESC
                   LIMIT ?""",
                (like, like, limit),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def list_all(self, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM skills ORDER BY usage_count DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def record_usage(self, name: str, success: bool) -> None:
        """Update success_rate and usage_count after a skill is run."""
        async with aiosqlite.connect(self._db_path) as db:
            # Fetch existing
            cursor = await db.execute(
                "SELECT usage_count, success_rate FROM skills WHERE name=?", (name,)
            )
            row = await cursor.fetchone()
            if not row:
                return
            count, rate = row
            new_count = count + 1
            new_rate = ((rate * count) + (1.0 if success else 0.0)) / new_count
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "UPDATE skills SET usage_count=?, success_rate=?, last_used=? WHERE name=?",
                (new_count, new_rate, now, name),
            )
            await db.commit()

    async def delete(self, name: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM skills WHERE name=?", (name,))
            await db.commit()

    async def stats(self) -> dict[str, Any]:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*), AVG(success_rate), SUM(usage_count) FROM skills"
            )
            row = await cursor.fetchone()
        return {
            "total_skills": row[0] or 0,
            "avg_success_rate": round(row[1] or 0.0, 3),
            "total_executions": row[2] or 0,
        }


_procedural: ProceduralMemory | None = None


def get_procedural_memory() -> ProceduralMemory:
    global _procedural
    if _procedural is None:
        _procedural = ProceduralMemory()
    return _procedural
