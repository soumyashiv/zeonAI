"""
JARVIS Audit Log
Every action taken by any agent is recorded here.
SQLite-backed, async-safe, with reversibility snapshots.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()

DB_PATH = cfg.sqlite_path_resolved


class ActionType(str, Enum):
    FILE_READ     = "file_read"
    FILE_WRITE    = "file_write"
    FILE_DELETE   = "file_delete"
    SHELL_EXEC    = "shell_exec"
    BROWSER_CTRL  = "browser_ctrl"
    MEMORY_READ   = "memory_read"
    MEMORY_WRITE  = "memory_write"
    LLM_CALL      = "llm_call"
    AGENT_SPAWN   = "agent_spawn"
    SKILL_RUN     = "skill_run"
    CODE_RUN      = "code_run"
    SELF_IMPROVE  = "self_improve"
    SYSTEM        = "system"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id            TEXT PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    agent         TEXT NOT NULL,
    action_type   TEXT NOT NULL,
    action_detail TEXT,
    status        TEXT DEFAULT 'pending',
    approved_by   TEXT,
    result        TEXT,
    reversible    INTEGER DEFAULT 0,
    reversal_data TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id               TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    agent            TEXT NOT NULL,
    task_description TEXT,
    action_taken     TEXT,
    result           TEXT,
    success          INTEGER,
    importance_score REAL DEFAULT 0.5,
    embedding_id     TEXT
);

CREATE TABLE IF NOT EXISTS skills (
    id              TEXT PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    description     TEXT,
    code            TEXT,
    trigger_pattern TEXT,
    success_rate    REAL DEFAULT 0.0,
    usage_count     INTEGER DEFAULT 0,
    created_at      TEXT,
    last_used       TEXT
);
"""


class AuditLog:
    """Async SQLite audit log. Call `await AuditLog.init()` before first use."""

    _db: aiosqlite.Connection | None = None

    @classmethod
    async def init(cls) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        cls._db = await aiosqlite.connect(str(DB_PATH))
        cls._db.row_factory = aiosqlite.Row
        await cls._db.executescript(CREATE_TABLE_SQL)
        await cls._db.commit()
        log.info("audit_log.initialized", path=str(DB_PATH))

    @classmethod
    async def close(cls) -> None:
        if cls._db:
            await cls._db.close()

    @classmethod
    async def record(
        cls,
        *,
        agent: str,
        action_type: ActionType,
        action_detail: str,
        approved_by: str = "auto",
        reversible: bool = False,
        reversal_data: dict[str, Any] | None = None,
    ) -> str:
        """Create a pending audit entry. Returns the entry ID."""
        entry_id = str(uuid.uuid4())
        await cls._db.execute(
            """INSERT INTO audit_log
               (id, timestamp, agent, action_type, action_detail,
                status, approved_by, reversible, reversal_data)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                entry_id,
                datetime.now(timezone.utc).isoformat(),
                agent,
                action_type.value,
                action_detail,
                "approved" if approved_by else "pending",
                approved_by or "",
                int(reversible),
                json.dumps(reversal_data) if reversal_data else None,
            ),
        )
        await cls._db.commit()
        return entry_id

    @classmethod
    async def complete(
        cls,
        entry_id: str,
        *,
        result: str,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Mark an audit entry as completed."""
        await cls._db.execute(
            """UPDATE audit_log
               SET status=?, result=?, error=?
               WHERE id=?""",
            ("success" if success else "failed", result, error, entry_id),
        )
        await cls._db.commit()

    @classmethod
    async def save_episode(
        cls,
        *,
        agent: str,
        task_description: str,
        action_taken: str,
        result: str,
        success: bool,
        importance_score: float = 0.5,
        embedding_id: str | None = None,
    ) -> str:
        ep_id = str(uuid.uuid4())
        await cls._db.execute(
            """INSERT INTO episodes
               (id, timestamp, agent, task_description, action_taken,
                result, success, importance_score, embedding_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                ep_id,
                datetime.now(timezone.utc).isoformat(),
                agent,
                task_description,
                action_taken,
                result,
                int(success),
                importance_score,
                embedding_id,
            ),
        )
        await cls._db.commit()
        return ep_id

    @classmethod
    async def recent_episodes(cls, agent: str | None = None, limit: int = 20) -> list[dict]:
        if agent:
            cursor = await cls._db.execute(
                "SELECT * FROM episodes WHERE agent=? ORDER BY timestamp DESC LIMIT ?",
                (agent, limit),
            )
        else:
            cursor = await cls._db.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
