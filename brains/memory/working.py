"""
Working Memory — Redis-backed (falls back to in-process dict).
Short-lived context: current task, recent turns, active goals.
TTL-based expiry. Thread-safe.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


class WorkingMemory:
    """
    Fast, ephemeral key-value store for the current session.
    Keys expire after `default_ttl` seconds.
    Uses in-process dict (no Redis needed for Phase 2).
    """

    def __init__(self, default_ttl: int = 3600) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        log.info("working_memory.initialized", backend="in-process")

    async def set(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        expires_at = time.monotonic() + (ttl or self._default_ttl)
        async with self._lock:
            self._store[key] = (value, expires_at)

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def set_json(self, key: str, value: dict, *, ttl: int | None = None) -> None:
        await self.set(key, json.dumps(value), ttl=ttl)

    async def get_json(self, key: str) -> dict | None:
        raw = await self.get(key)
        return json.loads(raw) if raw else None

    async def append_to_list(self, key: str, item: Any, *, max_len: int = 50) -> None:
        """Append to a list stored in working memory (e.g. conversation turns)."""
        existing = await self.get(key) or []
        existing.append(item)
        if len(existing) > max_len:
            existing = existing[-max_len:]
        await self.set(key, existing)

    async def get_list(self, key: str) -> list:
        return await self.get(key) or []

    async def purge_expired(self) -> int:
        """Remove all expired keys. Returns count removed."""
        now = time.monotonic()
        async with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
        return len(expired)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# Singleton per session
_wm: WorkingMemory | None = None


def get_working_memory() -> WorkingMemory:
    global _wm
    if _wm is None:
        _wm = WorkingMemory()
    return _wm
