"""
ZEON Memory Agent
Listens for MEMORY_UPDATE events and persists knowledge across all memory types.
Also handles memory queries from other agents.
"""
from __future__ import annotations

from typing import Any
import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from brains.memory import get_memory_brain

log = structlog.get_logger(__name__)


class MemoryAgent(BaseAgent):
    name = "memory_agent"
    description = "Persists and retrieves knowledge across all memory types."

    async def _on_message(self, message: AgentMessage) -> None:
        if message.message_type == MessageType.MEMORY_UPDATE:
            await self._handle_memory_update(message)
        else:
            await super()._on_message(message)

    async def _handle_memory_update(self, message: AgentMessage) -> None:
        payload = message.payload
        mem = get_memory_brain()

        task = payload.get("task", "")
        result = payload.get("result", "")
        success = payload.get("success", True)
        agent = payload.get("agent", message.from_agent)
        importance = 0.6 if success else 0.3

        try:
            await mem.remember(
                f"{task} → {result}",
                agent=agent,
                importance=importance,
            )
            self._log.debug("memory_agent.stored", agent=agent, task=task[:50])
        except Exception as e:
            self._log.error("memory_agent.store_failed", error=str(e))

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        task = original_message.payload.get("task", "")
        query = original_message.payload.get("query", task)
        mem = get_memory_brain()

        if original_message.payload.get("action") == "recall":
            results = await mem.recall(query)
            await self.send(
                original_message.from_agent,
                MessageType.TASK_RESULT,
                {"memory_results": results},
                priority=6,
            )
            return results

        if original_message.payload.get("action") == "stats":
            stats = await mem.stats()
            await self.send(
                original_message.from_agent,
                MessageType.TASK_RESULT,
                {"memory_stats": stats},
                priority=5,
            )
            return stats

        # Default: store
        ids = await mem.remember(task, importance=0.5)
        return {"stored": True, "ids": ids}
