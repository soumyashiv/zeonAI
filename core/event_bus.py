"""
ZEON Event Bus
Async message broker for all inter-agent communication.
Falls back to in-process asyncio queue when Redis is unavailable.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(str, Enum):
    TASK_CREATED       = "task_created"
    TASK_ACCEPTED      = "task_accepted"
    TASK_PROGRESS      = "task_progress"
    TASK_RESULT        = "task_result"
    CRITIQUE_REQUEST   = "critique_request"
    CRITIQUE_DONE      = "critique_done"
    VETO_RAISED        = "veto_raised"
    APPROVAL_NEEDED    = "approval_needed"
    APPROVAL_GRANTED   = "approval_granted"
    APPROVAL_DENIED    = "approval_denied"
    MEMORY_UPDATE      = "memory_update"
    SKILL_LEARNED      = "skill_learned"
    SELF_IMPROVE_PR    = "self_improve_pr"
    SYSTEM_EVENT       = "system_event"
    HEARTBEAT          = "heartbeat"


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str              # agent name or "broadcast"
    message_type: MessageType
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 5          # 1 (low) → 10 (critical)
    requires_approval: bool = False
    parent_task_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, raw: str) -> "AgentMessage":
        d = json.loads(raw)
        d["message_type"] = MessageType(d["message_type"])
        return cls(**d)

    def reply(
        self,
        message_type: MessageType,
        payload: dict[str, Any],
        *,
        priority: int | None = None,
    ) -> "AgentMessage":
        """Convenience: create a reply to this message."""
        return AgentMessage(
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=message_type,
            payload=payload,
            priority=priority or self.priority,
            parent_task_id=self.id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Subscriber type
# ─────────────────────────────────────────────────────────────────────────────

Handler = Callable[[AgentMessage], Awaitable[None]]


# ─────────────────────────────────────────────────────────────────────────────
# Event Bus
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """
    In-process async event bus with optional Redis pub/sub persistence.
    Supports wildcard subscription via agent name "broadcast".
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, AgentMessage]] = (
            asyncio.PriorityQueue()
        )
        self._running = False
        self._task: asyncio.Task | None = None
        self._redis = None   # lazy-loaded

    # ── Subscription ──────────────────────────────────────────────

    def subscribe(self, agent_name: str, handler: Handler) -> None:
        """Register a handler for messages addressed to agent_name."""
        self._subscribers.setdefault(agent_name, []).append(handler)
        log.debug("event_bus.subscribed", agent=agent_name)

    def unsubscribe(self, agent_name: str, handler: Handler) -> None:
        handlers = self._subscribers.get(agent_name, [])
        if handler in handlers:
            handlers.remove(handler)

    # ── Publishing ────────────────────────────────────────────────

    async def publish(self, message: AgentMessage) -> None:
        """Enqueue a message. Priority queue uses (10 - priority) so higher = first."""
        await self._queue.put((10 - message.priority, message))
        log.debug(
            "event_bus.published",
            msg_id=message.id,
            from_agent=message.from_agent,
            to_agent=message.to_agent,
            type=message.message_type.value,
        )

    def publish_sync(self, message: AgentMessage) -> None:
        """Fire-and-forget from sync context."""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.publish(message))
            )
        except RuntimeError:
            asyncio.run(self.publish(message))

    # ── Dispatch loop ─────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        log.info("event_bus.started")
        while self._running:
            try:
                _, message = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._deliver(message)
            self._queue.task_done()

    async def _deliver(self, message: AgentMessage) -> None:
        targets = set()
        # Direct subscribers
        for h in self._subscribers.get(message.to_agent, []):
            targets.add(h)
        # Broadcast subscribers
        for h in self._subscribers.get("broadcast", []):
            targets.add(h)

        if not targets:
            log.warning("event_bus.no_handler", to_agent=message.to_agent)
            return

        for handler in targets:
            try:
                await handler(message)
            except Exception as exc:
                log.error(
                    "event_bus.handler_error",
                    handler=handler.__qualname__,
                    error=str(exc),
                    exc_info=True,
                )

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._dispatch_loop())
        log.info("event_bus.running")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("event_bus.stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

_bus: EventBus | None = None


def get_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
