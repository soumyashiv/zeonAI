"""
ZEON Base Agent
All agents inherit from this class. Provides event bus integration,
audit logging, security gateway access, and the 10-step reasoning loop.
"""
from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import structlog

from core.config import get_config
from core.event_bus import EventBus, AgentMessage, MessageType, get_bus
from core.audit_log import AuditLog, ActionType
from core.security_gateway import get_gateway

log = structlog.get_logger(__name__)
cfg = get_config()


class BaseAgent(ABC):
    """
    Abstract base for all ZEON agents.

    Subclasses must implement `handle_task()`.
    The 10-step reasoning loop is wired in `process_message()`.
    """

    name: str = "base_agent"
    description: str = ""

    def __init__(self) -> None:
        self._bus: EventBus = get_bus()
        self._gateway = get_gateway()
        self._log = structlog.get_logger(self.__class__.__name__)
        self._running = False
        self._id = str(uuid.uuid4())

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Register with the event bus and begin listening."""
        self._bus.subscribe(self.name, self._on_message)
        self._bus.subscribe("broadcast", self._on_broadcast)
        self._running = True
        self._log.info("agent.started", name=self.name)
        await self._on_start()

    async def stop(self) -> None:
        self._running = False
        self._bus.unsubscribe(self.name, self._on_message)
        self._bus.unsubscribe("broadcast", self._on_broadcast)
        await self._on_stop()
        self._log.info("agent.stopped", name=self.name)

    async def _on_start(self) -> None:
        """Override for custom startup logic."""

    async def _on_stop(self) -> None:
        """Override for custom shutdown logic."""

    # ── Message handling ──────────────────────────────────────────

    async def _on_message(self, message: AgentMessage) -> None:
        if not self._running:
            return
        self._log.debug("agent.received", type=message.message_type.value)
        await self.process_message(message)

    async def _on_broadcast(self, message: AgentMessage) -> None:
        """Handle broadcast messages — override to react."""

    async def process_message(self, message: AgentMessage) -> None:
        """
        Default message handler. Dispatches to handle_task for TASK_CREATED.
        Override for custom routing.
        """
        if message.message_type == MessageType.TASK_CREATED:
            await self._bus.publish(
                message.reply(
                    MessageType.TASK_ACCEPTED,
                    {"status": "accepted"},
                )
            )
            await self._execute_reasoning_loop(message)
        elif message.message_type == MessageType.HEARTBEAT:
            await self._bus.publish(
                message.reply(
                    MessageType.SYSTEM_EVENT,
                    {"agent": self.name, "status": "alive"},
                )
            )

    # ── 10-Step Reasoning Loop ────────────────────────────────────

    async def _execute_reasoning_loop(self, task_message: AgentMessage) -> None:
        task_id = task_message.id
        task = task_message.payload.get("task", "")
        self._log.info("reasoning.start", task=task[:80])

        try:
            # 1. OBSERVE
            context = await self.observe(task_message.payload)

            # 2. UNDERSTAND
            understanding = await self.understand(task, context)

            # 3. RESEARCH (if needed)
            knowledge = await self.research(understanding)

            # 4. PLAN
            plan = await self.plan(task, understanding, knowledge)

            # 5. CRITIQUE
            for attempt in range(cfg.max_plan_revision_loops):
                score, feedback = await self.critique(plan)
                self._log.info(
                    "reasoning.critique",
                    score=score,
                    attempt=attempt + 1,
                )

                # 6. REVISE if needed
                if score >= cfg.critic_min_score:
                    break
                plan = await self.revise(plan, feedback)
            else:
                self._log.warning("reasoning.max_revisions_reached")

            # 7. EXECUTE
            result = await self.execute(plan, task_message)

            # 8. VERIFY
            verified, verify_notes = await self.verify(result, plan)

            # 9. LEARN
            await self.learn(task, plan, result, success=verified)

            # 10. MEMORY UPDATE
            await self.update_memory(task, result, verified)

            # Publish result
            await self._bus.publish(
                task_message.reply(
                    MessageType.TASK_RESULT,
                    {
                        "result": result,
                        "verified": verified,
                        "verify_notes": verify_notes,
                    },
                )
            )
            self._log.info("reasoning.complete", verified=verified)

        except Exception as exc:
            self._log.error("reasoning.error", error=str(exc), exc_info=True)
            await self._bus.publish(
                task_message.reply(
                    MessageType.TASK_RESULT,
                    {"error": str(exc), "verified": False},
                )
            )

    # ── Reasoning Steps (override in subclasses) ──────────────────

    async def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Gather context from memory and environment."""
        return {"payload": payload}

    async def understand(self, task: str, context: dict) -> dict[str, Any]:
        """Parse goal, identify constraints."""
        return {"task": task, "context": context}

    async def research(self, understanding: dict) -> dict[str, Any]:
        """Fill knowledge gaps. Override to call Research Brain."""
        return {}

    async def plan(
        self, task: str, understanding: dict, knowledge: dict
    ) -> dict[str, Any]:
        """Create step-by-step task graph. Override in Planner."""
        return {"steps": [task], "task": task}

    async def critique(self, plan: dict) -> tuple[int, str]:
        """Evaluate plan quality. Returns (score 1-10, feedback)."""
        return cfg.critic_min_score, "default pass"

    async def revise(self, plan: dict, feedback: str) -> dict[str, Any]:
        """Improve the plan based on critique feedback."""
        return plan

    @abstractmethod
    async def execute(
        self, plan: dict[str, Any], original_message: AgentMessage
    ) -> Any:
        """Carry out the plan. Must be implemented by every agent."""
        ...

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        """Check result matches expectations."""
        return True, "unverified"

    async def learn(
        self, task: str, plan: dict, result: Any, success: bool
    ) -> None:
        """Store episode in audit log."""
        await AuditLog.save_episode(
            agent=self.name,
            task_description=task,
            action_taken=str(plan),
            result=str(result)[:500],
            success=success,
        )

    async def update_memory(
        self, task: str, result: Any, success: bool
    ) -> None:
        """Publish memory update event for Memory Agent."""
        await self._bus.publish(
            AgentMessage(
                from_agent=self.name,
                to_agent="memory_agent",
                message_type=MessageType.MEMORY_UPDATE,
                payload={
                    "task": task,
                    "result": str(result)[:500],
                    "success": success,
                    "agent": self.name,
                },
                priority=3,
            )
        )

    # ── Helpers ───────────────────────────────────────────────────

    async def request_approval(
        self, action_type: str, detail: str, payload: dict | None = None
    ):
        return await self._gateway.request_approval(
            action_type=action_type,
            agent=self.name,
            detail=detail,
            payload=payload,
        )

    async def send(
        self,
        to_agent: str,
        message_type: MessageType,
        payload: dict[str, Any],
        *,
        priority: int = 5,
    ) -> None:
        """Send a message to another agent."""
        await self._bus.publish(
            AgentMessage(
                from_agent=self.name,
                to_agent=to_agent,
                message_type=message_type,
                payload=payload,
                priority=priority,
            )
        )
