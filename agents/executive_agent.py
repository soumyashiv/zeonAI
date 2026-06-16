"""
ZEON Executive Agent
Top-level reasoning agent. Has final authority over all other agents.
Responsibilities:
  - Parse user goal into structured task
  - Delegate to specialist agents
  - Resolve conflicts between agents
  - Approve/reject self-improvement proposals
"""
from __future__ import annotations

from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat
from brains.memory import get_memory_brain

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are ZEON, an autonomous AI operating system.
You are the Executive Agent — the highest-authority decision maker.

Your responsibilities:
1. Parse the user's goal into clear, structured tasks
2. Determine which specialist agents are needed
3. Maintain coherence across all agent actions
4. Reject plans that violate safety or ethics
5. Approve or deny self-improvement proposals

Always respond in this JSON format:
{
  "task_summary": "...",
  "requires_research": true/false,
  "requires_coding": true/false,
  "requires_computer_control": true/false,
  "next_agent": "planner|research_agent|coder_agent|executor_agent",
  "reasoning": "...",
  "flags": {}
}"""


class ExecutiveAgent(BaseAgent):
    name = "executive_agent"
    description = "Top-level authority. Parses goals, delegates, resolves conflicts."

    async def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        mem = get_memory_brain()
        history = await mem.get_history()
        recalled = await mem.recall(payload.get("task", ""), limit=3)
        await mem.set_context("current_task", payload.get("task", ""))
        return {"history": history[-6:], "recalled": recalled}

    async def understand(self, task: str, context: dict) -> dict[str, Any]:
        """Use LLM to parse the goal into structured understanding."""
        import json

        history = context.get("history", [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": task})

        try:
            raw = await chat(messages, temperature=0.3, max_tokens=512)
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
            else:
                parsed = {"task_summary": task, "next_agent": "planner",
                          "requires_research": False, "requires_coding": False,
                          "requires_computer_control": False, "reasoning": raw}
        except Exception as e:
            log.warning("executive.understand_failed", error=str(e))
            parsed = {"task_summary": task, "next_agent": "planner",
                      "requires_research": False, "requires_coding": False,
                      "requires_computer_control": False, "reasoning": str(e)}

        self._log.info("executive.understood",
                       next_agent=parsed.get("next_agent"),
                       summary=parsed.get("task_summary", "")[:60])
        return parsed

    async def plan(self, task: str, understanding: dict, knowledge: dict) -> dict[str, Any]:
        return {
            "delegate_to": understanding.get("next_agent", "planner"),
            "task_summary": understanding.get("task_summary", task),
            "flags": {
                "requires_research": understanding.get("requires_research", False),
                "requires_coding": understanding.get("requires_coding", False),
                "requires_computer_control": understanding.get("requires_computer_control", False),
            },
        }

    async def critique(self, plan: dict) -> tuple[int, str]:
        delegate = plan.get("delegate_to", "")
        if not delegate:
            return 3, "No delegation target specified."
        valid_agents = {"planner", "research_agent", "coder_agent",
                        "executor_agent", "memory_agent"}
        if delegate not in valid_agents:
            return 5, f"Unknown agent: {delegate}. Defaulting to planner."
        return 9, "Valid delegation plan."

    async def execute(
        self, plan: dict[str, Any], original_message: AgentMessage
    ) -> dict[str, Any]:
        mem = get_memory_brain()
        await mem.push_turn("user", original_message.payload.get("task", ""))

        delegate = plan.get("delegate_to", "planner")
        self._log.info("executive.delegating", to=delegate)

        # Dispatch to next agent
        await self.send(
            delegate,
            MessageType.TASK_CREATED,
            {
                "task": plan.get("task_summary"),
                "flags": plan.get("flags", {}),
                "parent_task_id": original_message.id,
            },
            priority=8,
        )
        return {"delegated_to": delegate, "plan": plan}

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        return bool(result), "executive delegation completed"
