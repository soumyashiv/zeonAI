"""
JARVIS Planner Agent
Converts a high-level task into a concrete, ordered step graph.
Uses LLM to decompose goals. Outputs machine-readable task steps.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat
from brains.memory import get_memory_brain

log = structlog.get_logger(__name__)

PLANNER_PROMPT = """You are JARVIS's Planner Agent.
Your job: decompose a task into clear, ordered, executable steps.

Rules:
- Each step must be atomic and verifiable
- Steps must be ordered by dependency
- Flag steps that need external tools (web, code, file, browser)
- Be realistic — if you lack information, add a research step first

Respond ONLY in this JSON format:
{
  "goal": "...",
  "steps": [
    {
      "id": 1,
      "action": "...",
      "tool": "none|search|code|file|browser|shell",
      "expected_output": "...",
      "depends_on": []
    }
  ],
  "estimated_complexity": "low|medium|high",
  "risks": ["..."]
}"""


class PlannerAgent(BaseAgent):
    name = "planner"
    description = "Decomposes goals into ordered, executable step graphs."

    async def observe(self, payload: dict[str, Any]) -> dict[str, Any]:
        mem = get_memory_brain()
        task = payload.get("task", "")
        recalled = await mem.recall(task, limit=3)
        skills = await mem.procedural.search(task, limit=3)
        return {"recalled": recalled, "skills": skills, "flags": payload.get("flags", {})}

    async def understand(self, task: str, context: dict) -> dict[str, Any]:
        return {"task": task, "context": context}

    async def plan(self, task: str, understanding: dict, knowledge: dict) -> dict[str, Any]:
        context = understanding.get("context", {})
        recalled = context.get("recalled", {})
        skills = context.get("skills", [])

        context_block = ""
        if recalled.get("episodes"):
            context_block += "\nPast experience:\n"
            for ep in recalled["episodes"][:2]:
                context_block += f"  - {ep.get('task_description','')}: {ep.get('result','')[:100]}\n"
        if skills:
            context_block += "\nAvailable skills:\n"
            for s in skills[:3]:
                context_block += f"  - {s.get('name','')}: {s.get('description','')}\n"

        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": f"Task: {task}\n{context_block}"},
        ]

        try:
            raw = await chat(messages, temperature=0.4, max_tokens=1024)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                plan_data = json.loads(raw[start:end])
            else:
                plan_data = self._fallback_plan(task, raw)
        except Exception as e:
            log.warning("planner.plan_failed", error=str(e))
            plan_data = self._fallback_plan(task, str(e))

        self._log.info(
            "planner.plan_created",
            steps=len(plan_data.get("steps", [])),
            complexity=plan_data.get("estimated_complexity", "?"),
        )
        return plan_data

    def _fallback_plan(self, task: str, reason: str) -> dict:
        return {
            "goal": task,
            "steps": [{"id": 1, "action": task, "tool": "none",
                       "expected_output": "task result", "depends_on": []}],
            "estimated_complexity": "low",
            "risks": [f"Fallback plan due to: {reason[:100]}"],
        }

    async def critique(self, plan: dict) -> tuple[int, str]:
        steps = plan.get("steps", [])
        if not steps:
            return 2, "No steps generated."
        if len(steps) == 1 and steps[0].get("tool") == "none":
            return 6, "Plan is minimal — may need more decomposition."
        return 8, f"Plan has {len(steps)} steps, complexity={plan.get('estimated_complexity')}."

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        """
        Publish plan to executor agent. Planner does not execute — it plans.
        """
        await self.send(
            "executor_agent",
            MessageType.TASK_CREATED,
            {"task": plan.get("goal"), "plan": plan, "parent_task_id": original_message.id},
            priority=7,
        )
        return {"plan_published": True, "steps": len(plan.get("steps", []))}

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        return result.get("plan_published", False), "plan dispatched to executor"
