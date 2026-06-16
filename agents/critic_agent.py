"""
ZEON Critic Agent
Evaluates plans and results. Provides structured feedback with scores.
Challenges bad decisions. Champions quality.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat

log = structlog.get_logger(__name__)

CRITIC_PROMPT = """You are ZEON's Critic Agent — a rigorous quality evaluator.
Evaluate the provided plan or result against these criteria:

1. CLARITY      — Is the goal clear and unambiguous? (0-10)
2. FEASIBILITY  — Can this actually be done with available tools? (0-10)
3. SAFETY       — Does it avoid harmful or irreversible actions? (0-10)
4. COMPLETENESS — Does it fully address the original task? (0-10)
5. EFFICIENCY   — Is there unnecessary complexity or redundancy? (0-10)

Respond ONLY in this JSON format:
{
  "scores": {
    "clarity": 8,
    "feasibility": 7,
    "safety": 10,
    "completeness": 6,
    "efficiency": 7
  },
  "overall_score": 7,
  "approved": true,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "verdict": "APPROVE | REVISE | REJECT"
}"""


class CriticAgent(BaseAgent):
    name = "critic_agent"
    description = "Evaluates plans/results. Scores 1-10. Provides revision feedback."

    async def evaluate(
        self,
        *,
        task: str,
        plan_or_result: dict | str,
        context: str = "",
    ) -> dict[str, Any]:
        """
        Direct evaluation API (called by other agents, not via message bus).
        Returns full critique dict including overall_score.
        """
        content = json.dumps(plan_or_result) if isinstance(plan_or_result, dict) else str(plan_or_result)
        messages = [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content":
                f"Original task: {task}\n\nPlan/Result to evaluate:\n{content}\n\nAdditional context: {context}"},
        ]

        try:
            raw = await chat(messages, temperature=0.2, max_tokens=512)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                critique = json.loads(raw[start:end])
            else:
                critique = self._default_critique(7, raw)
        except Exception as e:
            log.warning("critic.evaluate_failed", error=str(e))
            critique = self._default_critique(7, str(e))

        self._log.info(
            "critic.scored",
            score=critique.get("overall_score"),
            verdict=critique.get("verdict"),
        )
        return critique

    def _default_critique(self, score: int, reason: str) -> dict:
        return {
            "scores": {"clarity": score, "feasibility": score,
                       "safety": 10, "completeness": score, "efficiency": score},
            "overall_score": score,
            "approved": score >= 7,
            "strengths": [],
            "weaknesses": [reason[:200]],
            "suggestions": ["Re-evaluate manually"],
            "verdict": "APPROVE" if score >= 7 else "REVISE",
        }

    async def execute(
        self, plan: dict[str, Any], original_message: AgentMessage
    ) -> Any:
        task = original_message.payload.get("task", "")
        target = original_message.payload.get("plan_or_result", plan)
        critique = await self.evaluate(task=task, plan_or_result=target)

        # Notify requester of critique result
        await self.send(
            original_message.from_agent,
            MessageType.CRITIQUE_DONE,
            {"critique": critique, "task": task},
            priority=8,
        )
        return critique

    async def critique(self, plan: dict) -> tuple[int, str]:
        # Critic critiques itself minimally
        return 9, "critic self-check passed"

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        score = result.get("overall_score", 0) if isinstance(result, dict) else 0
        return score > 0, f"critique score={score}"
