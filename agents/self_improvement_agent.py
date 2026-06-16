"""
ZEON Self-Improvement Agent
Monitors performance, proposes improvements, and applies them with approval.
Phase 7 — fully operational after Phase 3-6 are mature.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat
from brains.memory import get_memory_brain

log = structlog.get_logger(__name__)

ANALYSER_PROMPT = """You are ZEON's Self-Improvement Analyst.
Analyse the performance data and propose ONE targeted improvement.

Improvement types:
- prompt_optimisation: Improve a system prompt
- skill_addition: Define a new skill/procedure
- parameter_tuning: Adjust temperature, max_tokens, etc.
- workflow_change: Restructure an agent workflow

Respond as JSON:
{
  "type": "...",
  "target": "which_agent_or_module",
  "rationale": "...",
  "proposal": "...",
  "expected_improvement": "...",
  "risk_level": "low|medium|high",
  "requires_human_approval": true/false
}"""


class SelfImprovementAgent(BaseAgent):
    name = "self_improvement_agent"
    description = "Analyses perf metrics, proposes and applies safe self-improvements."

    async def analyse(self) -> dict | None:
        """Gather performance stats and propose an improvement."""
        mem = get_memory_brain()
        ep_stats = await mem.episodic.stats()
        proc_stats = await mem.procedural.stats()

        perf_summary = (
            f"Episodes: {ep_stats['total_episodes']}, "
            f"Success rate: {ep_stats['success_rate']:.1%}, "
            f"Skills: {proc_stats['total_skills']}, "
            f"Skill success: {proc_stats['avg_success_rate']:.1%}"
        )

        messages = [
            {"role": "system", "content": ANALYSER_PROMPT},
            {"role": "user", "content": f"Performance data:\n{perf_summary}"},
        ]

        try:
            raw = await chat(messages, temperature=0.4, max_tokens=400)
            s, e = raw.find("{"), raw.rfind("}") + 1
            proposal = json.loads(raw[s:e]) if s >= 0 and e > s else None
        except Exception as ex:
            self._log.warning("self_improve.analyse_failed", error=str(ex))
            return None

        if proposal:
            self._log.info(
                "self_improve.proposal",
                type=proposal.get("type"),
                target=proposal.get("target"),
                risk=proposal.get("risk_level"),
            )
        return proposal

    async def apply_skill_addition(self, proposal: dict) -> bool:
        """Register a new skill based on the proposal."""
        mem = get_memory_brain()
        name = f"auto_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        await mem.procedural.register(
            name=name,
            description=proposal.get("proposal", ""),
            trigger_pattern=proposal.get("target", ""),
            code=None,
        )
        self._log.info("self_improve.skill_added", name=name)
        return True

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        action = original_message.payload.get("action", "analyse")

        if action == "analyse":
            proposal = await self.analyse()
            if not proposal:
                return {"status": "no_proposal"}

            # High-risk changes always require human approval
            if proposal.get("requires_human_approval") or proposal.get("risk_level") == "high":
                self._log.info("self_improve.awaiting_approval", type=proposal.get("type"))
                await self.send(
                    "executive_agent",
                    MessageType.SYSTEM_EVENT,
                    {"event": "improvement_proposal", "proposal": proposal},
                    priority=5,
                )
                return {"status": "pending_approval", "proposal": proposal}

            # Auto-apply low/medium risk skill additions in dev mode
            from core.config import get_config
            cfg = get_config()
            if cfg.dev_auto_approve and proposal.get("type") == "skill_addition":
                applied = await self.apply_skill_addition(proposal)
                return {"status": "applied", "proposal": proposal, "success": applied}

            return {"status": "queued", "proposal": proposal}

        return {"status": "unknown_action"}

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        if isinstance(result, dict):
            return result.get("status") != "error", result.get("status", "?")
        return False, "no result"
