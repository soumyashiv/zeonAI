"""
ZEON Agent Council — Unified Facade
Run a full multi-agent debate-and-decide cycle with one call.
"""
from __future__ import annotations

from typing import Any

import structlog

from brains.council.debate import DebateEngine, DebateResult
from brains.council.consensus import ConsensusEngine, CouncilDecision

log = structlog.get_logger(__name__)

# Default council members
DEFAULT_PROPOSERS = ["executive", "planner", "research"]
DEFAULT_CRITICS   = ["critic", "executive"]


class AgentCouncil:
    """
    High-level council API. Call council.deliberate(topic) to get a decision.
    """

    def __init__(
        self,
        proposers: list[str] | None = None,
        critics: list[str] | None = None,
    ) -> None:
        self._proposers = proposers or DEFAULT_PROPOSERS
        self._critics   = critics   or DEFAULT_CRITICS
        self._debate_engine = DebateEngine()
        self._consensus_engine = ConsensusEngine()

    async def deliberate(
        self,
        topic: str,
        context: dict[str, Any] | None = None,
        proposers: list[str] | None = None,
        critics: list[str] | None = None,
    ) -> CouncilDecision:
        """
        Run a full council debate and return a CouncilDecision.
        """
        used_proposers = proposers or self._proposers
        used_critics   = critics   or self._critics

        log.info("council.deliberate", topic=topic[:80])

        debate_result = await self._debate_engine.run(
            topic=topic,
            proposers=used_proposers,
            critics=used_critics,
            context=context,
        )

        decision = self._consensus_engine.decide(debate_result)
        log.info(
            "council.decided",
            method=decision.method,
            confidence=decision.confidence,
            answer_preview=decision.answer[:80],
        )
        return decision

    async def quick_vote(
        self,
        question: str,
        options: list[str],
    ) -> str:
        """
        Simple multi-agent vote on a set of options.
        Returns the option with most 'votes'.
        """
        from core.llm import chat

        votes: dict[str, int] = {opt: 0 for opt in options}
        all_agents = self._proposers + self._critics

        async def get_vote(agent_id: str) -> str:
            opts_str = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
            prompt = [
                {"role": "system", "content": f"You are ZEON agent '{agent_id}'. Vote for the best option."},
                {"role": "user", "content":
                 f"Question: {question}\n\nOptions:\n{opts_str}\n\n"
                 f"Reply with only the exact text of your chosen option."},
            ]
            try:
                response = await chat(prompt, max_tokens=50)
                response = response.strip()
                # Match to nearest option
                for opt in options:
                    if opt.lower() in response.lower() or response.lower() in opt.lower():
                        return opt
                return options[0]   # fallback
            except Exception:
                return options[0]

        import asyncio
        results = await asyncio.gather(*[get_vote(a) for a in all_agents])
        for vote in results:
            if vote in votes:
                votes[vote] += 1

        winner = max(votes, key=lambda k: votes[k])
        log.info("council.quick_vote", question=question[:60],
                 result=winner, votes=votes)
        return winner


_council: AgentCouncil | None = None


def get_council() -> AgentCouncil:
    global _council
    if _council is None:
        _council = AgentCouncil()
    return _council
