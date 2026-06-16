"""
ZEON Agent Council — Consensus Engine
Aggregates debate results into a final ZEON decision.
Uses majority-vote + confidence weighting + dissent logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from brains.council.debate import DebateResult, Position

log = structlog.get_logger(__name__)


@dataclass
class CouncilDecision:
    answer: str                      # final answer text
    confidence: float                # 0.0 – 1.0
    method: str                      # "consensus" | "majority" | "fallback"
    consensus_score: float
    rounds: int
    dissents: list[str]              # minority positions
    metadata: dict[str, Any]


class ConsensusEngine:
    """
    Takes a DebateResult and produces a final CouncilDecision.

    Decision rules (in order):
      1. If consensus_score >= 0.8 → unanimous, use winner directly
      2. If consensus_score >= 0.6 → majority, use winner with dissents noted
      3. If consensus_score < 0.6  → contested, synthesise a hedged answer
    """

    UNANIMOUS_THRESHOLD = 0.80
    MAJORITY_THRESHOLD  = 0.60

    def decide(self, result: DebateResult) -> CouncilDecision:
        winner = result.winning_position
        score  = result.consensus_score

        # Collect minority positions (not the winner)
        dissents = [
            f"{p.agent_id}: {p.content[:100]}"
            for p in result.all_positions
            if p.agent_id != winner.agent_id and p.content.strip()
        ]

        if score >= self.UNANIMOUS_THRESHOLD:
            method = "consensus"
            answer = winner.content
            confidence = winner.confidence

        elif score >= self.MAJORITY_THRESHOLD:
            method = "majority"
            answer = winner.content
            confidence = winner.confidence * 0.9   # slight uncertainty

        else:
            method = "fallback"
            # Synthesise a hedged answer from top two positions
            positions_text = "\n\n".join(
                f"[{p.agent_id}]: {p.content}"
                for p in result.all_positions[:2]
            )
            answer = (
                f"The council reached a split decision. "
                f"The leading position is:\n{winner.content}\n\n"
                f"However, there is significant disagreement. "
                f"Consider reviewing all perspectives."
            )
            confidence = max(0.3, winner.confidence * 0.6)

        decision = CouncilDecision(
            answer=answer,
            confidence=round(confidence, 3),
            method=method,
            consensus_score=score,
            rounds=result.rounds,
            dissents=dissents,
            metadata={
                "winner_agent": winner.agent_id,
                "total_positions": len(result.all_positions),
                "critiques": len(result.critiques),
                "unanimous": result.unanimous,
            }
        )

        log.info(
            "council.decision",
            method=method,
            confidence=decision.confidence,
            consensus=score,
            dissents=len(dissents),
        )
        return decision

    def format_for_speech(self, decision: CouncilDecision) -> str:
        """Format the decision for TTS output."""
        base = decision.answer.strip()
        if decision.method == "consensus":
            return base
        elif decision.method == "majority":
            suffix = f" Note: {len(decision.dissents)} agent(s) disagreed." if decision.dissents else ""
            return base + suffix
        else:
            return f"I'm not fully certain, but: {base}"
