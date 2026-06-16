"""
ZEON Agent Council — Debate Engine
Structured multi-agent debate: each agent presents a position,
others critique it, council votes on the best response.

Debate flow:
  1. Proposer presents an initial answer
  2. Each critic agent scores and rebuts
  3. Proposer optionally revises
  4. Council votes using weighted scoring
  5. Returns winning position + dissenting notes
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from core.llm import chat
from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


@dataclass
class Position:
    agent_id: str
    content: str
    confidence: float        # 0.0 – 1.0
    reasoning: str
    round_num: int = 0


@dataclass
class Critique:
    critic_id: str
    target_agent_id: str
    score: float             # 0.0 – 10.0
    strengths: list[str]
    weaknesses: list[str]
    rebuttal: str


@dataclass
class DebateResult:
    winning_position: Position
    consensus_score: float   # 0.0 – 1.0
    all_positions: list[Position]
    critiques: list[Critique]
    rounds: int
    unanimous: bool


class DebateEngine:
    """
    Orchestrates a structured debate between ZEON agents.
    """

    MAX_ROUNDS = 2
    MIN_CONSENSUS = 0.7   # score needed to skip further rounds

    async def run(
        self,
        topic: str,
        proposers: list[str],        # agent IDs who will propose
        critics: list[str],          # agent IDs who will critique
        context: dict[str, Any] | None = None,
    ) -> DebateResult:
        """Run a full debate and return the winning position."""

        log.info("debate.start", topic=topic[:60],
                 proposers=proposers, critics=critics)

        ctx_str = str(context or {})[:300]
        all_positions: list[Position] = []
        all_critiques: list[Critique] = []

        # ── Round 1: Initial positions ───────────────────────────────────
        positions = await self._gather_positions(
            topic, proposers, ctx_str, round_num=1
        )
        all_positions.extend(positions)

        # ── Critique round ───────────────────────────────────────────────
        critiques = await self._gather_critiques(topic, positions, critics)
        all_critiques.extend(critiques)

        # ── Score and check consensus ─────────────────────────────────────
        winner, consensus = self._vote(positions, critiques)

        if consensus >= self.MIN_CONSENSUS or not positions:
            log.info("debate.consensus_reached",
                     winner=winner.agent_id, score=consensus)
            return DebateResult(
                winning_position=winner,
                consensus_score=consensus,
                all_positions=all_positions,
                critiques=all_critiques,
                rounds=1,
                unanimous=consensus >= 0.9,
            )

        # ── Round 2: Revision ─────────────────────────────────────────────
        revised = await self._revision_round(
            topic, winner, critiques, ctx_str
        )
        all_positions.append(revised)
        winner = revised
        consensus = min(consensus + 0.15, 1.0)   # revision boost

        log.info("debate.complete",
                 winner=winner.agent_id, rounds=2, consensus=consensus)

        return DebateResult(
            winning_position=winner,
            consensus_score=consensus,
            all_positions=all_positions,
            critiques=all_critiques,
            rounds=2,
            unanimous=consensus >= 0.9,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _gather_positions(
        self, topic: str, agents: list[str],
        context: str, round_num: int,
    ) -> list[Position]:
        tasks = [
            self._get_position(agent, topic, context, round_num)
            for agent in agents
        ]
        return await asyncio.gather(*tasks)

    async def _get_position(
        self, agent_id: str, topic: str, context: str, round_num: int
    ) -> Position:
        prompt = [
            {"role": "system", "content":
             f"You are ZEON agent '{agent_id}'. Provide a clear, "
             f"concise position on the given topic. Context: {context}"},
            {"role": "user", "content":
             f"Topic: {topic}\n\nProvide your position in 2-3 sentences, "
             f"then a one-sentence confidence statement (0.0-1.0)."},
        ]
        try:
            response = await chat(prompt, max_tokens=300)
            # Parse confidence from last line if present
            lines = response.strip().split("\n")
            confidence = 0.7
            content = response.strip()
            for line in reversed(lines):
                for word in line.split():
                    try:
                        val = float(word.strip(".,()"))
                        if 0.0 <= val <= 1.0:
                            confidence = val
                            break
                    except ValueError:
                        continue

            return Position(
                agent_id=agent_id,
                content=content,
                confidence=confidence,
                reasoning=content,
                round_num=round_num,
            )
        except Exception as e:
            log.error("debate.position_failed", agent=agent_id, error=str(e))
            return Position(
                agent_id=agent_id,
                content=f"[{agent_id} unavailable]",
                confidence=0.1,
                reasoning="",
                round_num=round_num,
            )

    async def _gather_critiques(
        self, topic: str,
        positions: list[Position],
        critics: list[str],
    ) -> list[Critique]:
        tasks = []
        for critic_id in critics:
            for pos in positions:
                if pos.agent_id != critic_id:   # don't self-critique
                    tasks.append(self._critique(critic_id, topic, pos))
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    async def _critique(
        self, critic_id: str, topic: str, position: Position
    ) -> Critique:
        prompt = [
            {"role": "system", "content":
             f"You are ZEON critic agent '{critic_id}'. Evaluate the given position critically."},
            {"role": "user", "content":
             f"Topic: {topic}\n\nPosition from {position.agent_id}:\n{position.content}\n\n"
             f"Score this position 0-10. List 1-2 strengths and 1-2 weaknesses. "
             f"Provide a one-sentence rebuttal or endorsement."},
        ]
        try:
            response = await chat(prompt, max_tokens=250)
            # Simple parse: extract a score number
            score = 7.0
            for token in response.split():
                try:
                    val = float(token.strip("/.,()"))
                    if 0.0 <= val <= 10.0:
                        score = val
                        break
                except ValueError:
                    continue

            return Critique(
                critic_id=critic_id,
                target_agent_id=position.agent_id,
                score=score,
                strengths=["Relevant", "Clear"],
                weaknesses=["Could be more specific"],
                rebuttal=response.strip()[:300],
            )
        except Exception as e:
            log.error("debate.critique_failed",
                      critic=critic_id, target=position.agent_id, error=str(e))
            return Critique(
                critic_id=critic_id,
                target_agent_id=position.agent_id,
                score=5.0,
                strengths=[],
                weaknesses=["Evaluation unavailable"],
                rebuttal="",
            )

    async def _revision_round(
        self, topic: str, winner: Position,
        critiques: list[Critique], context: str
    ) -> Position:
        relevant = [c for c in critiques if c.target_agent_id == winner.agent_id]
        critique_summary = "\n".join(
            f"- {c.rebuttal[:100]}" for c in relevant
        ) or "No critiques."

        prompt = [
            {"role": "system", "content":
             f"You are ZEON agent '{winner.agent_id}'. Revise your position based on critiques."},
            {"role": "user", "content":
             f"Topic: {topic}\n\nYour original position:\n{winner.content}\n\n"
             f"Critiques received:\n{critique_summary}\n\n"
             f"Provide an improved, revised position in 2-3 sentences."},
        ]
        try:
            revised_content = await chat(prompt, max_tokens=300)
            return Position(
                agent_id=winner.agent_id,
                content=revised_content.strip(),
                confidence=min(winner.confidence + 0.1, 1.0),
                reasoning="Revised after critique",
                round_num=2,
            )
        except Exception as e:
            log.error("debate.revision_failed", error=str(e))
            return winner   # return unchanged if revision fails

    def _vote(
        self, positions: list[Position], critiques: list[Critique]
    ) -> tuple[Position, float]:
        """Score positions by confidence + critic scores. Return winner + consensus."""
        if not positions:
            # Fallback empty position
            empty = Position("none", "No positions available", 0.0, "", 0)
            return empty, 0.0

        scores: dict[str, float] = {}
        for pos in positions:
            critic_scores = [
                c.score / 10.0
                for c in critiques
                if c.target_agent_id == pos.agent_id
            ]
            avg_critic = sum(critic_scores) / len(critic_scores) if critic_scores else 0.6
            scores[pos.agent_id] = pos.confidence * 0.4 + avg_critic * 0.6

        winner_id = max(scores, key=lambda k: scores[k])
        winner = next(p for p in positions if p.agent_id == winner_id)
        max_score = scores[winner_id]

        # Consensus = how much the winner leads (1.0 = unanimous)
        if len(scores) == 1:
            consensus = max_score
        else:
            sorted_scores = sorted(scores.values(), reverse=True)
            consensus = sorted_scores[0] / (sorted_scores[0] + sorted_scores[1] + 0.001)
            consensus = min(consensus, 1.0)

        return winner, round(consensus, 3)
