"""
Phase 6 — Agent Council Tests
Debate engine, consensus, and council facade — all LLM calls mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Position / Critique dataclasses ─────────────────────────────────────────

def test_position_fields():
    from brains.council.debate import Position
    pos = Position(
        agent_id="executive",
        content="We should proceed with plan A.",
        confidence=0.85,
        reasoning="Plan A has the best ROI.",
        round_num=1,
    )
    assert pos.agent_id == "executive"
    assert pos.confidence == 0.85
    assert pos.round_num == 1


def test_critique_fields():
    from brains.council.debate import Critique
    c = Critique(
        critic_id="critic",
        target_agent_id="executive",
        score=7.5,
        strengths=["Clear", "Concise"],
        weaknesses=["Missing detail"],
        rebuttal="Good but needs more specifics.",
    )
    assert c.score == 7.5
    assert "Clear" in c.strengths


# ── DebateEngine voting ───────────────────────────────────────────────────────

def test_vote_single_position():
    from brains.council.debate import DebateEngine, Position
    engine = DebateEngine()
    positions = [Position("exec", "Answer A", 0.9, "reasoning", 1)]
    winner, consensus = engine._vote(positions, [])
    assert winner.agent_id == "exec"
    assert 0.0 <= consensus <= 1.0


def test_vote_selects_highest_confidence():
    from brains.council.debate import DebateEngine, Position, Critique
    engine = DebateEngine()
    pos_a = Position("exec",  "Answer A", 0.9, "", 1)
    pos_b = Position("planner","Answer B", 0.3, "", 1)
    # No critiques — pure confidence vote
    winner, consensus = engine._vote([pos_a, pos_b], [])
    assert winner.agent_id == "exec"


def test_vote_with_critique_scores():
    from brains.council.debate import DebateEngine, Position, Critique
    engine = DebateEngine()
    pos_a = Position("exec",    "Answer A", 0.5, "", 1)
    pos_b = Position("planner", "Answer B", 0.5, "", 1)
    # Give pos_b a higher critic score
    critiques = [
        Critique("critic", "planner", 9.0, [], [], ""),
        Critique("critic", "exec",    4.0, [], [], ""),
    ]
    winner, consensus = engine._vote([pos_a, pos_b], critiques)
    assert winner.agent_id == "planner"


def test_vote_empty_positions():
    from brains.council.debate import DebateEngine
    engine = DebateEngine()
    winner, consensus = engine._vote([], [])
    assert winner.agent_id == "none"
    assert consensus == 0.0


# ── DebateEngine async (mocked LLM) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_debate_run_mocked():
    """Full debate run with mocked LLM — verifies orchestration."""
    from brains.council.debate import DebateEngine

    mock_response = "This is the position. Confidence: 0.8"

    with patch("brains.council.debate.chat", new=AsyncMock(return_value=mock_response)):
        engine = DebateEngine()
        result = await engine.run(
            topic="Should we use plan A or plan B?",
            proposers=["executive", "planner"],
            critics=["critic"],
        )

    assert result.winning_position is not None
    assert len(result.all_positions) >= 2
    assert 0.0 <= result.consensus_score <= 1.0
    assert result.rounds >= 1


@pytest.mark.asyncio
async def test_debate_get_position_mocked():
    from brains.council.debate import DebateEngine

    with patch("brains.council.debate.chat",
               new=AsyncMock(return_value="The solution is X. Confidence: 0.75")):
        engine = DebateEngine()
        pos = await engine._get_position("exec", "Test topic", "{}", 1)

    assert pos.agent_id == "exec"
    assert "solution" in pos.content.lower() or pos.content
    assert 0.0 <= pos.confidence <= 1.0


@pytest.mark.asyncio
async def test_debate_critique_mocked():
    from brains.council.debate import DebateEngine, Position

    pos = Position("planner", "We should do X.", 0.7, "", 1)

    with patch("brains.council.debate.chat",
               new=AsyncMock(return_value="Score: 8. This is strong.")):
        engine = DebateEngine()
        critique = await engine._critique("critic", "topic", pos)

    assert critique.critic_id == "critic"
    assert critique.target_agent_id == "planner"
    assert 0.0 <= critique.score <= 10.0


@pytest.mark.asyncio
async def test_debate_llm_failure_graceful():
    """If LLM is down, debate should return a fallback position, not crash."""
    from brains.council.debate import DebateEngine

    with patch("brains.council.debate.chat",
               new=AsyncMock(side_effect=Exception("LLM offline"))):
        engine = DebateEngine()
        result = await engine.run(
            topic="Test topic",
            proposers=["exec"],
            critics=["critic"],
        )

    # Should not raise — returns graceful fallback
    assert result is not None
    assert result.winning_position.confidence <= 0.5


# ── ConsensusEngine ───────────────────────────────────────────────────────────

def test_consensus_unanimous():
    from brains.council.debate import DebateResult, Position
    from brains.council.consensus import ConsensusEngine

    winner = Position("exec", "Plan A is best.", 0.9, "", 1)
    result = DebateResult(
        winning_position=winner,
        consensus_score=0.85,
        all_positions=[winner],
        critiques=[],
        rounds=1,
        unanimous=True,
    )
    engine = ConsensusEngine()
    decision = engine.decide(result)
    assert decision.method == "consensus"
    assert decision.confidence >= 0.8
    assert decision.answer == winner.content


def test_consensus_majority():
    from brains.council.debate import DebateResult, Position
    from brains.council.consensus import ConsensusEngine

    winner = Position("exec", "Plan A.", 0.75, "", 1)
    minority = Position("planner", "Plan B.", 0.6, "", 1)
    result = DebateResult(
        winning_position=winner,
        consensus_score=0.65,
        all_positions=[winner, minority],
        critiques=[],
        rounds=1,
        unanimous=False,
    )
    engine = ConsensusEngine()
    decision = engine.decide(result)
    assert decision.method == "majority"
    assert len(decision.dissents) == 1
    assert "planner" in decision.dissents[0]


def test_consensus_fallback_low_score():
    from brains.council.debate import DebateResult, Position
    from brains.council.consensus import ConsensusEngine

    winner = Position("exec", "Plan A.", 0.5, "", 1)
    result = DebateResult(
        winning_position=winner,
        consensus_score=0.45,
        all_positions=[winner],
        critiques=[],
        rounds=2,
        unanimous=False,
    )
    engine = ConsensusEngine()
    decision = engine.decide(result)
    assert decision.method == "fallback"
    assert decision.confidence <= 0.6


def test_consensus_format_for_speech_consensus():
    from brains.council.consensus import ConsensusEngine, CouncilDecision
    engine = ConsensusEngine()
    decision = CouncilDecision(
        answer="The answer is X.",
        confidence=0.9,
        method="consensus",
        consensus_score=0.85,
        rounds=1,
        dissents=[],
        metadata={},
    )
    text = engine.format_for_speech(decision)
    assert "The answer is X." in text


def test_consensus_format_for_speech_fallback():
    from brains.council.consensus import ConsensusEngine, CouncilDecision
    engine = ConsensusEngine()
    decision = CouncilDecision(
        answer="Maybe X.",
        confidence=0.3,
        method="fallback",
        consensus_score=0.4,
        rounds=2,
        dissents=[],
        metadata={},
    )
    text = engine.format_for_speech(decision)
    assert "certain" in text.lower() or "not" in text.lower()


# ── AgentCouncil facade ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_council_deliberate_mocked():
    from brains.council import AgentCouncil

    mock_resp = "The council recommends approach X. Confidence: 0.8"
    with patch("brains.council.debate.chat", new=AsyncMock(return_value=mock_resp)):
        council = AgentCouncil(
            proposers=["exec", "planner"],
            critics=["critic"],
        )
        decision = await council.deliberate("What is the best strategy?")

    assert decision.answer
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.method in ("consensus", "majority", "fallback")


@pytest.mark.asyncio
async def test_council_quick_vote_mocked():
    from brains.council import AgentCouncil

    options = ["Option A", "Option B", "Option C"]

    council = AgentCouncil(
        proposers=["exec", "planner"],
        critics=["critic"],
    )
    # quick_vote calls core.llm.chat internally
    with patch("core.llm.chat", new=AsyncMock(return_value="Option A")):
        result = await council.quick_vote("Which option?", options)

    assert result in options


def test_council_singleton():
    from brains.council import get_council, AgentCouncil
    import brains.council as cm
    cm._council = None   # reset
    c1 = get_council()
    c2 = get_council()
    assert c1 is c2
    cm._council = None   # cleanup
