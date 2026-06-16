"""
Phase 8 — Interfaces Tests
CLI dispatch logic and Web API endpoints — no real LLM calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── CLI — ZeonCLI dispatch ────────────────────────────────────────────────────

def test_cli_imports():
    from interfaces.cli import ZeonCLI, BANNER, HELP_TEXT
    assert "ZEON" in BANNER
    assert "ask" in HELP_TEXT.lower()


@pytest.mark.asyncio
async def test_cli_dispatch_exit():
    from interfaces.cli import ZeonCLI
    cli = ZeonCLI()
    cli._running = True
    await cli._dispatch("exit")
    assert cli._running is False


@pytest.mark.asyncio
async def test_cli_dispatch_quick_mocked():
    from interfaces.cli import ZeonCLI
    cli = ZeonCLI()
    with patch("core.llm.chat", new=AsyncMock(return_value="42")):
        await cli._cmd_quick("what is 6*7")
    assert len(cli._history) == 1
    assert cli._history[0][1] == "42"


@pytest.mark.asyncio
async def test_cli_quick_llm_success():
    from interfaces.cli import ZeonCLI
    cli = ZeonCLI()
    with patch("core.llm.chat", new=AsyncMock(return_value="test answer")):
        result = await cli._quick_llm("hello?")
    assert result == "test answer"


@pytest.mark.asyncio
async def test_cli_quick_llm_failure_graceful():
    from interfaces.cli import ZeonCLI
    cli = ZeonCLI()
    with patch("core.llm.chat", new=AsyncMock(side_effect=Exception("offline"))):
        result = await cli._quick_llm("hello?")
    assert "unavailable" in result.lower()


def test_cli_history_empty():
    from interfaces.cli import ZeonCLI
    from io import StringIO
    import contextlib
    cli = ZeonCLI()
    # Should not crash
    cli._cmd_history()


def test_cli_history_records():
    from interfaces.cli import ZeonCLI
    cli = ZeonCLI()
    cli._history = [("q1", "a1"), ("q2", "a2")]
    assert len(cli._history) == 2


# ── Web — FastAPI endpoints ───────────────────────────────────────────────────

@pytest.fixture
def web_client():
    from fastapi.testclient import TestClient
    from interfaces.web import app
    return TestClient(app)


def test_web_dashboard_returns_html(web_client):
    r = web_client.get("/")
    assert r.status_code == 200
    assert "ZEON" in r.text
    assert "<!DOCTYPE html>" in r.text


def test_web_status_ok(web_client):
    r = web_client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "model" in data
    assert "version" in data


def test_web_ask_success(web_client):
    with patch("core.llm.chat", new=AsyncMock(return_value="The answer is 42.")):
        r = web_client.post("/api/ask", json={"question": "What is the meaning?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["answer"] == "The answer is 42."


def test_web_ask_empty_question(web_client):
    r = web_client.post("/api/ask", json={"question": ""})
    assert r.status_code == 400


def test_web_ask_llm_failure(web_client):
    with patch("core.llm.chat", new=AsyncMock(side_effect=Exception("LLM down"))):
        r = web_client.post("/api/ask", json={"question": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "unavailable" in data["answer"].lower() or "LLM" in data["answer"]


def test_web_memory_stats(web_client):
    r = web_client.get("/api/memory/stats")
    assert r.status_code == 200
    data = r.json()
    assert "tiers" in data
    assert len(data["tiers"]) == 5
    tier_names = [t["name"] for t in data["tiers"]]
    assert "Working" in tier_names
    assert "Episodic" in tier_names


def test_web_memory_search_empty_query(web_client):
    r = web_client.get("/api/memory/search")
    assert r.status_code == 200
    assert r.json()["results"] == []


def test_web_memory_search_with_query(web_client):
    mock_ep = MagicMock()
    mock_ep.content = "Found research results"
    with patch("brains.memory.get_memory_brain") as mock_brain_factory:
        mock_brain = MagicMock()
        mock_brain.episodic.search = AsyncMock(return_value=[mock_ep])
        mock_brain_factory.return_value = mock_brain
        r = web_client.get("/api/memory/search?q=research")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data


def test_web_council_endpoint(web_client):
    from brains.council.consensus import CouncilDecision
    mock_decision = CouncilDecision(
        answer="The council recommends X.",
        confidence=0.85,
        method="consensus",
        consensus_score=0.85,
        rounds=1,
        dissents=[],
        metadata={},
    )
    with patch("brains.council.get_council") as mock_factory:
        mock_council = MagicMock()
        mock_council.deliberate = AsyncMock(return_value=mock_decision)
        mock_factory.return_value = mock_council
        r = web_client.post("/api/council", json={"topic": "What should we build?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "The council recommends X."
    assert data["method"] == "consensus"
    assert data["confidence"] == 0.85


def test_web_council_failure_graceful(web_client):
    with patch("brains.council.get_council") as mock_factory:
        mock_council = MagicMock()
        mock_council.deliberate = AsyncMock(side_effect=Exception("debate failed"))
        mock_factory.return_value = mock_council
        r = web_client.post("/api/council", json={"topic": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "error" in data["answer"].lower() or "Council" in data["answer"]


def test_web_skills_empty(web_client):
    with patch("brains.memory.get_memory_brain") as mock_factory:
        mock_brain = MagicMock()
        mock_brain.procedural.list_skills = AsyncMock(return_value=[])
        mock_factory.return_value = mock_brain
        r = web_client.get("/api/skills")
    assert r.status_code == 200
    assert r.json()["skills"] == []


def test_web_history_empty(web_client):
    from interfaces import web as web_mod
    web_mod._interactions.clear()
    r = web_client.get("/api/history")
    assert r.status_code == 200
    assert r.json()["interactions"] == []


def test_web_improve_endpoint(web_client):
    from improvements.skill_compiler import ImprovementReport
    mock_report = ImprovementReport(
        skills_extracted=2,
        skills_registered=2,
        episodes_analysed=10,
        patterns_found=3,
        failed_patterns=0,
        summary="Compiled 2 skills.",
    )
    with patch("improvements.skill_compiler.SkillCompiler.compile",
               new=AsyncMock(return_value=mock_report)):
        with patch("brains.memory.get_memory_brain", return_value=MagicMock()):
            r = web_client.post("/api/improve")
    assert r.status_code == 200
    data = r.json()
    assert data["skills_extracted"] == 2
    assert "Compiled" in data["summary"]
