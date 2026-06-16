"""
Phase 3 — Orchestration & Agent Tests
LangGraph, router, executive/planner/critic agents, search tool.
All LLM calls are mocked to run offline.
"""
from __future__ import annotations

import json
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch


# ── LangGraph import check ────────────────────────────────────────────────────

def test_langgraph_importable():
    try:
        from langgraph.graph import StateGraph, END
        assert StateGraph is not None
    except ImportError:
        pytest.skip("langgraph not installed")


def test_jarvis_state_structure():
    from orchestration.state import JarvisState
    state: JarvisState = {
        "task_id": "abc",
        "task": "what is 2+2?",
        "agent_origin": "user",
        "revision_count": 0,
        "iteration": 0,
        "messages": [],
    }
    assert state["task"] == "what is 2+2?"
    assert state["revision_count"] == 0


# ── Router ────────────────────────────────────────────────────────────────────

def test_router_after_critique_low_score_revises():
    from orchestration.router import route_after_critique
    state = {"critique_score": 4, "revision_count": 0}
    assert route_after_critique(state) == "revise"


def test_router_after_critique_max_revisions_executes():
    from orchestration.router import route_after_critique
    state = {"critique_score": 4, "revision_count": 3}  # max reached
    # Should NOT revise again — should route to executor
    result = route_after_critique(state)
    assert result in ("general_executor", "research_executor", "coder_executor", "computer_executor")


def test_router_after_critique_passes_to_research():
    from orchestration.router import route_after_critique
    state = {
        "critique_score": 8,
        "revision_count": 0,
        "requires_research": True,
        "requires_coding": False,
        "requires_computer_control": False,
    }
    assert route_after_critique(state) == "research_executor"


def test_router_after_critique_passes_to_coder():
    from orchestration.router import route_after_critique
    state = {
        "critique_score": 8,
        "revision_count": 0,
        "requires_research": False,
        "requires_coding": True,
        "requires_computer_control": False,
    }
    assert route_after_critique(state) == "coder_executor"


def test_router_general_executor_by_default():
    from orchestration.router import route_after_critique
    state = {
        "critique_score": 9,
        "revision_count": 0,
        "requires_research": False,
        "requires_coding": False,
        "requires_computer_control": False,
    }
    assert route_after_critique(state) == "general_executor"


# ── Graph nodes (mocked LLM) ──────────────────────────────────────────────────

MOCK_UNDERSTAND_JSON = json.dumps({
    "goal": "what is 2+2",
    "constraints": [],
    "unknowns": [],
    "needs_research": False,
    "needs_coding": False,
    "needs_computer_control": False,
    "complexity": "low"
})

MOCK_PLAN_JSON = json.dumps({
    "goal": "what is 2+2",
    "steps": [
        {"id": 1, "action": "compute 2+2=4", "tool": "none",
         "expected_output": "4", "depends_on": []}
    ],
    "estimated_complexity": "low",
    "risks": []
})

MOCK_CRITIQUE_JSON = json.dumps({
    "scores": {"clarity": 9, "feasibility": 10, "safety": 10, "completeness": 9, "efficiency": 9},
    "overall_score": 9,
    "approved": True,
    "strengths": ["simple and clear"],
    "weaknesses": [],
    "suggestions": [],
    "verdict": "APPROVE"
})


@pytest.mark.asyncio
@patch("core.llm.chat", new_callable=AsyncMock, return_value=MOCK_UNDERSTAND_JSON)
async def test_node_understand(mock_chat, tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "u.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    # Reset memory singletons
    import brains.memory.working as wm_mod
    wm_mod._wm = None
    import brains.memory.episodic as ep_mod
    importlib.reload(ep_mod)
    import brains.memory as mem_mod
    importlib.reload(mem_mod)
    mem_mod._brain = None

    from orchestration.graph import node_understand
    state = {"task": "what is 2+2?", "context": {}, "messages": []}
    result = await node_understand(state)

    assert "understanding" in result
    assert result["understanding"]["goal"] == "what is 2+2"
    assert result["requires_research"] is False

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
@patch("core.llm.chat", new_callable=AsyncMock, return_value=MOCK_PLAN_JSON)
async def test_node_plan(mock_chat, tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "p.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    from orchestration.graph import node_plan
    state = {
        "task": "what is 2+2?",
        "understanding": {"goal": "compute 2+2", "needs_research": False},
        "knowledge": {},
        "revision_count": 0,
        "messages": [],
    }
    result = await node_plan(state)

    assert "plan" in result
    assert len(result["plan"]["steps"]) == 1
    assert result["plan"]["steps"][0]["tool"] == "none"
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
@patch("core.llm.chat", new_callable=AsyncMock, return_value=MOCK_CRITIQUE_JSON)
async def test_node_critique(mock_chat):
    from orchestration.graph import node_critique
    state = {
        "task": "what is 2+2?",
        "plan": {"goal": "compute 2+2", "steps": [{"id": 1, "action": "compute", "tool": "none"}]},
        "messages": [],
    }
    result = await node_critique(state)

    assert result["critique_score"] == 9
    assert "APPROVE" in result.get("critique_feedback", "") or result["critique_score"] >= 7


@pytest.mark.asyncio
async def test_node_execute_none_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "exec.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from orchestration.graph import node_execute
    state = {
        "task": "what is 2+2?",
        "plan": {
            "goal": "compute 2+2",
            "steps": [{"id": 1, "action": "Answer: 2+2=4", "tool": "none",
                       "expected_output": "4", "depends_on": []}],
        },
        "messages": [],
    }
    result = await node_execute(state)

    assert "result" in result
    assert result["result"]["steps_run"] == 1
    assert result["result"]["all_success"] is True

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_node_verify_success(tmp_path, monkeypatch):
    from orchestration.graph import node_verify
    state = {
        "task": "2+2",
        "plan": {},
        "result": {"steps_run": 1, "all_success": True, "results": []},
        "messages": [],
    }
    result = await node_verify(state)
    assert result["verified"] is True


@pytest.mark.asyncio
async def test_node_verify_failure():
    from orchestration.graph import node_verify
    state = {
        "task": "2+2",
        "plan": {},
        "result": {"steps_run": 2, "all_success": False,
                   "results": [{"success": False, "output": "error"}]},
        "messages": [],
    }
    result = await node_verify(state)
    assert result["verified"] is False


# ── Critic Agent standalone ───────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("core.llm.chat", new_callable=AsyncMock, return_value=MOCK_CRITIQUE_JSON)
async def test_critic_evaluate(mock_chat):
    from agents.critic_agent import CriticAgent
    critic = CriticAgent()
    result = await critic.evaluate(
        task="sort a list",
        plan_or_result={"steps": [{"id": 1, "action": "use sorted()", "tool": "code"}]}
    )
    assert result["overall_score"] == 9
    assert result["verdict"] == "APPROVE"
    assert result["approved"] is True


@pytest.mark.asyncio
@patch(
    "agents.critic_agent.chat",
    new_callable=AsyncMock,
    return_value='{"overall_score": 3, "verdict": "REVISE", "approved": false, "scores": {}, "strengths": [], "weaknesses": ["too vague"], "suggestions": ["be specific"]}'
)
async def test_critic_reject(mock_chat):
    from agents.critic_agent import CriticAgent
    critic = CriticAgent()
    result = await critic.evaluate(task="do things", plan_or_result={})
    assert result["overall_score"] == 3
    assert result["verdict"] == "REVISE"


# ── Registry + full graph build ───────────────────────────────────────────────

def test_graph_builds():
    try:
        from orchestration.graph import build_jarvis_graph
        graph = build_jarvis_graph()
        assert graph is not None
    except ImportError:
        pytest.skip("langgraph not installed")


# ── Search Tool ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quick_search_returns_snippets():
    """Mock DDGS inside quick_search's local scope."""
    fake_result = [
        {"title": "Python", "body": "A programming language", "href": "https://python.org"}
    ]
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text = MagicMock(return_value=fake_result)

    with patch("duckduckgo_search.DDGS", return_value=mock_ddgs_instance):
        from tools.search import quick_search
        results = await quick_search("python programming")
    assert len(results) >= 1
    assert "Python" in results[0]


@pytest.mark.asyncio
async def test_quick_search_import_error():
    with patch.dict("sys.modules", {"duckduckgo_search": None}):
        import importlib
        import tools.search as ts_mod
        importlib.reload(ts_mod)
        results = await ts_mod.quick_search("test query")
        # Should return graceful fallback message
        assert isinstance(results, list)
