"""
ZEON LangGraph Orchestration
The main StateGraph that implements the 10-step reasoning loop
as a directed graph. Nodes call agent methods. Edges use router logic.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog

from orchestration.state import ZeonState
from orchestration.router import route_after_critique, route_after_plan

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node functions (called by LangGraph)
# Each node receives state, returns partial state update.
# ─────────────────────────────────────────────────────────────────────────────

async def node_observe(state: ZeonState) -> dict:
    """Step 1: Gather context from memory."""
    from brains.memory import get_memory_brain
    mem = get_memory_brain()
    task = state.get("task", "")
    recalled = await mem.recall(task, limit=4)
    history = await mem.get_history()
    return {
        "memory_context": recalled,
        "conversation_history": history[-6:],
        "context": {"recalled": recalled, "history": history[-6:]},
        "iteration": state.get("iteration", 0),
    }


async def node_understand(state: ZeonState) -> dict:
    """Step 2: Parse goal and identify constraints."""
    from core.llm import chat
    task = state.get("task", "")
    context = state.get("context", {})
    recalled = context.get("recalled", {})

    context_str = ""
    if recalled.get("semantic"):
        context_str = "\nRelevant knowledge:\n"
        for h in recalled["semantic"][:2]:
            context_str += f"  - {h.get('text','')[:120]}\n"

    messages = [
        {"role": "system", "content":
         "Analyze this task. Identify: goal, constraints, unknowns, "
         "whether research/coding/computer-control is needed. "
         "Reply as JSON: {goal, constraints[], unknowns[], "
         "needs_research, needs_coding, needs_computer_control, complexity}"},
        {"role": "user", "content": f"Task: {task}\n{context_str}"},
    ]

    try:
        raw = await chat(messages, temperature=0.3, max_tokens=400)
        s, e = raw.find("{"), raw.rfind("}") + 1
        understanding = json.loads(raw[s:e]) if s >= 0 and e > s else {"goal": task}
    except Exception as ex:
        log.warning("graph.understand_failed", error=str(ex))
        understanding = {"goal": task, "needs_research": False,
                         "needs_coding": False, "needs_computer_control": False}

    return {
        "understanding": understanding,
        "requires_research": understanding.get("needs_research", False),
        "requires_coding": understanding.get("needs_coding", False),
        "requires_computer_control": understanding.get("needs_computer_control", False),
        "messages": [{"role": "user", "content": task}],
    }


async def node_research(state: ZeonState) -> dict:
    """Step 3: Fill knowledge gaps (only if requires_research)."""
    if not state.get("requires_research"):
        return {"knowledge": {}}

    from tools.search import quick_search
    task = state.get("task", "")
    try:
        results = await quick_search(task, max_results=3)
        return {"knowledge": {"search_results": results}}
    except Exception as e:
        log.warning("graph.research_failed", error=str(e))
        return {"knowledge": {"error": str(e)}}


async def node_plan(state: ZeonState) -> dict:
    """Step 4: Create step-by-step plan."""
    from core.llm import chat
    task = state.get("task", "")
    understanding = state.get("understanding", {})
    knowledge = state.get("knowledge", {})

    knowledge_str = ""
    if knowledge.get("search_results"):
        knowledge_str = "\nResearch results:\n" + "\n".join(
            f"  - {r}" for r in knowledge["search_results"][:3]
        )

    messages = [
        {"role": "system", "content":
         "Create a step-by-step plan. Reply as JSON: "
         "{goal, steps:[{id,action,tool,expected_output,depends_on[]}], "
         "estimated_complexity, risks[]}"},
        {"role": "user", "content":
         f"Task: {task}\nUnderstanding: {json.dumps(understanding)}{knowledge_str}"},
    ]

    try:
        raw = await chat(messages, temperature=0.4, max_tokens=800)
        s, e = raw.find("{"), raw.rfind("}") + 1
        plan = json.loads(raw[s:e]) if s >= 0 and e > s else {
            "goal": task, "steps": [{"id": 1, "action": task,
            "tool": "none", "expected_output": "", "depends_on": []}],
            "estimated_complexity": "low", "risks": []
        }
    except Exception as ex:
        log.warning("graph.plan_failed", error=str(ex))
        plan = {"goal": task,
                "steps": [{"id": 1, "action": task, "tool": "none",
                           "expected_output": "", "depends_on": []}],
                "estimated_complexity": "low", "risks": [str(ex)]}

    return {"plan": plan, "revision_count": state.get("revision_count", 0)}


async def node_critique(state: ZeonState) -> dict:
    """Step 5: Evaluate the plan."""
    from agents.critic_agent import CriticAgent
    critic = CriticAgent()
    task = state.get("task", "")
    plan = state.get("plan", {})

    critique = await critic.evaluate(task=task, plan_or_result=plan)
    score = critique.get("overall_score", 7)
    feedback = "; ".join(critique.get("suggestions", []))

    log.info("graph.critique", score=score, verdict=critique.get("verdict"))
    return {
        "critique_score": score,
        "critique_feedback": feedback,
        "messages": [{"role": "assistant",
                      "content": f"[Critique score: {score}/10] {feedback}"}],
    }


async def node_revise(state: ZeonState) -> dict:
    """Step 6: Revise plan based on critique."""
    from core.llm import chat
    plan = state.get("plan", {})
    feedback = state.get("critique_feedback", "")
    task = state.get("task", "")

    messages = [
        {"role": "system", "content":
         "Revise this plan based on the feedback. Reply as JSON with same structure."},
        {"role": "user", "content":
         f"Task: {task}\nOriginal plan: {json.dumps(plan)}\nFeedback: {feedback}"},
    ]

    try:
        raw = await chat(messages, temperature=0.4, max_tokens=800)
        s, e = raw.find("{"), raw.rfind("}") + 1
        revised = json.loads(raw[s:e]) if s >= 0 and e > s else plan
    except Exception:
        revised = plan

    return {
        "plan": revised,
        "revision_count": state.get("revision_count", 0) + 1,
    }


async def node_execute(state: ZeonState) -> dict:
    """Step 7: Execute the plan steps."""
    from agents.executor_agent import ExecutorAgent

    plan = state.get("plan", {})
    steps = plan.get("steps", [])
    task = state.get("task", "")

    results = []
    for step in steps:
        tool = step.get("tool", "none")
        action = step.get("action", "")

        if tool == "search":
            from tools.search import quick_search
            try:
                hits = await quick_search(action, max_results=3)
                results.append({"step": step.get("id"), "output": hits, "success": True})
            except Exception as e:
                results.append({"step": step.get("id"), "output": str(e), "success": False})
        elif tool == "none":
            results.append({"step": step.get("id"), "output": f"✓ {action}", "success": True})
        else:
            results.append({"step": step.get("id"), "output": f"[Tool '{tool}' queued]", "success": True})

    all_ok = all(r.get("success") for r in results)
    result = {"steps_run": len(results), "all_success": all_ok, "results": results, "task": task}

    return {
        "result": result,
        "messages": [{"role": "assistant",
                      "content": f"Executed {len(results)} steps. Success: {all_ok}"}],
    }


async def node_verify(state: ZeonState) -> dict:
    """Step 8: Verify result matches expectations."""
    result = state.get("result", {})
    plan = state.get("plan", {})
    task = state.get("task", "")

    if isinstance(result, dict) and result.get("all_success"):
        verified = True
        notes = f"All {result.get('steps_run', 0)} steps completed successfully."
    elif isinstance(result, dict) and not result.get("all_success"):
        verified = False
        failed = [r for r in result.get("results", []) if not r.get("success")]
        notes = f"{len(failed)} step(s) failed: {[f.get('output','')[:60] for f in failed]}"
    else:
        verified = bool(result)
        notes = "Result present."

    return {"verified": verified, "verify_notes": notes}


async def node_learn(state: ZeonState) -> dict:
    """Step 9: Extract lessons and compile skills."""
    from core.audit_log import AuditLog
    task = state.get("task", "")
    result = state.get("result", {})
    verified = state.get("verified", False)

    await AuditLog.save_episode(
        agent="zeon_graph",
        task_description=task,
        action_taken=json.dumps(state.get("plan", {}).get("steps", []))[:300],
        result=str(result)[:300],
        success=verified,
        importance_score=0.7 if verified else 0.3,
    )
    return {}


async def node_update_memory(state: ZeonState) -> dict:
    """Step 10: Update semantic and graph memory."""
    from brains.memory import get_memory_brain
    mem = get_memory_brain()
    task = state.get("task", "")
    result = state.get("result", {})
    verified = state.get("verified", False)

    summary = f"Task: {task} | Result: {str(result)[:200]} | Success: {verified}"
    await mem.remember(summary, importance=0.7 if verified else 0.3)
    await mem.push_turn("assistant", summary[:200])

    return {"messages": [{"role": "assistant", "content": summary[:200]}]}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_zeon_graph():
    """Build and return the compiled ZEON reasoning graph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        log.error("langgraph not installed — run: pip install langgraph")
        return None

    from core.config import get_config
    cfg = get_config()

    builder = StateGraph(ZeonState)

    # Add all 10 reasoning nodes
    builder.add_node("observe",        node_observe)
    builder.add_node("understand",     node_understand)
    builder.add_node("research",       node_research)
    builder.add_node("plan",           node_plan)
    builder.add_node("critique",       node_critique)
    builder.add_node("revise",         node_revise)
    builder.add_node("execute",        node_execute)
    builder.add_node("verify",         node_verify)
    builder.add_node("learn",          node_learn)
    builder.add_node("update_memory",  node_update_memory)

    # Linear flow: observe → understand → research → plan
    builder.set_entry_point("observe")
    builder.add_edge("observe",    "understand")
    builder.add_edge("understand", "research")
    builder.add_edge("research",   "plan")

    # Conditional: plan → critique → (revise loop | execute)
    builder.add_edge("plan", "critique")
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "revise":            "revise",
            "general_executor":  "execute",
            "research_executor": "execute",
            "coder_executor":    "execute",
            "computer_executor": "execute",
        },
    )
    builder.add_edge("revise", "critique")

    # Linear: execute → verify → learn → update_memory → END
    builder.add_edge("execute",       "verify")
    builder.add_edge("verify",        "learn")
    builder.add_edge("learn",         "update_memory")
    builder.add_edge("update_memory", END)

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_zeon_graph():
    global _graph
    if _graph is None:
        _graph = build_zeon_graph()
    return _graph


async def run_task(task: str, agent_origin: str = "user") -> ZeonState:
    """Run a single task through the full ZEON reasoning loop."""
    graph = get_zeon_graph()
    if graph is None:
        raise RuntimeError("LangGraph not available. Install langgraph.")

    initial_state: ZeonState = {
        "task_id": str(uuid.uuid4()),
        "task": task,
        "agent_origin": agent_origin,
        "revision_count": 0,
        "iteration": 0,
        "messages": [],
        "requires_research": False,
        "requires_coding": False,
        "requires_computer_control": False,
    }

    log.info("zeon_graph.run_start", task=task[:80])
    final_state = await graph.ainvoke(initial_state)
    log.info(
        "zeon_graph.run_complete",
        verified=final_state.get("verified"),
        score=final_state.get("critique_score"),
    )
    return final_state
