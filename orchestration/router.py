"""
JARVIS Orchestration Router
Conditional edge logic for the LangGraph StateGraph.
Determines which node runs next based on current state.
"""
from __future__ import annotations
from orchestration.state import JarvisState
from core.config import get_config

cfg = get_config()


def route_after_plan(state: JarvisState) -> str:
    """After planning: go to critique (always)."""
    return "critique"


def route_after_critique(state: JarvisState) -> str:
    """
    After critique:
    - Score too low AND revisions remaining → revise
    - Score OK → route to correct executor
    """
    score = state.get("critique_score", 0)
    revisions = state.get("revision_count", 0)
    max_revisions = cfg.max_plan_revision_loops

    if score < cfg.critic_min_score and revisions < max_revisions:
        return "revise"

    # Route to specialised executor based on task flags
    if state.get("requires_coding"):
        return "coder_executor"
    if state.get("requires_computer_control"):
        return "computer_executor"
    if state.get("requires_research"):
        return "research_executor"
    return "general_executor"


def route_after_verify(state: JarvisState) -> str:
    """After verification: always learn then end."""
    return "learn"


def route_after_learn(state: JarvisState) -> str:
    """After learning: update memory, then END."""
    return "update_memory"
