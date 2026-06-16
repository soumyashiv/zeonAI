"""
JARVIS Orchestration State
The shared state object that flows through all LangGraph nodes.
"""
from __future__ import annotations

from typing import Any, TypedDict, Annotated
import operator


class JarvisState(TypedDict, total=False):
    # ── Task ───────────────────────────────────────────────────────
    task_id: str
    task: str                           # original user request
    agent_origin: str                   # which agent submitted the task

    # ── Reasoning loop ─────────────────────────────────────────────
    context: dict[str, Any]             # gathered context (observe step)
    understanding: dict[str, Any]       # parsed goal + constraints
    knowledge: dict[str, Any]           # research results
    plan: dict[str, Any]                # step-by-step plan
    critique_score: int                 # 1-10
    critique_feedback: str
    revision_count: int                 # how many revisions so far
    result: Any                         # final execution result
    verified: bool
    verify_notes: str
    error: str | None

    # ── Multi-agent routing ────────────────────────────────────────
    next_agent: str                     # which agent should handle next
    requires_research: bool
    requires_coding: bool
    requires_computer_control: bool

    # ── Memory ─────────────────────────────────────────────────────
    memory_context: dict[str, Any]      # recalled memories for this task
    conversation_history: list[dict]

    # ── Self-improvement ───────────────────────────────────────────
    improvement_proposal: dict | None
    improvement_approved: bool

    # ── Metadata ───────────────────────────────────────────────────
    messages: Annotated[list[dict], operator.add]   # accumulated LLM messages
    iteration: int
