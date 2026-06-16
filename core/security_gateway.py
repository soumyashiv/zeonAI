"""
JARVIS Security Gateway
HITL approval system. In dev mode with JARVIS_DEV_AUTO_APPROVE=true,
all actions are auto-approved. In production, destructive actions
require explicit human confirmation.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


class RiskLevel(str, Enum):
    LOW        = "low"       # file reads, memory lookups — never blocked
    MEDIUM     = "medium"    # file writes, network requests
    HIGH       = "high"      # shell execution, browser control
    CRITICAL   = "critical"  # self-modification, system changes


# Map action keywords to risk levels
_RISK_MAP: dict[str, RiskLevel] = {
    "file_read":     RiskLevel.LOW,
    "memory_read":   RiskLevel.LOW,
    "llm_call":      RiskLevel.LOW,
    "memory_write":  RiskLevel.MEDIUM,
    "file_write":    RiskLevel.MEDIUM,
    "browser_ctrl":  RiskLevel.HIGH,
    "shell_exec":    RiskLevel.HIGH,
    "file_delete":   RiskLevel.HIGH,
    "agent_spawn":   RiskLevel.MEDIUM,
    "skill_run":     RiskLevel.MEDIUM,
    "self_improve":  RiskLevel.CRITICAL,
}

# Actions never allowed without explicit approval even in dev
_ALWAYS_REQUIRE_APPROVAL = {RiskLevel.CRITICAL}


@dataclass
class ApprovalRequest:
    action_type: str
    agent: str
    detail: str
    risk_level: RiskLevel
    payload: dict[str, Any]


@dataclass
class ApprovalResult:
    approved: bool
    approved_by: str
    reason: str | None = None


class SecurityGateway:
    """
    Validates agent actions against risk policy.
    In dev mode: auto-approves LOW/MEDIUM/HIGH, prompts for CRITICAL.
    In prod mode: prompts for MEDIUM+.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}

    def classify_risk(self, action_type: str) -> RiskLevel:
        return _RISK_MAP.get(action_type, RiskLevel.MEDIUM)

    async def request_approval(
        self,
        *,
        action_type: str,
        agent: str,
        detail: str,
        payload: dict[str, Any] | None = None,
    ) -> ApprovalResult:
        """
        Check if action is permitted. Returns ApprovalResult.
        Auto-approves in dev mode unless action is CRITICAL.
        """
        risk = self.classify_risk(action_type)
        req = ApprovalRequest(
            action_type=action_type,
            agent=agent,
            detail=detail,
            risk_level=risk,
            payload=payload or {},
        )

        # Always-block rules
        if risk in _ALWAYS_REQUIRE_APPROVAL:
            return await self._prompt_human(req)

        # Dev auto-approve
        if cfg.is_dev and cfg.dev_auto_approve:
            log.debug(
                "security.auto_approved",
                agent=agent,
                action=action_type,
                risk=risk.value,
            )
            return ApprovalResult(
                approved=True,
                approved_by="dev_auto_approve",
            )

        # Production: approve LOW, prompt MEDIUM+
        if risk == RiskLevel.LOW:
            return ApprovalResult(approved=True, approved_by="policy_auto")

        return await self._prompt_human(req)

    async def _prompt_human(self, req: ApprovalRequest) -> ApprovalResult:
        """
        In a real deployment this would send a notification and await input.
        For Phase 0 we use a simple terminal prompt via asyncio.
        """
        log.warning(
            "security.approval_required",
            agent=req.agent,
            action=req.action_type,
            risk=req.risk_level.value,
            detail=req.detail,
        )
        # Run blocking input() in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: input(
                f"\n⚠ JARVIS ACTION APPROVAL REQUIRED\n"
                f"  Agent  : {req.agent}\n"
                f"  Action : {req.action_type} [{req.risk_level.value.upper()}]\n"
                f"  Detail : {req.detail}\n"
                f"  Approve? (y/n): "
            ).strip().lower(),
        )
        approved = answer in ("y", "yes")
        log.info(
            "security.human_decision",
            approved=approved,
            agent=req.agent,
            action=req.action_type,
        )
        return ApprovalResult(
            approved=approved,
            approved_by="human" if approved else "human_denied",
            reason=None,
        )


# Singleton
_gateway: SecurityGateway | None = None


def get_gateway() -> SecurityGateway:
    global _gateway
    if _gateway is None:
        _gateway = SecurityGateway()
    return _gateway
