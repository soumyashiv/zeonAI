"""
ZEON Tools — Shell
Sandboxed subprocess execution with timeout, approval, and output capture.
Blocklist prevents destructive commands (rm -rf, format, etc.).
"""
from __future__ import annotations

import asyncio
import re
import shlex
from typing import Any

import structlog

from core.audit_log import AuditLog, ActionType
from core.security_gateway import SecurityGateway
from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()
gateway = SecurityGateway()

# Commands that are always blocked, even in dev mode
HARD_BLOCKED: list[re.Pattern] = [
    re.compile(r"\brm\s+-rf\b"),
    re.compile(r"\bformat\s+\w:"),        # Windows format C:
    re.compile(r"\bdel\s+/[sS]\b"),           # del /s recursive
    re.compile(r"\brd\s+/[sS]\b"),            # rd /s
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bpoweroff\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),                # disk wipe
    re.compile(r"::\s*$"),                    # Windows infinite loop trick
    re.compile(r"while\s*true\s*;?\s*do"),   # bash infinite loop
]

# Commands requiring extra risk classification
HIGH_RISK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bcurl\b.*\|\s*(bash|sh|python)"),
    re.compile(r"\bwget\b.*\|\s*(bash|sh|python)"),
    re.compile(r"\bnetsh\b"),
    re.compile(r"\breg\s+(add|delete)\b"),
    re.compile(r"\bschtasks\b"),
]

DEFAULT_TIMEOUT = 30   # seconds


def classify_command(command: str) -> str:
    """Return 'blocked', 'high', 'medium', or 'low'."""
    cmd_lower = command.lower()
    for pattern in HARD_BLOCKED:
        if pattern.search(cmd_lower):
            return "blocked"
    for pattern in HIGH_RISK_PATTERNS:
        if pattern.search(cmd_lower):
            return "high"
    if any(kw in cmd_lower for kw in ["mkdir", "copy", "move", "ren", "attrib"]):
        return "medium"
    return "low"


async def run_shell(
    command: str,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    shell: bool = True,
) -> dict[str, Any]:
    """
    Execute a shell command safely.
    Returns: {stdout, stderr, returncode, success, command, duration_ms}
    """
    risk = classify_command(command)

    if risk == "blocked":
        log.warning("shell.hard_blocked", command=command[:80])
        return {
            "stdout": "",
            "stderr": "Command permanently blocked by ZEON safety policy.",
            "returncode": -1,
            "success": False,
            "command": command,
        }

    # Map our risk string to gateway action_type
    _action_type_map = {"low": "shell_exec", "medium": "shell_exec", "high": "shell_exec"}
    approved_result = await gateway.request_approval(
        action_type="shell_exec",
        agent="shell_tool",
        detail=f"[{risk.upper()}] {command[:200]}",
        payload={"command": command, "risk": risk},
    )
    if not approved_result.approved:
        return {
            "stdout": "",
            "stderr": "Execution not approved.",
            "returncode": -1,
            "success": False,
            "command": command,
        }

    audit_id = await AuditLog.record(
        agent="shell_tool",
        action_type=ActionType.SHELL_EXEC,
        action_detail=command[:500],
    )

    import time
    t0 = time.monotonic()

    try:
        if shell:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        duration_ms = int((time.monotonic() - t0) * 1000)

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        success = proc.returncode == 0

        await AuditLog.complete(
            audit_id,
            result=out[:300] or err[:300],
            success=success,
        )

        log.info("shell.executed",
                 command=command[:60],
                 returncode=proc.returncode,
                 duration_ms=duration_ms)

        return {
            "stdout": out[:8000],
            "stderr": err[:2000],
            "returncode": proc.returncode,
            "success": success,
            "command": command,
            "duration_ms": duration_ms,
        }

    except asyncio.TimeoutError:
        await AuditLog.complete(audit_id, result="timeout", success=False)
        log.warning("shell.timeout", command=command[:60], timeout=timeout)
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
            "success": False,
            "command": command,
        }
    except Exception as e:
        await AuditLog.complete(audit_id, result=str(e), success=False)
        log.error("shell.error", command=command[:60], error=str(e))
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "success": False,
            "command": command,
        }
