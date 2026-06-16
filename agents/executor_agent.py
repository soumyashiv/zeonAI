"""
ZEON Executor Agent
General-purpose task executor. Runs plan steps one by one,
handles tool calls, and reports results.
"""
from __future__ import annotations

from typing import Any
import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.audit_log import AuditLog, ActionType
from core.security_gateway import get_gateway

log = structlog.get_logger(__name__)


class ExecutorAgent(BaseAgent):
    name = "executor_agent"
    description = "Executes plan steps. Calls tools, runs actions, reports results."

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        plan_data = original_message.payload.get("plan", plan)
        steps = plan_data.get("steps", [])
        task = plan_data.get("goal", original_message.payload.get("task", ""))

        results = []
        all_success = True

        for step in steps:
            step_id = step.get("id", "?")
            action = step.get("action", "")
            tool = step.get("tool", "none")

            self._log.info("executor.step", id=step_id, action=action[:60], tool=tool)

            # Get security approval for non-trivial tools
            if tool in ("shell", "file", "browser"):
                approval = await self.request_approval(
                    f"shell_exec" if tool == "shell" else f"file_write" if tool == "file" else "browser_ctrl",
                    detail=f"Step {step_id}: {action[:100]}",
                )
                if not approval.approved:
                    results.append({"step": step_id, "status": "denied", "action": action})
                    all_success = False
                    continue

            # Audit the action
            audit_id = await AuditLog.record(
                agent=self.name,
                action_type=ActionType.SKILL_RUN,
                action_detail=f"step {step_id}: {action[:200]}",
                approved_by="dev_auto_approve",
            )

            # Execute based on tool type
            step_result = await self._run_step(step)
            results.append(step_result)

            await AuditLog.complete(
                audit_id,
                result=str(step_result.get("output", ""))[:300],
                success=step_result.get("success", True),
            )

            if not step_result.get("success", True):
                all_success = False
                self._log.warning("executor.step_failed", step=step_id)

        final = {"steps_run": len(results), "all_success": all_success, "results": results}

        # Notify originator
        await self.send(
            original_message.from_agent or "executive_agent",
            MessageType.TASK_RESULT,
            {"task": task, "execution_result": final},
            priority=7,
        )
        return final

    async def _run_step(self, step: dict) -> dict[str, Any]:
        """Route step to appropriate tool handler."""
        tool = step.get("tool", "none")
        action = step.get("action", "")

        try:
            if tool == "none":
                # Pure reasoning / information step
                return {"step": step.get("id"), "tool": "none",
                        "output": f"Noted: {action}", "success": True}

            elif tool == "search":
                return await self._tool_search(action)

            elif tool == "file":
                return await self._tool_file(step)

            elif tool == "shell":
                return await self._tool_shell(action)

            elif tool == "code":
                return {"step": step.get("id"), "tool": "code",
                        "output": "Coder agent required (Phase 6)", "success": True}

            elif tool == "browser":
                return {"step": step.get("id"), "tool": "browser",
                        "output": "Browser tool (Phase 6)", "success": True}

            else:
                return {"step": step.get("id"), "tool": tool,
                        "output": f"Unknown tool: {tool}", "success": False}

        except Exception as e:
            return {"step": step.get("id"), "tool": tool,
                    "output": str(e), "success": False}

    async def _tool_search(self, query: str) -> dict:
        """DuckDuckGo search (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=3))
            )
            snippets = [f"{r.get('title','')}: {r.get('body','')[:150]}" for r in results]
            return {"tool": "search", "output": "\n".join(snippets), "success": True}
        except ImportError:
            return {"tool": "search", "output": f"[duckduckgo-search not installed]", "success": False}
        except Exception as e:
            return {"tool": "search", "output": str(e), "success": False}

    async def _tool_file(self, step: dict) -> dict:
        """Read a file."""
        import aiofiles
        path = step.get("path", "")
        mode = step.get("mode", "read")
        if not path:
            return {"tool": "file", "output": "No path specified", "success": False}
        try:
            if mode == "read":
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    content = await f.read()
                return {"tool": "file", "output": content[:2000], "success": True}
            return {"tool": "file", "output": "Write requires explicit approval", "success": False}
        except Exception as e:
            return {"tool": "file", "output": str(e), "success": False}

    async def _tool_shell(self, command: str) -> dict:
        """Run a shell command in a subprocess."""
        import asyncio
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\nSTDERR: " + stderr.decode("utf-8", errors="replace")[:200]
            return {"tool": "shell", "output": output[:2000],
                    "success": proc.returncode == 0, "returncode": proc.returncode}
        except asyncio.TimeoutError:
            return {"tool": "shell", "output": "Command timed out (30s)", "success": False}
        except Exception as e:
            return {"tool": "shell", "output": str(e), "success": False}

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        if isinstance(result, dict):
            return result.get("all_success", False), f"{result.get('steps_run', 0)} steps run"
        return False, "no result"
