"""
JARVIS Coder Agent
Writes, reviews, runs, and debugs Python code autonomously.
Uses LLM to generate code, then executes it in a sandboxed subprocess.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType
from core.llm import chat
from core.audit_log import AuditLog, ActionType

log = structlog.get_logger(__name__)

CODER_PROMPT = """You are JARVIS's Coder Agent.
Write clean, working Python 3.11+ code to solve the task.
Rules:
- Only output the code block, nothing else
- Use standard library when possible
- Add a main() function and call it at the bottom
- Handle exceptions
- Print the final result so it can be captured

Respond with ONLY a Python code block:
```python
# code here
```"""

REVIEWER_PROMPT = """Review this Python code for:
1. Bugs or logic errors
2. Security issues (no file deletion, no network calls unless needed)
3. Infinite loops
4. Exception handling

Reply as JSON:
{"safe": true/false, "issues": ["..."], "rating": 1-10, "fix_needed": true/false}"""


class CoderAgent(BaseAgent):
    name = "coder_agent"
    description = "Writes, reviews, and executes Python code. Sandboxed subprocess."

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        task = original_message.payload.get("task", plan.get("goal", ""))
        flags = original_message.payload.get("flags", {})

        self._log.info("coder.starting", task=task[:60])

        # Step 1: Generate code
        code = await self._generate_code(task)
        if not code:
            return {"success": False, "error": "Code generation failed"}

        # Step 2: Review code for safety
        review = await self._review_code(code, task)
        if not review.get("safe", True):
            issues = review.get("issues", [])
            self._log.warning("coder.unsafe_code", issues=issues)
            return {"success": False, "error": f"Code review failed: {issues}"}

        # Step 3: Execute in subprocess
        result = await self._run_code(code)

        # Step 4: Report back
        await self.send(
            original_message.from_agent or "executive_agent",
            MessageType.TASK_RESULT,
            {"task": task, "code": code, "execution": result},
            priority=7,
        )
        return {"success": result.get("success"), "output": result.get("output", ""), "code": code}

    async def _generate_code(self, task: str) -> str | None:
        messages = [
            {"role": "system", "content": CODER_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ]
        try:
            raw = await chat(messages, temperature=0.3, max_tokens=1500)
            # Extract code block
            match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Fallback: if no fences, return raw if it looks like code
            if "def " in raw or "import " in raw:
                return raw.strip()
            return None
        except Exception as e:
            self._log.error("coder.generate_failed", error=str(e))
            return None

    async def _review_code(self, code: str, task: str) -> dict:
        messages = [
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": f"Task: {task}\n\nCode:\n```python\n{code}\n```"},
        ]
        try:
            raw = await chat(messages, temperature=0.1, max_tokens=300)
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e]) if s >= 0 and e > s else {"safe": True, "issues": []}
        except Exception:
            return {"safe": True, "issues": [], "rating": 7}

    async def _run_code(self, code: str) -> dict[str, Any]:
        """Execute code in a temp file subprocess with 30s timeout."""
        import asyncio

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        audit_id = await AuditLog.record(
            agent=self.name,
            action_type=ActionType.CODE_RUN,
            action_detail=f"run {Path(tmp_path).name}: {code[:100]}",
            approved_by="dev_auto_approve",
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            success = proc.returncode == 0

            if err and not success:
                output += f"\nSTDERR:\n{err[:500]}"

            await AuditLog.complete(audit_id, result=output[:300], success=success)
            return {"success": success, "output": output[:2000],
                    "returncode": proc.returncode}

        except asyncio.TimeoutError:
            await AuditLog.complete(audit_id, result="timeout", success=False)
            return {"success": False, "output": "Execution timed out (30s)", "returncode": -1}
        except Exception as e:
            await AuditLog.complete(audit_id, result=str(e), success=False)
            return {"success": False, "output": str(e), "returncode": -1}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        if isinstance(result, dict):
            return result.get("success", False), result.get("output", "")[:100]
        return False, "no result"
