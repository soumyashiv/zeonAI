"""
ZEON Self-Improvement — Git Branch Manager
Manages a dedicated git branch for each self-improvement cycle.
Enables approval-gated merges: ZEON proposes changes on a branch,
a human (or auto in dev) reviews and merges.
"""
from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from core.config import get_config
from core.security_gateway import get_gateway

log = structlog.get_logger(__name__)
cfg = get_config()


@dataclass
class ImprovementBranch:
    name: str
    base: str
    created_at: str
    description: str
    status: str   # "open" | "merged" | "rejected"


class ZeonGit:
    """
    Git abstraction for ZEON's self-improvement workflow.
    Each improvement cycle gets its own branch for review.
    """

    BRANCH_PREFIX = "zeon/improve"
    BASE_BRANCH   = "main"

    def __init__(self, repo_path: str | None = None) -> None:
        self._repo = Path(repo_path or cfg.jarvis_root)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self._repo),
            capture_output=True,
            text=True,
            check=check,
        )

    def current_branch(self) -> str:
        result = self._run("branch", "--show-current")
        return result.stdout.strip() or "main"

    def create_improvement_branch(self, description: str) -> ImprovementBranch:
        """Create a new improvement branch from main."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_desc = description[:30].lower().replace(" ", "_").replace("/", "-")
        branch_name = f"{self.BRANCH_PREFIX}/{ts}_{safe_desc}"

        try:
            # Ensure we're on main first
            self._run("checkout", self.BASE_BRANCH, check=False)
            self._run("checkout", "-b", branch_name)
            log.info("zeon_git.branch_created", branch=branch_name)
        except subprocess.CalledProcessError as e:
            log.error("zeon_git.branch_failed",
                      branch=branch_name, error=e.stderr)
            raise

        return ImprovementBranch(
            name=branch_name,
            base=self.BASE_BRANCH,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            status="open",
        )

    def commit_improvement(self, message: str, files: list[str] | None = None) -> str:
        """Stage and commit improvements. Returns commit hash."""
        if files:
            for f in files:
                self._run("add", f)
        else:
            self._run("add", "-A")

        result = self._run("commit", "-m", f"[ZEON-IMPROVE] {message}", check=False)
        if result.returncode != 0:
            if "nothing to commit" in result.stdout + result.stderr:
                log.info("zeon_git.nothing_to_commit")
                return ""
            raise subprocess.CalledProcessError(
                result.returncode, "git commit", result.stderr
            )

        # Get commit hash
        hash_result = self._run("rev-parse", "--short", "HEAD")
        return hash_result.stdout.strip()

    async def request_merge(
        self, branch: ImprovementBranch, summary: str
    ) -> bool:
        """
        Request a merge of the improvement branch into main.
        In dev mode: auto-approved.
        In prod mode: requires human confirmation.
        """
        gateway = get_gateway()
        approval = await gateway.request_approval(
            action_type="self_improve",
            agent="zeon_git",
            detail=f"Merge branch '{branch.name}' — {summary}",
            payload={"branch": branch.name, "summary": summary},
        )

        if approval.approved:
            try:
                self._run("checkout", self.BASE_BRANCH)
                self._run("merge", "--no-ff", branch.name,
                          "-m", f"merge: {branch.description}")
                log.info("zeon_git.merged", branch=branch.name)
                return True
            except subprocess.CalledProcessError as e:
                log.error("zeon_git.merge_failed",
                          branch=branch.name, error=e.stderr)
                return False
        else:
            log.info("zeon_git.merge_rejected", branch=branch.name)
            return False

    def list_improvement_branches(self) -> list[str]:
        """List all open improvement branches."""
        result = self._run("branch", "--list", f"{self.BRANCH_PREFIX}/*", check=False)
        return [
            b.strip().lstrip("* ")
            for b in result.stdout.splitlines()
            if b.strip()
        ]

    def diff_summary(self, branch: str) -> str:
        """Get a short diff summary of an improvement branch vs main."""
        result = self._run(
            "diff", "--stat", f"{self.BASE_BRANCH}...{branch}",
            check=False,
        )
        return result.stdout.strip()[:500] or "No changes."
