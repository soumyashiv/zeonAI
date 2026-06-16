"""
ZEON Computer Brain — Unified facade for all computer-control tools.
Brings together filesystem, shell, and browser into one interface.
"""
from __future__ import annotations

from typing import Any

import structlog

from tools.filesystem import (
    read_file, write_file, list_dir,
    search_in_files, delete_file,
)
from tools.shell import run_shell, classify_command

log = structlog.get_logger(__name__)


class ComputerBrain:
    """Single entry point for all OS-level computer control actions."""

    # ── Filesystem ─────────────────────────────────────────────────────────

    async def read(self, path: str) -> dict[str, Any]:
        return await read_file(path)

    async def write(self, path: str, content: str, backup: bool = True) -> dict[str, Any]:
        return await write_file(path, content, create_backup=backup)

    async def ls(self, path: str, depth: int = 1) -> dict[str, Any]:
        return await list_dir(path, depth=depth)

    async def search(
        self, directory: str, pattern: str,
        extensions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await search_in_files(directory, pattern, extensions=extensions)

    async def delete(self, path: str) -> dict[str, Any]:
        return await delete_file(path)

    # ── Shell ──────────────────────────────────────────────────────────────

    async def shell(self, command: str, cwd: str | None = None,
                    timeout: int = 30) -> dict[str, Any]:
        return await run_shell(command, cwd=cwd, timeout=timeout)

    def classify_shell(self, command: str) -> str:
        return classify_command(command)

    # ── Browser (stub — Playwright in Phase 6) ────────────────────────────

    async def browse(self, url: str, action: str = "screenshot") -> dict[str, Any]:
        """Navigate browser. Full implementation uses Playwright in Phase 6."""
        try:
            from tools.browser import browse_page
            return await browse_page(url, action=action)
        except ImportError:
            log.warning("computer.browser_unavailable",
                        hint="pip install playwright && playwright install chromium")
            return {"error": "Browser tool not yet installed", "url": url}

    # ── Convenience ────────────────────────────────────────────────────────

    async def python_eval(self, code: str) -> dict[str, Any]:
        """
        Run a Python snippet in a subprocess.
        Safe alternative to eval() — full sandbox via shell tool.
        """
        import tempfile, json as jsonlib
        from pathlib import Path

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name

        result = await run_shell(f"python \"{tmp}\"", timeout=20)
        Path(tmp).unlink(missing_ok=True)
        return result


_brain: ComputerBrain | None = None


def get_computer_brain() -> ComputerBrain:
    global _brain
    if _brain is None:
        _brain = ComputerBrain()
    return _brain
