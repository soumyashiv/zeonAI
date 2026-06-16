"""
ZEON Tools — Filesystem
Safe, audited file operations: read, write, search, list, diff.
All writes require security approval (auto in dev mode).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from core.audit_log import AuditLog, ActionType
from core.security_gateway import SecurityGateway
from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()
gateway = SecurityGateway()

# Paths ZEON is allowed to write to (sandboxed)
SAFE_WRITE_ROOTS: list[Path] = [
    Path(cfg.zeon_root) / "workspace",
    Path(cfg.zeon_root) / "data",
    Path(cfg.zeon_root) / "improvements",
]

# Extensions ZEON should never read without extra approval
BLOCKED_EXTENSIONS = {".exe", ".dll", ".bat", ".cmd", ".ps1", ".key", ".pem"}


def _is_safe_write(path: Path) -> bool:
    """Check path is within a sandboxed write root."""
    path = path.resolve()
    return any(
        str(path).startswith(str(root.resolve()))
        for root in SAFE_WRITE_ROOTS
    )


async def read_file(path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Read a file. Returns {content, lines, size_bytes, path}."""
    p = Path(path).resolve()

    if p.suffix.lower() in BLOCKED_EXTENSIONS:
        return {"error": f"Blocked extension: {p.suffix}", "content": None}

    if not p.exists():
        return {"error": f"File not found: {p}", "content": None}

    audit_id = await AuditLog.record(
        agent="filesystem",
        action_type=ActionType.FILE_READ,
        action_detail=str(p),
        approved_by="auto",
    )

    try:
        content = p.read_text(encoding=encoding, errors="replace")
        lines = content.splitlines()
        await AuditLog.complete(audit_id, result=f"{len(lines)} lines", success=True)
        log.info("fs.read", path=str(p), lines=len(lines))
        return {
            "content": content,
            "lines": len(lines),
            "size_bytes": p.stat().st_size,
            "path": str(p),
            "encoding": encoding,
        }
    except Exception as e:
        await AuditLog.complete(audit_id, result=str(e), success=False)
        return {"error": str(e), "content": None}


async def write_file(
    path: str | Path,
    content: str,
    encoding: str = "utf-8",
    create_backup: bool = True,
) -> dict[str, Any]:
    """Write a file. Requires sandbox or security approval."""
    p = Path(path).resolve()

    # Ensure parent exists if within sandbox
    if not _is_safe_write(p):
        approved_result = await gateway.request_approval(
            action_type="file_write",
            agent="filesystem",
            detail=f"write outside sandbox: {p}",
        )
        if not approved_result.approved:
            return {"error": "Write not approved", "written": False}

    p.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing file
    backup_path = None
    if create_backup and p.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = p.with_suffix(f".bak_{ts}{p.suffix}")
        shutil.copy2(str(p), str(backup_path))

    audit_id = await AuditLog.record(
        agent="filesystem",
        action_type=ActionType.FILE_WRITE,
        action_detail=str(p),
        reversible=backup_path is not None,
        reversal_data={"backup": str(backup_path)} if backup_path else None,
    )

    try:
        p.write_text(content, encoding=encoding)
        lines = len(content.splitlines())
        await AuditLog.complete(audit_id, result=f"{lines} lines written", success=True)
        log.info("fs.write", path=str(p), lines=lines, backup=str(backup_path) if backup_path else None)
        return {
            "written": True,
            "path": str(p),
            "lines": lines,
            "backup": str(backup_path) if backup_path else None,
        }
    except Exception as e:
        await AuditLog.complete(audit_id, result=str(e), success=False)
        return {"error": str(e), "written": False}


async def list_dir(path: str | Path, depth: int = 1) -> dict[str, Any]:
    """List directory contents up to `depth` levels."""
    p = Path(path).resolve()
    if not p.exists():
        return {"error": f"Path not found: {p}", "entries": []}

    entries = []
    try:
        for item in sorted(p.iterdir()):
            entry = {
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            }
            if depth > 1 and item.is_dir():
                sub = await list_dir(item, depth - 1)
                entry["children"] = sub.get("entries", [])
            entries.append(entry)
        return {"path": str(p), "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": str(e), "entries": []}


async def search_in_files(
    directory: str | Path,
    pattern: str,
    extensions: list[str] | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Regex/substring search across files in a directory."""
    root = Path(directory).resolve()
    results: list[dict] = []
    exts = {e.lower() for e in (extensions or [".py", ".txt", ".md", ".json"])}

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    for filepath in root.rglob("*"):
        if len(results) >= max_results:
            break
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in exts:
            continue

        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if compiled.search(line):
                    results.append({
                        "file": str(filepath.relative_to(root)),
                        "line": i,
                        "content": line.strip()[:200],
                    })
                    if len(results) >= max_results:
                        break
        except Exception:
            continue

    log.info("fs.search", pattern=pattern, matches=len(results))
    return results


async def delete_file(path: str | Path) -> dict[str, Any]:
    """Delete a file — always requires approval outside dev mode."""
    p = Path(path).resolve()
    approved_result = await gateway.request_approval(
        action_type="file_delete",
        agent="filesystem",
        detail=f"delete: {p}",
    )
    if not approved_result.approved:
        return {"error": "Delete not approved", "deleted": False}

    audit_id = await AuditLog.record(
        agent="filesystem",
        action_type=ActionType.FILE_DELETE,
        action_detail=str(p),
    )
    try:
        p.unlink(missing_ok=True)
        await AuditLog.complete(audit_id, result="deleted", success=True)
        return {"deleted": True, "path": str(p)}
    except Exception as e:
        await AuditLog.complete(audit_id, result=str(e), success=False)
        return {"error": str(e), "deleted": False}
