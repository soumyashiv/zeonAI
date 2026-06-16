"""
Phase 5 — Computer Control Tests
Filesystem, shell classifier, ComputerBrain.
All actual subprocess/disk calls sandboxed via tmp_path.
"""
from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch


# ── Filesystem — read ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello ZEON\nLine 2")

    from tools.filesystem import read_file
    result = await read_file(str(test_file))
    assert result["content"] == "Hello ZEON\nLine 2"
    assert result["lines"] == 2

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_read_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "nf.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from tools.filesystem import read_file
    result = await read_file(str(tmp_path / "nonexistent.txt"))
    assert "error" in result
    assert result["content"] is None

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_read_blocked_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "blk.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    bad_file = tmp_path / "secret.key"
    bad_file.write_text("secret")

    from tools.filesystem import read_file
    result = await read_file(str(bad_file))
    assert "error" in result

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── Filesystem — write ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "wr.db"))
    monkeypatch.setenv("ZEON_ROOT", str(tmp_path))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    # Write inside workspace (sandboxed)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "output.txt"

    import tools.filesystem as fs_mod
    importlib.reload(fs_mod)
    fs_mod.SAFE_WRITE_ROOTS = [workspace]

    result = await fs_mod.write_file(str(target), "Hello World")
    assert result["written"] is True
    assert target.read_text() == "Hello World"

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── Filesystem — list_dir ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ls.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    (tmp_path / "file_a.txt").write_text("a")
    (tmp_path / "file_b.py").write_text("b")
    sub = tmp_path / "subdir"
    sub.mkdir()

    from tools.filesystem import list_dir
    result = await list_dir(str(tmp_path))
    assert result["count"] == 3
    names = {e["name"] for e in result["entries"]}
    assert "file_a.txt" in names
    assert "subdir" in names
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_list_dir_not_found():
    from tools.filesystem import list_dir
    result = await list_dir("/nonexistent/path/xyz")
    assert "error" in result


# ── Filesystem — search ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_in_files(tmp_path):
    (tmp_path / "code.py").write_text("def hello():\n    print('ZEON')\n")
    (tmp_path / "notes.txt").write_text("ZEON is an AI\n")
    (tmp_path / "other.md").write_text("No match here\n")

    from tools.filesystem import search_in_files
    results = await search_in_files(
        str(tmp_path), "ZEON",
        extensions=[".py", ".txt", ".md"]
    )
    assert len(results) == 2
    files = {r["file"] for r in results}
    assert any("code.py" in f for f in files)
    assert any("notes.txt" in f for f in files)


@pytest.mark.asyncio
async def test_search_no_match(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing here")
    from tools.filesystem import search_in_files
    results = await search_in_files(str(tmp_path), "ZEON", extensions=[".txt"])
    assert results == []


# ── Shell — classifier ────────────────────────────────────────────────────────

def test_classify_hard_blocked_rm():
    from tools.shell import classify_command
    assert classify_command("rm -rf /") == "blocked"


def test_classify_hard_blocked_shutdown():
    from tools.shell import classify_command
    assert classify_command("shutdown /s /t 0") == "blocked"


def test_classify_hard_blocked_format():
    from tools.shell import classify_command
    assert classify_command("format C:") == "blocked"


def test_classify_high_risk_pip():
    from tools.shell import classify_command
    assert classify_command("pip install requests") == "high"


def test_classify_medium_mkdir():
    from tools.shell import classify_command
    assert classify_command("mkdir new_folder") == "medium"


def test_classify_low_risk():
    from tools.shell import classify_command
    assert classify_command("python --version") == "low"


def test_classify_low_echo():
    from tools.shell import classify_command
    assert classify_command("echo Hello World") == "low"


# ── Shell — execution ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shell_blocked_command(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "sh.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from tools.shell import run_shell
    result = await run_shell("rm -rf /tmp")
    assert result["success"] is False
    assert "blocked" in result["stderr"].lower()

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_shell_low_risk_executes(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "sh2.db"))
    monkeypatch.setenv("ZEON_DEV_AUTO_APPROVE", "true")
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    import tools.shell as sh_mod
    importlib.reload(sh_mod)

    result = await sh_mod.run_shell("python --version")
    assert result["returncode"] == 0
    assert "Python" in result["stdout"] or "Python" in result["stderr"]

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_shell_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "sh3.db"))
    monkeypatch.setenv("ZEON_DEV_AUTO_APPROVE", "true")
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    import tools.shell as sh_mod
    importlib.reload(sh_mod)

    result = await sh_mod.run_shell(
        'python -c "import time; time.sleep(10)"', timeout=1
    )
    assert result["success"] is False
    assert "timeout" in result["stderr"].lower() or result["returncode"] == -1

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── ComputerBrain facade ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_computer_brain_read(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "cb.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    f = tmp_path / "brain_test.txt"
    f.write_text("ComputerBrain read test")

    # Reload filesystem to pick up fresh AuditLog
    import tools.filesystem as fs_mod
    importlib.reload(fs_mod)

    import brains.computer as cb_mod
    importlib.reload(cb_mod)
    cb_mod._brain = None

    cb = cb_mod.get_computer_brain()
    result = await cb.read(str(f))
    assert result["content"] == "ComputerBrain read test"

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


def test_computer_brain_classify_shell():
    from brains.computer import get_computer_brain
    cb = get_computer_brain()
    assert cb.classify_shell("rm -rf /") == "blocked"
    assert cb.classify_shell("echo hi") == "low"


@pytest.mark.asyncio
async def test_computer_brain_browse_stub():
    """Browser stub returns error message when playwright not installed."""
    from brains.computer import ComputerBrain
    cb = ComputerBrain()
    result = await cb.browse("https://example.com")
    # Either returns error (stub) or actual result if playwright installed
    assert isinstance(result, dict)
    assert "url" in result or "error" in result or "content" in result
