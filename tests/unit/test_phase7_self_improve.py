"""
ZEON Self-Improvement Tests
Skill compiler parsing, improvement report, ZeonGit, and self_improvement_agent.
All LLM + git + memory calls are mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── SkillCompiler — task type inference ──────────────────────────────────────

def test_infer_task_type_research():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    assert sc._infer_task_type("I searched and found documentation") == "research"


def test_infer_task_type_coding():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    assert sc._infer_task_type("Generated a Python function class") == "coding"


def test_infer_task_type_filesystem():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    assert sc._infer_task_type("Read and wrote a file to directory") == "filesystem"


def test_infer_task_type_planning():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    assert sc._infer_task_type("Created a plan with steps") == "planning"


def test_infer_task_type_general():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    assert sc._infer_task_type("Did something unrecognised") == "general"


# ── SkillCompiler — clustering ───────────────────────────────────────────────

def test_cluster_episodes():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    episodes = [
        {"task_type": "research", "outcome": "success"},
        {"task_type": "research", "outcome": "success"},
        {"task_type": "coding",   "outcome": "success"},
    ]
    clusters = sc._cluster_episodes(episodes)
    assert "research" in clusters
    assert len(clusters["research"]) == 2
    assert "coding" in clusters


def test_cluster_episodes_empty():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    clusters = sc._cluster_episodes([])
    assert clusters == {}


# ── SkillCompiler — parse skill response ─────────────────────────────────────

def test_parse_skill_response_valid():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    response = (
        "NAME: web_search_skill\n"
        "DESCRIPTION: Search the web for information.\n"
        "STEPS: Form query | Execute search | Parse results\n"
        "TRIGGERS: user asks for information | research task\n"
        "OUTCOME: Relevant information retrieved\n"
        "CONFIDENCE: 0.85"
    )
    skill = sc._parse_skill_response(response, "research", [{"id": "ep1"}])
    assert skill is not None
    assert skill.name == "web_search_skill"
    assert skill.confidence == 0.85
    assert len(skill.steps) == 3
    assert "ep1" in skill.source_episode_ids


def test_parse_skill_response_missing_fields():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    # Minimal response — should not crash
    response = "NAME: minimal_skill"
    skill = sc._parse_skill_response(response, "general", [])
    assert skill is not None
    assert skill.name == "minimal_skill"
    assert skill.confidence == 0.7   # default


def test_parse_skill_response_bad_confidence():
    from improvements.skill_compiler import SkillCompiler
    sc = SkillCompiler()
    response = "NAME: test\nCONFIDENCE: not_a_number"
    skill = sc._parse_skill_response(response, "general", [])
    assert skill.confidence == 0.7   # fallback default


# ── SkillCompiler — compile (mocked memory + LLM) ────────────────────────────

@pytest.mark.asyncio
async def test_skill_compiler_compile_no_episodes():
    from improvements.skill_compiler import SkillCompiler

    mock_memory = MagicMock()
    mock_memory.episodic.search = AsyncMock(return_value=[])

    sc = SkillCompiler()
    report = await sc.compile(memory_brain=mock_memory, max_episodes=10)

    assert report.episodes_analysed == 0
    assert report.skills_extracted == 0
    assert "No episodes" in report.summary


@pytest.mark.asyncio
async def test_skill_compiler_compile_with_episodes():
    from improvements.skill_compiler import SkillCompiler

    # Create mock episodes
    mock_ep = MagicMock()
    mock_ep.content = "Searched for information and found results successfully"
    mock_ep.importance = 0.8

    mock_memory = MagicMock()
    mock_memory.episodic.search = AsyncMock(return_value=[mock_ep, mock_ep])
    mock_memory.procedural.register_skill = AsyncMock()

    llm_response = (
        "NAME: research_skill\n"
        "DESCRIPTION: Performs web research.\n"
        "STEPS: Query | Search | Parse\n"
        "TRIGGERS: research request\n"
        "OUTCOME: Information retrieved\n"
        "CONFIDENCE: 0.8"
    )

    with patch("improvements.skill_compiler.chat",
               new=AsyncMock(return_value=llm_response)):
        sc = SkillCompiler()
        report = await sc.compile(memory_brain=mock_memory, max_episodes=10)

    assert report.episodes_analysed == 2
    assert report.skills_extracted >= 1
    assert report.skills_registered >= 1


# ── ZeonGit ───────────────────────────────────────────────────────────────────

def test_zeon_git_current_branch(tmp_path):
    """ZeonGit.current_branch() returns a string."""
    import subprocess
    # Init a temp git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(tmp_path), capture_output=True)
    # Create initial commit so branch exists
    (tmp_path / "README.md").write_text("test")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"],
                   cwd=str(tmp_path), capture_output=True)

    from improvements.zeon_git import ZeonGit
    zg = ZeonGit(repo_path=str(tmp_path))
    branch = zg.current_branch()
    assert isinstance(branch, str)
    assert len(branch) > 0


def test_zeon_git_create_branch(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"],
                   cwd=str(tmp_path), capture_output=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"],
                   cwd=str(tmp_path), capture_output=True)

    from improvements.zeon_git import ZeonGit
    zg = ZeonGit(repo_path=str(tmp_path))
    branch = zg.create_improvement_branch("add new skill")

    assert branch.name.startswith("zeon/improve/")
    assert branch.status == "open"
    assert branch.description == "add new skill"


def test_zeon_git_list_branches_empty(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"],
                   cwd=str(tmp_path), capture_output=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"],
                   cwd=str(tmp_path), capture_output=True)

    from improvements.zeon_git import ZeonGit
    zg = ZeonGit(repo_path=str(tmp_path))
    branches = zg.list_improvement_branches()
    assert isinstance(branches, list)


def test_zeon_git_commit_improvement(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"],
                   cwd=str(tmp_path), capture_output=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "i"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"],
                   cwd=str(tmp_path), capture_output=True)

    from improvements.zeon_git import ZeonGit
    zg = ZeonGit(repo_path=str(tmp_path))
    zg.create_improvement_branch("test skill")

    # Add a file on the branch
    (tmp_path / "new_skill.py").write_text("# skill")
    commit_hash = zg.commit_improvement("Add test skill")
    assert isinstance(commit_hash, str)


# ── ImprovementReport ─────────────────────────────────────────────────────────

def test_improvement_report_fields():
    from improvements.skill_compiler import ImprovementReport
    report = ImprovementReport(
        skills_extracted=3,
        skills_registered=2,
        episodes_analysed=50,
        patterns_found=5,
        failed_patterns=1,
        summary="Compiled 3 skills from 50 episodes.",
    )
    assert report.skills_extracted == 3
    assert report.skills_registered == 2
    assert "Compiled" in report.summary


# ── SelfImprovementAgent integration ─────────────────────────────────────────

def test_self_improvement_agent_imports():
    """Verify the self_improvement_agent module loads cleanly."""
    import agents.self_improvement_agent as sia
    assert hasattr(sia, "SelfImprovementAgent")
