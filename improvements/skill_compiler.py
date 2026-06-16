"""
ZEON Self-Improvement — Skill Compiler
Analyses episodic memory to identify recurring successful patterns
and automatically extracts them as reusable procedural skills.

Pipeline:
  1. Fetch recent episodes from episodic memory
  2. Cluster by task_type / outcome
  3. For successful clusters → extract skill via LLM
  4. Register in procedural memory
  5. Log improvement to audit trail
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog

from core.llm import chat
from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


@dataclass
class ExtractedSkill:
    name: str
    description: str
    steps: list[str]
    trigger_conditions: list[str]
    expected_outcome: str
    confidence: float        # 0.0 – 1.0
    source_episode_ids: list[str]


@dataclass
class ImprovementReport:
    skills_extracted: int
    skills_registered: int
    episodes_analysed: int
    patterns_found: int
    failed_patterns: int
    summary: str


class SkillCompiler:
    """
    Mines episodic memory for successful patterns and registers them
    as procedural skills.
    """

    MIN_OCCURRENCES = 2       # pattern must appear ≥ this many times
    MIN_SUCCESS_RATE = 0.7    # episode cluster success rate threshold

    async def compile(
        self,
        memory_brain=None,
        max_episodes: int = 100,
    ) -> ImprovementReport:
        """
        Main entry point. Returns an ImprovementReport.
        """
        if memory_brain is None:
            from brains.memory import get_memory_brain
            memory_brain = get_memory_brain()

        log.info("skill_compiler.start", max_episodes=max_episodes)

        # 1. Fetch recent successful episodes
        episodes = await self._fetch_episodes(memory_brain, max_episodes)
        if not episodes:
            return ImprovementReport(0, 0, 0, 0, 0, "No episodes to analyse.")

        # 2. Cluster episodes by task type
        clusters = self._cluster_episodes(episodes)
        log.info("skill_compiler.clustered", clusters=len(clusters))

        # 3. Extract skills from high-success clusters
        extracted: list[ExtractedSkill] = []
        failed = 0
        for task_type, cluster_episodes in clusters.items():
            if len(cluster_episodes) < self.MIN_OCCURRENCES:
                continue
            success_rate = sum(
                1 for e in cluster_episodes if e.get("outcome") == "success"
            ) / len(cluster_episodes)
            if success_rate < self.MIN_SUCCESS_RATE:
                continue

            skill = await self._extract_skill(task_type, cluster_episodes)
            if skill:
                extracted.append(skill)
            else:
                failed += 1

        # 4. Register extracted skills
        registered = 0
        for skill in extracted:
            try:
                await memory_brain.procedural.register_skill(
                    name=skill.name,
                    description=skill.description,
                    code_snippet="\n".join(f"  {i+1}. {s}" for i, s in enumerate(skill.steps)),
                    tags=skill.trigger_conditions[:3],
                )
                registered += 1
                log.info("skill_compiler.skill_registered", name=skill.name)
            except Exception as e:
                log.error("skill_compiler.register_failed",
                          skill=skill.name, error=str(e))

        summary = (
            f"Analysed {len(episodes)} episodes, found {len(clusters)} patterns, "
            f"extracted {len(extracted)} skills, registered {registered}."
        )

        log.info("skill_compiler.complete",
                 episodes=len(episodes),
                 skills_extracted=len(extracted),
                 registered=registered)

        return ImprovementReport(
            skills_extracted=len(extracted),
            skills_registered=registered,
            episodes_analysed=len(episodes),
            patterns_found=len(clusters),
            failed_patterns=failed,
            summary=summary,
        )

    async def _fetch_episodes(
        self, memory_brain, max_episodes: int
    ) -> list[dict[str, Any]]:
        try:
            episodes = await memory_brain.episodic.search(
                query="success", limit=max_episodes
            )
            return [
                {
                    "id": str(e.id) if hasattr(e, "id") else str(id(e)),
                    "content": e.content if hasattr(e, "content") else str(e),
                    "task_type": self._infer_task_type(
                        e.content if hasattr(e, "content") else str(e)
                    ),
                    "outcome": "success",
                    "importance": getattr(e, "importance", 0.5),
                }
                for e in episodes
            ]
        except Exception as e:
            log.error("skill_compiler.fetch_failed", error=str(e))
            return []

    def _infer_task_type(self, content: str) -> str:
        """Heuristic task type inference from episode content."""
        content_lower = content.lower()
        if any(w in content_lower for w in ["search", "research", "found"]):
            return "research"
        if any(w in content_lower for w in ["code", "python", "function", "class"]):
            return "coding"
        if any(w in content_lower for w in ["file", "read", "write", "directory"]):
            return "filesystem"
        if any(w in content_lower for w in ["memory", "store", "retrieve"]):
            return "memory_ops"
        if any(w in content_lower for w in ["plan", "task", "step"]):
            return "planning"
        return "general"

    def _cluster_episodes(
        self, episodes: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group episodes by inferred task_type."""
        clusters: dict[str, list] = {}
        for ep in episodes:
            key = ep.get("task_type", "general")
            clusters.setdefault(key, []).append(ep)
        return clusters

    async def _extract_skill(
        self,
        task_type: str,
        episodes: list[dict[str, Any]],
    ) -> ExtractedSkill | None:
        """Use LLM to extract a reusable skill from a cluster of episodes."""
        sample = episodes[:5]
        episode_texts = "\n---\n".join(
            ep.get("content", "")[:300] for ep in sample
        )

        prompt = [
            {"role": "system", "content":
             "You are ZEON's skill extraction system. "
             "Analyse successful task episodes and extract a reusable skill."},
            {"role": "user", "content":
             f"Task type: {task_type}\n\n"
             f"Successful episodes:\n{episode_texts}\n\n"
             f"Extract a reusable skill. Reply in this exact format:\n"
             f"NAME: <short skill name>\n"
             f"DESCRIPTION: <one sentence>\n"
             f"STEPS: <step1> | <step2> | <step3>\n"
             f"TRIGGERS: <condition1> | <condition2>\n"
             f"OUTCOME: <expected result>\n"
             f"CONFIDENCE: <0.0-1.0>"},
        ]

        try:
            response = await chat(prompt, max_tokens=400)
            return self._parse_skill_response(response, task_type, episodes)
        except Exception as e:
            log.error("skill_compiler.extract_failed",
                      task_type=task_type, error=str(e))
            return None

    def _parse_skill_response(
        self,
        response: str,
        task_type: str,
        episodes: list[dict],
    ) -> ExtractedSkill | None:
        """Parse structured LLM skill response."""
        lines = {
            line.split(":")[0].strip().upper(): ":".join(line.split(":")[1:]).strip()
            for line in response.strip().split("\n")
            if ":" in line
        }

        name = lines.get("NAME", f"skill_{task_type}")
        description = lines.get("DESCRIPTION", f"Reusable skill for {task_type}")
        steps_raw = lines.get("STEPS", "Execute task")
        triggers_raw = lines.get("TRIGGERS", task_type)
        outcome = lines.get("OUTCOME", "Task completed successfully")
        conf_raw = lines.get("CONFIDENCE", "0.7")

        try:
            confidence = float(conf_raw.strip())
        except ValueError:
            confidence = 0.7

        return ExtractedSkill(
            name=name[:64],
            description=description[:256],
            steps=[s.strip() for s in steps_raw.split("|") if s.strip()],
            trigger_conditions=[t.strip() for t in triggers_raw.split("|") if t.strip()],
            expected_outcome=outcome[:256],
            confidence=confidence,
            source_episode_ids=[ep["id"] for ep in episodes],
        )
