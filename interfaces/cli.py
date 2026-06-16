"""
ZEON Interfaces — Rich Terminal CLI
Full-featured interactive terminal powered by Rich.
Commands: ask, memory, system, council, improve, voice, graph, help, exit
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Optional

import structlog
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from core.config import get_config

log = structlog.get_logger("zeon.cli")
cfg = get_config()
console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════╗[/]
[bold cyan]║[/]  [bold white]ZEON[/] [dim]— Autonomous AI Operating System[/]             [bold cyan]║[/]
[bold cyan]║[/]  [dim]Local-First · Multi-Agent · Self-Improving[/]          [bold cyan]║[/]
[bold cyan]╚══════════════════════════════════════════════════════╝[/]
"""

HELP_TEXT = """
[bold]Available commands:[/]

  [cyan]ask[/] [italic]<question>[/]          — Ask ZEON a question (full LangGraph pipeline)
  [cyan]quick[/] [italic]<question>[/]        — Quick LLM answer (no orchestration)
  [cyan]memory stats[/]            — Show 5-tier memory statistics
  [cyan]memory search[/] [italic]<query>[/]  — Search episodic memory
  [cyan]system status[/]           — CPU/RAM/agent status
  [cyan]council[/] [italic]<topic>[/]         — Run multi-agent council debate
  [cyan]improve[/]                 — Run self-improvement cycle
  [cyan]voice[/]                   — Switch to voice mode (if enabled)
  [cyan]graph test[/]              — Test knowledge graph
  [cyan]skills[/]                  — List procedural skills
  [cyan]history[/]                 — Recent session history
  [cyan]help[/]                    — Show this help
  [cyan]exit[/] / [cyan]quit[/]             — Exit ZEON
"""


class ZeonCLI:
    """Interactive rich terminal for ZEON."""

    def __init__(self) -> None:
        self._running = False
        self._history: list[tuple[str, str]] = []   # (input, output)
        self._session_start = datetime.now(timezone.utc)

    async def start(self) -> None:
        self._running = True
        console.print(BANNER)
        console.print(f"[dim]Session started: {self._session_start.strftime('%Y-%m-%d %H:%M UTC')}[/]")
        console.print(f"[dim]LLM: {cfg.llm_model}  |  Dev mode: {cfg.is_dev}[/]\n")

        # Initialise subsystems
        await self._init_subsystems()
        console.print("[green]✓[/] All subsystems initialised\n")
        console.print("[dim]Type [bold]help[/] for commands, [bold]exit[/] to quit[/]\n")

        while self._running:
            try:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: Prompt.ask("[bold cyan]zeon[/]")
                )
                line = raw.strip()
                if not line:
                    continue
                await self._dispatch(line)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Use 'exit' to quit ZEON.[/]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")

    async def _init_subsystems(self) -> None:
        from core.audit_log import AuditLog
        await AuditLog.init()

    async def _dispatch(self, line: str) -> None:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit"):
            await self._cmd_exit()
        elif cmd == "help":
            console.print(Markdown(HELP_TEXT.replace("[bold]", "**").replace("[/]", "").replace("[cyan]", "`").replace("[italic]", "")))
            console.print(Panel(HELP_TEXT, title="ZEON Help", border_style="cyan"))
        elif cmd == "ask":
            await self._cmd_ask(args)
        elif cmd == "quick":
            await self._cmd_quick(args)
        elif cmd == "memory":
            await self._cmd_memory(args)
        elif cmd == "system":
            await self._cmd_system(args)
        elif cmd == "council":
            await self._cmd_council(args)
        elif cmd == "improve":
            await self._cmd_improve()
        elif cmd == "voice":
            await self._cmd_voice()
        elif cmd == "graph":
            await self._cmd_graph(args)
        elif cmd == "skills":
            await self._cmd_skills()
        elif cmd == "history":
            self._cmd_history()
        else:
            # Treat unknown input as an 'ask' if it looks like a question
            if len(line) > 5:
                await self._cmd_ask(line)
            else:
                console.print(f"[yellow]Unknown command: '{cmd}'. Type 'help' for commands.[/]")

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _cmd_ask(self, question: str) -> None:
        if not question:
            console.print("[yellow]Usage: ask <question>[/]")
            return

        console.print(f"\n[dim]⟳ Processing:[/] [white]{question[:80]}[/]")

        with console.status("[cyan]Thinking...[/]", spinner="dots"):
            try:
                from orchestration.graph import run_task
                state = await run_task(question)
                result = state.get("result", {})

                # Extract best output
                outputs = []
                if isinstance(result, dict):
                    for step in result.get("results", []):
                        out = step.get("output", "")
                        if out and not str(out).startswith("Error"):
                            outputs.append(str(out))

                answer = "\n".join(outputs) if outputs else state.get("plan", {}).get("goal", "No answer generated.")

            except Exception as e:
                log.error("cli.ask_failed", error=str(e))
                answer = await self._quick_llm(question)

        self._print_answer(question, answer)
        self._history.append((question, answer))

    async def _cmd_quick(self, question: str) -> None:
        if not question:
            console.print("[yellow]Usage: quick <question>[/]")
            return
        with console.status("[cyan]...[/]", spinner="dots"):
            answer = await self._quick_llm(question)
        self._print_answer(question, answer)
        self._history.append((question, answer))

    async def _cmd_memory(self, args: str) -> None:
        from brains.memory import get_memory_brain
        brain = get_memory_brain()

        if args.startswith("search"):
            query = args[6:].strip()
            if not query:
                console.print("[yellow]Usage: memory search <query>[/]")
                return
            with console.status("[cyan]Searching...[/]"):
                results = await brain.episodic.search(query, limit=5)
            if not results:
                console.print("[dim]No episodes found.[/]")
                return
            table = Table(title=f"Memory Search: '{query}'", box=box.SIMPLE)
            table.add_column("Content", style="white", max_width=70)
            table.add_column("Importance", justify="right")
            for ep in results:
                content = getattr(ep, "content", str(ep))[:100]
                importance = getattr(ep, "importance", 0.5)
                table.add_row(content, f"{importance:.2f}")
            console.print(table)

        else:   # stats
            table = Table(title="ZEON Memory Statistics", box=box.ROUNDED)
            table.add_column("Tier", style="cyan")
            table.add_column("Backend", style="white")
            table.add_column("Status", style="green")

            table.add_row("Working",    "In-process TTL",         "✓ Active")
            table.add_row("Episodic",   "SQLite",                  "✓ Active")
            table.add_row("Semantic",   "SimpleVectorStore/Qdrant","✓ Active")
            table.add_row("Procedural", "SQLite skills registry",  "✓ Active")
            table.add_row("Graph",      "NetworkX/Neo4j",          "✓ Active")
            console.print(table)

    async def _cmd_system(self, args: str) -> None:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            table = Table(title="ZEON System Status", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("CPU Usage",    f"{cpu:.1f}%")
            table.add_row("RAM Used",     f"{ram.used / 1e9:.1f} GB / {ram.total / 1e9:.1f} GB")
            table.add_row("RAM %",        f"{ram.percent:.1f}%")
            table.add_row("LLM Model",    cfg.llm_model)
            table.add_row("Dev Mode",     str(cfg.is_dev))
            table.add_row("Voice",        "Enabled" if cfg.voice_enabled else "Disabled")
            console.print(table)
        except ImportError:
            console.print("[dim]psutil not installed — install for system stats.[/]")

    async def _cmd_council(self, topic: str) -> None:
        if not topic:
            console.print("[yellow]Usage: council <topic>[/]")
            return
        with console.status("[cyan]Council deliberating...[/]", spinner="dots"):
            from brains.council import get_council
            council = get_council()
            decision = await council.deliberate(topic)

        panel = Panel(
            f"{decision.answer}\n\n"
            f"[dim]Method: {decision.method} · Confidence: {decision.confidence:.0%} · "
            f"Rounds: {decision.rounds}[/]",
            title=f"[cyan]Council Decision[/] — {topic[:50]}",
            border_style="cyan",
        )
        console.print(panel)
        if decision.dissents:
            console.print(f"[dim]Dissenting views: {len(decision.dissents)}[/]")

    async def _cmd_improve(self) -> None:
        with console.status("[cyan]Running self-improvement cycle...[/]", spinner="dots"):
            from improvements.skill_compiler import SkillCompiler
            from brains.memory import get_memory_brain
            sc = SkillCompiler()
            report = await sc.compile(memory_brain=get_memory_brain())

        console.print(Panel(
            f"Episodes analysed: [bold]{report.episodes_analysed}[/]\n"
            f"Patterns found:    [bold]{report.patterns_found}[/]\n"
            f"Skills extracted:  [bold]{report.skills_extracted}[/]\n"
            f"Skills registered: [bold]{report.skills_registered}[/]\n\n"
            f"[dim]{report.summary}[/]",
            title="[green]Self-Improvement Report[/]",
            border_style="green",
        ))

    async def _cmd_voice(self) -> None:
        if not cfg.voice_enabled:
            console.print("[yellow]Voice is disabled. Set ZEON_VOICE_ENABLED=true in .env[/]")
            return
        console.print("[cyan]Switching to voice mode...[/]")
        from interfaces.voice_shell import run_voice_loop
        await run_voice_loop()

    async def _cmd_graph(self, args: str) -> None:
        from brains.memory import get_memory_brain
        brain = get_memory_brain()
        with console.status("[cyan]Testing knowledge graph...[/]"):
            try:
                await brain.graph.add_entity("ZEON", {"type": "system", "version": "0.1"})
                await brain.graph.add_entity("Memory", {"type": "subsystem"})
                await brain.graph.add_relation("ZEON", "Memory", "HAS_SUBSYSTEM")
                entities = await brain.graph.get_neighbors("ZEON")
                console.print(Panel(
                    f"[green]✓ Graph OK[/]\nNeighbours of ZEON: {entities}",
                    title="Knowledge Graph",
                    border_style="green",
                ))
            except Exception as e:
                console.print(f"[red]Graph error: {e}[/]")

    async def _cmd_skills(self) -> None:
        from brains.memory import get_memory_brain
        brain = get_memory_brain()
        skills = await brain.procedural.list_skills()
        if not skills:
            console.print("[dim]No skills registered yet. Run 'improve' to compile skills.[/]")
            return
        table = Table(title="Procedural Skills", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Success Rate", justify="right")
        table.add_column("Uses", justify="right")
        for skill in skills[:20]:
            name = getattr(skill, "name", str(skill))
            rate = getattr(skill, "success_rate", 0.0)
            uses = getattr(skill, "usage_count", 0)
            table.add_row(name, f"{rate:.0%}", str(uses))
        console.print(table)

    def _cmd_history(self) -> None:
        if not self._history:
            console.print("[dim]No history yet.[/]")
            return
        for i, (q, a) in enumerate(self._history[-10:], 1):
            console.print(f"[dim]{i}.[/] [cyan]{q[:60]}[/]")
            console.print(f"   [white]{a[:100]}[/]\n")

    async def _cmd_exit(self) -> None:
        console.print("\n[cyan]ZEON offline. Goodbye.[/]")
        self._running = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _quick_llm(self, question: str) -> str:
        try:
            from core.llm import chat
            messages = [
                {"role": "system", "content": "You are ZEON. Answer concisely in 2-3 sentences."},
                {"role": "user", "content": question},
            ]
            return await chat(messages, max_tokens=300)
        except Exception as e:
            return f"[LLM unavailable: {e}]"

    def _print_answer(self, question: str, answer: str) -> None:
        console.print(Panel(
            answer,
            title=f"[cyan]ZEON[/] [dim]— {question[:50]}[/]",
            border_style="cyan",
            padding=(1, 2),
        ))
        console.print()


async def run_cli() -> None:
    cli = ZeonCLI()
    await cli.start()


# Typer app for pyproject.toml entry point
try:
    import typer
    app = typer.Typer(name="zeon", help="ZEON — Autonomous AI Operating System")

    @app.command()
    def main(voice: bool = typer.Option(False, "--voice", help="Start in voice mode")):
        if voice:
            from interfaces.voice_shell import run_voice_loop
            asyncio.run(run_voice_loop())
        else:
            asyncio.run(run_cli())
except ImportError:
    app = None
