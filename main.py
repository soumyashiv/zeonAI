"""
JARVIS — Autonomous AI Operating System
Main entry point. Initializes all subsystems and starts the runtime.
"""
from __future__ import annotations

import asyncio
import signal
import sys

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)

log = structlog.get_logger("jarvis.main")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║      J . A . R . V . I . S                              ║
║      Autonomous AI Operating System  v0.1.0             ║
║      Local-First | Multi-Agent | Self-Improving         ║
╚══════════════════════════════════════════════════════════╝
"""


async def startup() -> dict:
    """Initialize all JARVIS subsystems. Returns handles dict."""
    from core.config import get_config
    from core.event_bus import get_bus
    from core.audit_log import AuditLog

    cfg = get_config()
    log.info("jarvis.starting", env=cfg.jarvis_env, model=cfg.llm_model)

    # 1. Audit log (SQLite)
    await AuditLog.init()

    # 2. Event bus
    bus = get_bus()
    await bus.start()

    # 3. Memory brain (initializes all 5 memory types)
    from brains.memory import get_memory_brain
    mem = get_memory_brain()

    # 4. Register and start all agents
    from agents.executive_agent import ExecutiveAgent
    from agents.planner_agent import PlannerAgent
    from agents.critic_agent import CriticAgent
    from agents.memory_agent import MemoryAgent
    from agents.observer_agent import ObserverAgent
    from agents.executor_agent import ExecutorAgent

    agents = {
        "executive": ExecutiveAgent(),
        "planner": PlannerAgent(),
        "critic": CriticAgent(),
        "memory": MemoryAgent(),
        "observer": ObserverAgent(),
        "executor": ExecutorAgent(),
    }

    for name, agent in agents.items():
        try:
            await agent.start()
        except Exception as e:
            log.warning(f"jarvis.agent_start_failed", agent=name, error=str(e))

    # 5. LangGraph (optional — needs langgraph installed)
    graph = None
    try:
        from orchestration.graph import get_jarvis_graph
        graph = get_jarvis_graph()
        if graph:
            log.info("jarvis.graph_ready", nodes=10)
    except Exception as e:
        log.warning("jarvis.graph_unavailable", error=str(e))

    # 6. LLM backend check
    try:
        from core.llm import get_llm
        llm = await get_llm()
        healthy = await llm.health_check()
        log.info("jarvis.llm", model=cfg.llm_model, healthy=healthy)
    except Exception as e:
        log.warning("jarvis.llm_unavailable", error=str(e))
        llm = None

    log.info("jarvis.ready")
    return {"bus": bus, "agents": agents, "graph": graph, "mem": mem, "llm": llm}


async def shutdown(handles: dict) -> None:
    from core.audit_log import AuditLog
    log.info("jarvis.shutdown")
    for agent in handles.get("agents", {}).values():
        try:
            await agent.stop()
        except Exception:
            pass
    await handles["bus"].stop()
    await AuditLog.close()
    log.info("jarvis.stopped")


async def run_cli() -> None:
    print(BANNER)
    handles = await startup()
    graph = handles.get("graph")
    mem = handles.get("mem")

    print("  Commands: ask <query> | memory stats | system status | quit\n")

    loop = asyncio.get_event_loop()

    def _sigint(sig, frame):
        print("\n  Ctrl+C — shutting down...")
        asyncio.ensure_future(shutdown(handles))
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    while True:
        try:
            raw = await loop.run_in_executor(
                None, lambda: input("  jarvis> ").strip()
            )
        except EOFError:
            break

        if not raw:
            continue

        cmd = raw.lower()

        # ── Built-in commands ──────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            break

        elif cmd in ("help", "?"):
            print("""
  ask <your question>   — Run full reasoning loop
  memory stats          — Show memory statistics
  system status         — Show system info
  graph test            — Test LangGraph (requires langgraph)
  quit                  — Exit
""")

        elif cmd == "system status":
            from core.config import get_config
            cfg = get_config()
            agents = handles.get("agents", {})
            print(f"\n  LLM Model  : {cfg.llm_model}")
            print(f"  Agents     : {', '.join(agents.keys())}")
            print(f"  Graph      : {'ready' if graph else 'unavailable (pip install langgraph)'}")
            print(f"  Memory     : 5-tier (working/episodic/semantic/procedural/graph)\n")

        elif cmd == "memory stats":
            try:
                stats = await mem.stats()
                print(f"\n  Working keys      : {stats['working_memory_keys']}")
                print(f"  Episodes          : {stats['episodic']['total_episodes']}")
                print(f"  Skills            : {stats['procedural']['total_skills']}")
                print(f"  Graph nodes/edges : {stats['knowledge_graph']['nodes']}/{stats['knowledge_graph']['edges']}\n")
            except Exception as e:
                print(f"  Error: {e}")

        elif cmd == "graph test":
            if not graph:
                print("  ✗ LangGraph not available. Run: pip install langgraph")
            else:
                print("  Running graph test task...")
                try:
                    from orchestration.graph import run_task
                    result = await run_task("What is 2 + 2?")
                    print(f"  ✓ Verified: {result.get('verified')} | Score: {result.get('critique_score')}")
                    print(f"  Result: {str(result.get('result',''))[:200]}")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

        elif raw.lower().startswith("ask "):
            query = raw[4:].strip()
            if not query:
                print("  Usage: ask <your question>")
                continue

            if graph:
                print(f"  [JARVIS] Thinking...\n")
                try:
                    from orchestration.graph import run_task
                    result = await run_task(query)
                    verified = result.get("verified", False)
                    score = result.get("critique_score", "?")
                    steps = result.get("result", {})

                    print(f"  ┌─ JARVIS Response (Score: {score}/10, Verified: {verified})")
                    if isinstance(steps, dict) and steps.get("results"):
                        for r in steps["results"]:
                            print(f"  │  Step {r.get('step','?')}: {str(r.get('output',''))[:200]}")
                    else:
                        print(f"  │  {str(steps)[:300]}")
                    print(f"  └─ {result.get('verify_notes','')}\n")

                except Exception as e:
                    print(f"  ✗ Error: {e}\n")
            else:
                # Fallback: direct LLM call
                try:
                    from core.llm import chat
                    response = await chat([{"role": "user", "content": query}])
                    print(f"\n  [JARVIS] {response}\n")
                except Exception as e:
                    print(f"  ✗ LLM Error: {e}\n")
                    print("  Start Ollama with: ollama serve")

        else:
            # Treat anything else as an implicit ask
            if graph:
                print(f"  [JARVIS] Processing...\n")
                try:
                    from orchestration.graph import run_task
                    result = await run_task(raw)
                    steps = result.get("result", {})
                    print(f"  [JARVIS] {str(steps)[:400]}\n")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
            else:
                try:
                    from core.llm import chat
                    response = await chat([{"role": "user", "content": raw}])
                    print(f"\n  [JARVIS] {response}\n")
                except Exception as e:
                    print(f"  ✗ {e}")

    await shutdown(handles)


if __name__ == "__main__":
    if "--voice" in sys.argv:
        # Voice shell mode
        from interfaces.voice_shell import run_voice_loop
        asyncio.run(run_voice_loop())
    else:
        asyncio.run(run_cli())
