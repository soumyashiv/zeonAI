# 🤖 Project JARVIS

**Autonomous AI Operating System** — Local-First, Multi-Agent, Self-Improving

> Not a chatbot. Not a wrapper. A persistent AI entity.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                    JARVIS OS v0.1.0                     │
  ├──────────────┬──────────────┬──────────────────────────-┤
  │  Voice Shell │   CLI / Web  │     Event Bus             │
  ├──────────────┴──────────────┴───────────────────────────┤
  │            LangGraph Executive Reasoning Loop            │
  │  Observe→Understand→Research→Plan→Critique→Execute      │
  ├──────────┬──────────┬───────────┬────────────────────---┤
  │Executive │ Planner  │  Critic   │ Research │ Coder      │
  │  Agent   │  Agent   │  Agent    │  Agent   │ Agent      │
  ├──────────┴──────────┴───────────┴──────────┴────────────┤
  │                 5-Tier Memory System                     │
  │  Working│Episodic│Semantic(Qdrant)│Skills│Graph(Neo4j)  │
  └─────────────────────────────────────────────────────────┘
```

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Infrastructure | Config, event bus, audit log, security, LLM | ✅ Complete |
| 1 — Core OS | 9 agents, base reasoning loop, registry | ✅ Complete |
| 2 — Memory System | 5-tier memory + MemoryBrain unified API | ✅ Complete |
| 3 — Executive Reasoning | LangGraph 10-node graph, router, search | ✅ Complete |
| **4 — Voice System** | **Wake word → VAD → STT → TTS (EN/HI)** | **🔄 In Progress** |
| 5 — Agent Council | Multi-agent debates, consensus | ⏳ Planned |
| 6 — Computer Control | pyautogui, browser, file system | ⏳ Planned |
| 7 — Self-Improvement | Skill compilation, MYTHOS prep | ⏳ Planned |
| 8 — UI Interfaces | Web dashboard, voice HUD | ⏳ Planned |
| 9 — MYTHOS Model | Custom LLM (needs GPU) | ⏳ Deferred |

**Tests:** 49/49 passing · **LLM:** gemma4:latest (Ollama) · **Python:** 3.13

---

## Quick Start

```bash
# 1. Start Ollama
ollama serve

# 2. Run JARVIS CLI
python main.py

# 3. Commands
jarvis> ask What is machine learning?
jarvis> memory stats
jarvis> system status
jarvis> graph test
```

## Voice Mode (Phase 4 — in progress)

```bash
# Install voice dependencies
pip install openwakeword webrtcvad faster-whisper

# Start with voice enabled
python main.py --voice

# Say: "Hey JARVIS, what is the weather today?"
```

## Register local Qwen3.5-9B model

```bash
ollama create qwen3.5-9b -f config/Modelfile.qwen35
# Then: JARVIS_LLM_MODEL=qwen3.5-9b in .env
```

## Run Tests

```bash
python -m pytest tests/ -v
# Expected: 49 passed
```

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 5700U (8 cores @ 4.3 GHz) |
| RAM | 16 GB DDR4 |
| GPU | Integrated Radeon (CPU inference via Ollama) |
| Storage | 512 GB NVMe |
| LLM | gemma4:latest (9.6 GB) / Qwen3.5-9B-Q6_K |
| Embeddings | all-MiniLM-L6-v2 (384-dim, CPU) |

---

## Project Structure

```
Zeon/
├── agents/          ← 9 autonomous agents
│   ├── executive_agent.py
│   ├── planner_agent.py
│   ├── critic_agent.py
│   ├── executor_agent.py
│   ├── coder_agent.py
│   ├── research_agent.py
│   ├── memory_agent.py
│   ├── observer_agent.py
│   └── self_improvement_agent.py
├── brains/
│   └── memory/      ← 5-tier memory system
│       ├── working.py         # TTL in-process
│       ├── episodic.py        # SQLite experience log
│       ├── semantic.py        # Qdrant vector store
│       ├── procedural.py      # Skills registry
│       ├── knowledge_graph.py # Neo4j / NetworkX
│       └── __init__.py        # MemoryBrain unified API
├── core/            ← infrastructure
│   ├── config.py    event_bus.py   audit_log.py
│   ├── llm.py       security_gateway.py   registry.py
├── orchestration/   ← LangGraph
│   ├── graph.py     state.py     router.py
├── tools/
│   └── search.py    ← DuckDuckGo + Crawl4AI
├── interfaces/      ← CLI, Voice (Phase 4), Web (Phase 8)
├── tests/unit/      ← 49 tests
└── main.py          ← entry point
```

---

## GitHub

[github.com/soumyashiv/zeonAI](https://github.com/soumyashiv/zeonAI)
