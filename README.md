# 🤖 Project JARVIS

**Autonomous AI Operating System** — Local-First, Multi-Agent, Self-Improving

> Not a chatbot. Not a wrapper. A persistent AI entity.

---

## Architecture

```
Executive Brain → Planner → Critic → Agent Council → Brain Services → Memory
     ↑                                                                    ↓
   Voice ←←←←←←←←← Computer Control ←←←←←←← Learning ←←←←←← Neo4j/Qdrant
```

## Current Phase

**Phase 0 — Core Infrastructure** ✅
- Event bus, audit log, security gateway, base agent, LLM backend

**Next:** Phase 1 (Memory System), Phase 2 (Executive Reasoning)

## Quick Start

```bash
# 1. Start Ollama (required)
ollama serve

# 2. Configure environment
cp .env .env.local  # edit as needed

# 3. Run JARVIS
python main.py
```

## Register local Qwen3.5-9B model

```bash
ollama create qwen3.5-9b -f config/Modelfile.qwen35
# Then update .env: JARVIS_LLM_MODEL=qwen3.5-9b
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Hardware

- CPU: AMD Ryzen 7 5700U (8 cores)
- RAM: 16 GB
- GPU: Integrated (CPU inference via Ollama)
- Primary LLM: gemma4:latest / Qwen3.5-9B-Q6_K

## Roadmap

| Phase | Status |
|-------|--------|
| 0 — Core Infrastructure | ✅ Complete |
| 1 — Core OS (event bus, agents) | ✅ Complete |
| 2 — Memory System | 🔄 Next |
| 3 — Executive Reasoning | ⏳ Planned |
| 4 — Agent Council | ⏳ Planned |
| 5 — Voice System | ⏳ Planned |
| 6 — Computer Control | ⏳ Planned |
| 7 — Self-Improvement | ⏳ Planned |
| 8 — Interfaces | ⏳ Planned |
| 9 — MYTHOS Model | ⏳ Deferred (needs GPU) |
