# 🤖 Project ZEON

**Autonomous AI Operating System** — Local-First, Multi-Agent, Self-Improving

> Not a chatbot. Not a wrapper. A persistent AI entity that learns, debates, and improves itself.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │                      ZEON OS v0.1.0                         │
  ├──────────────┬──────────────┬──────────────────────────────┤
  │  Voice Shell │  Rich CLI    │  FastAPI Web Dashboard        │
  ├──────────────┴──────────────┴──────────────────────────────┤
  │            LangGraph Executive Reasoning Loop                │
  │    Observe→Understand→Research→Plan→Critique→Execute        │
  ├────────────┬────────────┬───────────┬──────────────────────┤
  │ Executive  │  Planner   │  Critic   │ Research │ Coder      │
  │   Agent    │   Agent    │   Agent   │  Agent   │ Agent      │
  ├────────────┴────────────┴───────────┴──────────┴────────────┤
  │              Agent Council (Debate → Consensus)              │
  ├─────────────────────────────────────────────────────────────┤
  │                   5-Tier Memory System                       │
  │  Working │ Episodic │ Semantic(Vector) │ Skills │ Graph      │
  ├─────────────────────────────────────────────────────────────┤
  │         Computer Control: Filesystem + Shell + Browser       │
  ├─────────────────────────────────────────────────────────────┤
  │     Self-Improvement: SkillCompiler + ZeonGit branches       │
  └─────────────────────────────────────────────────────────────┘
```

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 — Infrastructure | Config, event bus, audit log, security gateway, LLM | ✅ Complete |
| 1 — Core OS | 9 agents, base reasoning loop, registry | ✅ Complete |
| 2 — Memory System | 5-tier memory + MemoryBrain unified API | ✅ Complete |
| 3 — Executive Reasoning | LangGraph 10-node graph, router, search tools | ✅ Complete |
| 4 — Voice System | openWakeWord → VAD → faster-whisper → Piper TTS (EN/HI) | ✅ Complete |
| 5 — Computer Control | Sandboxed FS, Shell executor, ComputerBrain facade | ✅ Complete |
| 6 — Agent Council | Multi-round debate, critique, consensus, quick-vote | ✅ Complete |
| 7 — Self-Improvement | SkillCompiler, ZeonGit approval-gated branches | ✅ Complete |
| 8 — Interfaces | Rich CLI + FastAPI Web Dashboard (SPA, WebSocket) | ✅ Complete |
| 9 — MYTHOS Model | Custom LLM fine-tune (needs GPU) | ⏳ Deferred |

**Tests:** 149/149 passing · **LLM:** gemma4:latest (Ollama) · **Python:** 3.13

---

## Quick Start

```bash
# 1. Start Ollama
ollama serve

# 2. Interactive CLI
python main.py

# 3. Web dashboard (localhost:8000)
python main.py --web

# 4. Voice mode
python main.py --voice
```

### CLI Commands

```
zeon> ask What is machine learning?
zeon> quick Summarise quantum computing in 3 sentences
zeon> memory stats
zeon> memory search <query>
zeon> system status
zeon> council What should ZEON prioritise next?
zeon> improve
zeon> skills
zeon> graph test
zeon> history
```

### Web Dashboard

`http://localhost:8000` — tabs: Chat · Council · Memory · Skills  
Real-time WebSocket event stream at `/ws`

---

## Voice Mode

```bash
pip install openwakeword webrtcvad faster-whisper sounddevice

# .env:
ZEON_VOICE_ENABLED=true

python main.py --voice
# Say: "Hey ZEON, ..."
```

---

## Run Tests

```bash
python -m pytest tests/ -v
# Expected: 149 passed
```

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 5700U (8 cores @ 4.3 GHz) |
| RAM | 16 GB DDR4 |
| GPU | Integrated Radeon (CPU inference via Ollama) |
| LLM | gemma4:latest (9.6 GB) / Qwen3.5-9B-Q6_K |
| Embeddings | all-MiniLM-L6-v2 (384-dim, CPU) |

---

## Project Structure

```
Zeon/
├── agents/              ← 9 autonomous agents
├── brains/
│   ├── memory/          ← 5-tier memory system
│   ├── council/         ← debate + consensus engine
│   ├── computer/        ← ComputerBrain facade
│   └── voice/           ← wake word, VAD, STT, TTS
├── core/                ← config, event bus, audit, LLM, security
├── improvements/
│   ├── skill_compiler.py   ← mine episodes → skills
│   └── zeon_git.py         ← approval-gated improvement branches
├── interfaces/
│   ├── cli.py              ← rich terminal (ask/memory/council/improve)
│   ├── web.py              ← FastAPI dashboard + WebSocket
│   └── voice_shell.py      ← voice state machine
├── orchestration/       ← LangGraph executive loop
├── tools/
│   ├── filesystem.py    ← sandboxed FS operations
│   ├── shell.py         ← risk-classified shell executor
│   └── search.py        ← DuckDuckGo + Crawl4AI
├── tests/unit/          ← 149 tests (Phase 0–8)
└── main.py              ← entry point (--voice / --web / --api)
```

---

## GitHub

[github.com/soumyashiv/zeonAI](https://github.com/soumyashiv/zeonAI)
