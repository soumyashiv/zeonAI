"""
ZEON Web Dashboard — FastAPI Backend
Provides REST endpoints and WebSocket streaming for the web UI.

Endpoints:
  GET  /                 → serve dashboard HTML
  GET  /api/status       → system health
  POST /api/ask          → ask ZEON (streaming via SSE)
  GET  /api/memory/stats → memory tier statistics
  GET  /api/memory/search?q=... → episodic search
  POST /api/council      → agent council debate
  GET  /api/skills       → list procedural skills
  GET  /api/history      → recent interactions
  WS   /ws               → real-time event stream
"""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from core.config import get_config

log = structlog.get_logger("zeon.web")
cfg = get_config()

# In-memory store for recent interactions
_interactions: list[dict] = []
_ws_clients: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.audit_log import AuditLog
    await AuditLog.init()
    log.info("zeon.web.started", host="localhost", port=8000)
    yield
    await AuditLog.close()


app = FastAPI(
    title="ZEON Dashboard",
    description="ZEON Autonomous AI OS — Web Interface",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    use_orchestration: bool = False


class CouncilRequest(BaseModel):
    topic: str
    proposers: list[str] | None = None
    critics: list[str] | None = None



# ── Dashboard HTML (inline single-page app) ──────────────────────────────────

DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZEON Dashboard</title>
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
    --accent: #00d4ff; --accent2: #7c3aed; --text: #e2e8f0;
    --dim: #64748b; --success: #10b981; --warn: #f59e0b; --err: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 16px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; background: var(--surface); }
  header h1 { font-size: 1.4rem; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .subtitle { color: var(--dim); font-size: .85rem; }

  .main { display: grid; grid-template-columns: 280px 1fr; flex: 1; overflow: hidden; }

  /* Sidebar */
  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 20px 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 24px; }
  .sidebar-section h3 { font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; color: var(--dim); margin-bottom: 12px; }
  .stat-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .stat-card .label { font-size: .75rem; color: var(--dim); }
  .stat-card .value { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
  .memory-tier { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: .85rem; }
  .memory-tier:last-child { border-bottom: none; }
  .badge { background: var(--accent2); color: white; font-size: .65rem; padding: 2px 6px; border-radius: 4px; }

  /* Chat area */
  .chat-area { display: flex; flex-direction: column; overflow: hidden; }
  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .message { max-width: 85%; animation: fadeIn .3s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  .message.user { align-self: flex-end; }
  .message.zeon { align-self: flex-start; }
  .message-bubble { padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: .92rem; }
  .message.user .message-bubble { background: var(--accent2); color: white; border-bottom-right-radius: 4px; }
  .message.zeon .message-bubble { background: var(--surface); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .message-meta { font-size: .7rem; color: var(--dim); margin-top: 4px; }
  .message.user .message-meta { text-align: right; }
  .thinking { display: flex; gap: 4px; padding: 12px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; width: fit-content; }
  .thinking span { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: bounce .9s infinite; }
  .thinking span:nth-child(2){animation-delay:.15s}.thinking span:nth-child(3){animation-delay:.3s}
  @keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}

  .input-area { padding: 16px 24px; border-top: 1px solid var(--border); background: var(--surface); display: flex; gap: 12px; align-items: flex-end; }
  textarea { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: var(--text); font-family: inherit; font-size: .92rem; resize: none; min-height: 44px; max-height: 120px; outline: none; transition: border-color .2s; }
  textarea:focus { border-color: var(--accent); }
  button { background: linear-gradient(135deg, var(--accent), var(--accent2)); border: none; color: white; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: .9rem; white-space: nowrap; transition: opacity .2s; }
  button:hover { opacity: .85; }
  button:disabled { opacity: .4; cursor: default; }
  .btn-row { display: flex; gap: 8px; }
  .btn-secondary { background: var(--border); font-size: .8rem; padding: 6px 12px; }

  /* Tabs */
  .tab-bar { display: flex; gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); }
  .tab { padding: 10px 20px; background: var(--surface); cursor: pointer; font-size: .85rem; color: var(--dim); transition: all .2s; }
  .tab.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
  .tab-content { display: none; flex: 1; overflow-y: auto; padding: 20px; }
  .tab-content.active { display: block; }
</style>
</head>
<body>
<header>
  <div class="status-dot" id="statusDot"></div>
  <h1>ZEON</h1>
  <span class="subtitle">Autonomous AI Operating System</span>
  <span style="margin-left:auto;font-size:.8rem;color:var(--dim)" id="modelLabel">Loading...</span>
</header>

<div class="main">
  <!-- Sidebar -->
  <div class="sidebar">
    <div class="sidebar-section">
      <h3>System</h3>
      <div class="stat-card">
        <div class="label">LLM Model</div>
        <div class="value" style="font-size:.9rem" id="sysModel">—</div>
      </div>
    </div>
    <div class="sidebar-section">
      <h3>Memory Tiers</h3>
      <div id="memoryTiers">
        <div class="memory-tier"><span>Working</span><span class="badge">TTL</span></div>
        <div class="memory-tier"><span>Episodic</span><span class="badge">SQLite</span></div>
        <div class="memory-tier"><span>Semantic</span><span class="badge">Vector</span></div>
        <div class="memory-tier"><span>Procedural</span><span class="badge">Skills</span></div>
        <div class="memory-tier"><span>Graph</span><span class="badge">NetworkX</span></div>
      </div>
    </div>
    <div class="sidebar-section">
      <h3>Quick Actions</h3>
      <div class="btn-row" style="flex-direction:column;gap:8px">
        <button class="btn-secondary" onclick="runCouncil()">🏛 Council Debate</button>
        <button class="btn-secondary" onclick="runImprove()">🔧 Self-Improve</button>
        <button class="btn-secondary" onclick="loadStatus()">📊 Refresh Status</button>
      </div>
    </div>
  </div>

  <!-- Chat + Tabs -->
  <div class="chat-area">
    <div class="tab-bar">
      <div class="tab active" onclick="showTab('chat',this)">💬 Chat</div>
      <div class="tab" onclick="showTab('council',this)">🏛 Council</div>
      <div class="tab" onclick="showTab('memory',this)">🧠 Memory</div>
      <div class="tab" onclick="showTab('skills',this)">⚡ Skills</div>
    </div>

    <!-- Chat Tab -->
    <div id="tab-chat" class="tab-content active" style="display:flex;flex-direction:column;padding:0">
      <div class="messages" id="messages"></div>
      <div class="input-area">
        <textarea id="input" placeholder="Ask ZEON anything..." rows="1"
          onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
        <button onclick="sendMessage()" id="sendBtn">Send</button>
      </div>
    </div>

    <!-- Council Tab -->
    <div id="tab-council" class="tab-content">
      <h3 style="margin-bottom:16px;color:var(--accent)">Agent Council Debate</h3>
      <div style="display:flex;gap:12px;margin-bottom:16px">
        <input id="councilTopic" placeholder="Enter debate topic..." style="flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px;color:var(--text);font-size:.9rem;outline:none">
        <button onclick="runCouncil()">Deliberate</button>
      </div>
      <div id="councilResult" style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;min-height:120px;line-height:1.7"></div>
    </div>

    <!-- Memory Tab -->
    <div id="tab-memory" class="tab-content">
      <h3 style="margin-bottom:16px;color:var(--accent)">Memory Search</h3>
      <div style="display:flex;gap:12px;margin-bottom:16px">
        <input id="memoryQuery" placeholder="Search episodic memory..." style="flex:1;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px;color:var(--text);font-size:.9rem;outline:none">
        <button onclick="searchMemory()">Search</button>
      </div>
      <div id="memoryResults"></div>
    </div>

    <!-- Skills Tab -->
    <div id="tab-skills" class="tab-content">
      <h3 style="margin-bottom:16px;color:var(--accent)">Procedural Skills</h3>
      <div id="skillsList"><div style="color:var(--dim)">Loading skills...</div></div>
    </div>
  </div>
</div>

<script>
const api = loc => fetch(loc).then(r=>r.json());

async function loadStatus() {
  try {
    const s = await api('/api/status');
    document.getElementById('sysModel').textContent = s.model || '—';
    document.getElementById('modelLabel').textContent = s.model || 'ZEON';
    document.getElementById('statusDot').style.background = s.ok ? 'var(--success)' : 'var(--err)';
  } catch(e) { document.getElementById('statusDot').style.background = 'var(--err)'; }
}

async function loadSkills() {
  try {
    const s = await api('/api/skills');
    const el = document.getElementById('skillsList');
    if (!s.skills || !s.skills.length) { el.innerHTML = '<div style="color:var(--dim)">No skills yet. Run self-improvement to compile.</div>'; return; }
    el.innerHTML = s.skills.map(sk => `
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:8px">
        <div style="font-weight:600;color:var(--accent)">${sk.name}</div>
        <div style="font-size:.85rem;color:var(--dim);margin-top:4px">${sk.description||''}</div>
      </div>`).join('');
  } catch(e) {}
}

function showTab(name, el) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>{t.classList.remove('active');t.style.display='none';});
  el.classList.add('active');
  const tc = document.getElementById('tab-'+name);
  tc.classList.add('active');
  tc.style.display = name==='chat' ? 'flex' : 'block';
  if(name==='skills') loadSkills();
}

function addMessage(role, text) {
  const msgs = document.getElementById('messages');
  const time = new Date().toLocaleTimeString();
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `<div class="message-bubble">${text.replace(/\n/g,'<br>')}</div><div class="message-meta">${time}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function showThinking() {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.id = 'thinking';
  div.className = 'message zeon';
  div.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeThinking() {
  const t = document.getElementById('thinking');
  if(t) t.remove();
}

async function sendMessage() {
  const inp = document.getElementById('input');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  document.getElementById('sendBtn').disabled = true;
  addMessage('user', q);
  showThinking();
  try {
    const r = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q})
    });
    const data = await r.json();
    removeThinking();
    addMessage('zeon', data.answer || data.error || 'No response.');
  } catch(e) {
    removeThinking();
    addMessage('zeon', `Error: ${e.message}`);
  }
  document.getElementById('sendBtn').disabled = false;
}

async function runCouncil() {
  const topic = document.getElementById('councilTopic')?.value || 'What should ZEON prioritise?';
  const el = document.getElementById('councilResult');
  if(el) el.innerHTML = '<div style="color:var(--dim)">Council deliberating...</div>';
  showTab('council', document.querySelectorAll('.tab')[1]);
  try {
    const r = await fetch('/api/council', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({topic})
    });
    const d = await r.json();
    el.innerHTML = `<div style="font-weight:600;color:var(--accent);margin-bottom:8px">${d.method?.toUpperCase()} (${Math.round((d.confidence||0)*100)}% confidence)</div><div>${d.answer||''}</div>`;
  } catch(e) { if(el) el.innerHTML = `Error: ${e.message}`; }
}

async function runImprove() {
  const r = await fetch('/api/improve', {method:'POST'}).then(r=>r.json()).catch(e=>({error:e.message}));
  alert(r.summary || r.error || 'Done');
}

async function searchMemory() {
  const q = document.getElementById('memoryQuery').value.trim();
  if(!q) return;
  const el = document.getElementById('memoryResults');
  el.innerHTML = '<div style="color:var(--dim)">Searching...</div>';
  try {
    const r = await fetch(`/api/memory/search?q=${encodeURIComponent(q)}`).then(r=>r.json());
    if(!r.results?.length) { el.innerHTML = '<div style="color:var(--dim)">No results.</div>'; return; }
    el.innerHTML = r.results.map(ep=>`
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:8px;font-size:.85rem">
        ${ep.content || ep}
      </div>`).join('');
  } catch(e) { el.innerHTML = `Error: ${e.message}`; }
}

// Init
loadStatus();
loadSkills();
setInterval(loadStatus, 30000);
</script>
</body>
</html>'''


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def status():
    return {
        "ok": True,
        "model": cfg.llm_model,
        "dev_mode": cfg.is_dev,
        "voice_enabled": cfg.voice_enabled,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "0.1.0",
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(400, "question is required")

    try:
        from core.llm import chat
        messages = [
            {"role": "system", "content": "You are ZEON. Answer concisely and helpfully."},
            {"role": "user",   "content": req.question},
        ]
        answer = await chat(messages, max_tokens=500)
    except Exception as e:
        log.error("web.ask_failed", error=str(e))
        answer = f"[ZEON LLM unavailable: {e}]"

    interaction = {
        "q": req.question,
        "a": answer,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    _interactions.append(interaction)
    if len(_interactions) > 100:
        _interactions.pop(0)

    await _broadcast({"type": "ask", "question": req.question, "answer": answer[:200]})
    return {"answer": answer, "question": req.question}


@app.get("/api/memory/stats")
async def memory_stats():
    return {
        "tiers": [
            {"name": "Working",    "backend": "In-process TTL",     "status": "active"},
            {"name": "Episodic",   "backend": "SQLite",              "status": "active"},
            {"name": "Semantic",   "backend": "SimpleVectorStore",   "status": "active"},
            {"name": "Procedural", "backend": "SQLite skills",       "status": "active"},
            {"name": "Graph",      "backend": "NetworkX",            "status": "active"},
        ]
    }


@app.get("/api/memory/search")
async def memory_search(q: str = ""):
    if not q:
        return {"results": []}
    try:
        from brains.memory import get_memory_brain
        brain = get_memory_brain()
        results = await brain.episodic.search(q, limit=10)
        return {
            "results": [
                {"content": getattr(ep, "content", str(ep))[:300]}
                for ep in results
            ]
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/council")
async def council(req: CouncilRequest):
    try:
        from brains.council import get_council
        c = get_council()
        decision = await c.deliberate(
            topic=req.topic,
            proposers=req.proposers,
            critics=req.critics,
        )
        return {
            "answer": decision.answer,
            "method": decision.method,
            "confidence": decision.confidence,
            "consensus_score": decision.consensus_score,
            "rounds": decision.rounds,
            "dissents": decision.dissents,
        }
    except Exception as e:
        log.error("web.council_failed", error=str(e))
        return {"answer": f"Council error: {e}", "method": "error", "confidence": 0.0}


@app.post("/api/improve")
async def improve():
    try:
        from improvements.skill_compiler import SkillCompiler
        from brains.memory import get_memory_brain
        sc = SkillCompiler()
        report = await sc.compile(memory_brain=get_memory_brain())
        return {
            "skills_extracted": report.skills_extracted,
            "skills_registered": report.skills_registered,
            "episodes_analysed": report.episodes_analysed,
            "summary": report.summary,
        }
    except Exception as e:
        return {"error": str(e), "summary": f"Improvement failed: {e}"}


@app.get("/api/skills")
async def skills():
    try:
        from brains.memory import get_memory_brain
        brain = get_memory_brain()
        skill_list = await brain.procedural.list_skills()
        return {
            "skills": [
                {
                    "name": getattr(s, "name", str(s)),
                    "description": getattr(s, "description", ""),
                    "usage_count": getattr(s, "usage_count", 0),
                    "success_rate": getattr(s, "success_rate", 0.0),
                }
                for s in skill_list[:50]
            ]
        }
    except Exception as e:
        return {"skills": [], "error": str(e)}


@app.get("/api/history")
async def history():
    return {"interactions": _interactions[-20:]}


# ── WebSocket event stream ────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    log.info("web.ws_connected", clients=len(_ws_clients))
    try:
        while True:
            await websocket.receive_text()   # keep alive
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)
        log.info("web.ws_disconnected", clients=len(_ws_clients))


async def _broadcast(message: dict) -> None:
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


async def run_web(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    log.info("zeon.web.launching", host=host, port=port)
    await server.serve()
