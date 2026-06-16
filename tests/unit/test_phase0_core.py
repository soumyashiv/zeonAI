"""
Phase 0 — Core Infrastructure Tests
"""
import asyncio
import pytest

from core.config import get_config
from core.event_bus import get_bus, AgentMessage, MessageType, EventBus
from core.audit_log import AuditLog, ActionType
from core.security_gateway import get_gateway, RiskLevel


# ── Config ────────────────────────────────────────────────────────────────────

def test_config_loads():
    cfg = get_config()
    assert cfg.zeon_env == "development"
    assert cfg.dev_auto_approve is True
    assert cfg.llm_model  # model name is set (value may vary by .env)


def test_config_voice_languages():
    cfg = get_config()
    langs = cfg.voice_language_list
    assert "en" in langs
    assert "hi" in langs


def test_config_sqlite_path_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "sub" / "zeon.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    cfg = cfg_mod.get_config()
    p = cfg.sqlite_path_resolved
    assert p.parent.exists()
    cfg_mod.get_config.cache_clear()


# ── Event Bus ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    bus = EventBus()
    await bus.start()
    received = []

    async def handler(msg: AgentMessage):
        received.append(msg)

    bus.subscribe("agent_a", handler)
    msg = AgentMessage(
        from_agent="agent_b",
        to_agent="agent_a",
        message_type=MessageType.HEARTBEAT,
        payload={"test": True},
    )
    await bus.publish(msg)
    await asyncio.sleep(0.15)
    await bus.stop()

    assert len(received) == 1
    assert received[0].message_type == MessageType.HEARTBEAT


@pytest.mark.asyncio
async def test_event_bus_priority():
    bus = EventBus()
    await bus.start()
    order = []

    async def handler(msg: AgentMessage):
        order.append(msg.priority)

    bus.subscribe("prio_agent", handler)
    for prio in [3, 9, 1, 7]:
        await bus.publish(
            AgentMessage(
                from_agent="test",
                to_agent="prio_agent",
                message_type=MessageType.SYSTEM_EVENT,
                priority=prio,
            )
        )
    await asyncio.sleep(0.2)
    await bus.stop()
    # Higher priority messages should be delivered first
    assert order[0] == 9
    assert order[1] == 7


@pytest.mark.asyncio
async def test_event_bus_broadcast():
    bus = EventBus()
    await bus.start()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("broadcast", handler)
    await bus.publish(
        AgentMessage(
            from_agent="system",
            to_agent="anyone",
            message_type=MessageType.SYSTEM_EVENT,
        )
    )
    await asyncio.sleep(0.15)
    await bus.stop()
    assert len(received) == 1


# ── Audit Log ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audit_log_record_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    # Reload audit_log with new config
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)

    await al_mod.AuditLog.init()
    eid = await al_mod.AuditLog.record(
        agent="test_agent",
        action_type=al_mod.ActionType.SYSTEM,
        action_detail="unit test action",
    )
    assert eid
    await al_mod.AuditLog.complete(eid, result="done", success=True)
    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_audit_log_save_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ep.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()
    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)

    await al_mod.AuditLog.init()
    ep_id = await al_mod.AuditLog.save_episode(
        agent="research_agent",
        task_description="search web",
        action_taken="called duckduckgo",
        result="found 10 results",
        success=True,
        importance_score=0.8,
    )
    eps = await al_mod.AuditLog.recent_episodes(limit=5)
    assert len(eps) >= 1
    assert eps[0]["agent"] == "research_agent"
    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── Security Gateway ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_security_auto_approve_low_risk():
    gw = get_gateway()
    result = await gw.request_approval(
        action_type="file_read",
        agent="test_agent",
        detail="reading config.yaml",
    )
    assert result.approved is True
    assert result.approved_by == "dev_auto_approve"


@pytest.mark.asyncio
async def test_security_classify_risk():
    gw = get_gateway()
    assert gw.classify_risk("file_read") == RiskLevel.LOW
    assert gw.classify_risk("shell_exec") == RiskLevel.HIGH
    assert gw.classify_risk("self_improve") == RiskLevel.CRITICAL


@pytest.mark.asyncio
async def test_security_auto_approve_high_risk_in_dev():
    gw = get_gateway()
    result = await gw.request_approval(
        action_type="shell_exec",
        agent="coder_agent",
        detail="run pytest",
    )
    # In dev mode, HIGH risk is auto-approved
    assert result.approved is True


# ── Agent Message Protocol ────────────────────────────────────────────────────

def test_agent_message_serialization():
    msg = AgentMessage(
        from_agent="executive",
        to_agent="planner",
        message_type=MessageType.TASK_CREATED,
        payload={"task": "plan a research session"},
        priority=8,
    )
    json_str = msg.to_json()
    restored = AgentMessage.from_json(json_str)
    assert restored.from_agent == "executive"
    assert restored.message_type == MessageType.TASK_CREATED
    assert restored.priority == 8
    assert restored.id == msg.id


def test_agent_message_reply():
    original = AgentMessage(
        from_agent="executive",
        to_agent="planner",
        message_type=MessageType.TASK_CREATED,
        payload={"task": "do something"},
    )
    reply = original.reply(
        MessageType.TASK_ACCEPTED, {"status": "ok"}
    )
    assert reply.from_agent == "planner"
    assert reply.to_agent == "executive"
    assert reply.parent_task_id == original.id
