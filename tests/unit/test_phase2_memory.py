"""
Phase 2 — Memory System Tests
Tests all 5 memory types + unified MemoryBrain.
"""
import asyncio
import pytest
import tempfile
import os
from pathlib import Path


# ── Working Memory ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_working_memory_set_get():
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory(default_ttl=60)
    await wm.set("key1", "value1")
    val = await wm.get("key1")
    assert val == "value1"


@pytest.mark.asyncio
async def test_working_memory_ttl_expiry():
    import time
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory(default_ttl=60)
    await wm.set("expire_key", "soon", ttl=0)
    await asyncio.sleep(0.01)
    # Force expire by manipulating internal store
    wm._store["expire_key"] = ("soon", time.monotonic() - 1)
    val = await wm.get("expire_key")
    assert val is None


@pytest.mark.asyncio
async def test_working_memory_list():
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory()
    await wm.append_to_list("turns", {"role": "user", "content": "hi"})
    await wm.append_to_list("turns", {"role": "assistant", "content": "hello"})
    turns = await wm.get_list("turns")
    assert len(turns) == 2
    assert turns[0]["role"] == "user"


@pytest.mark.asyncio
async def test_working_memory_json():
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory()
    data = {"task": "plan", "steps": [1, 2, 3]}
    await wm.set_json("plan_key", data)
    loaded = await wm.get_json("plan_key")
    assert loaded == data


@pytest.mark.asyncio
async def test_working_memory_delete():
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory()
    await wm.set("del_key", "x")
    await wm.delete("del_key")
    assert await wm.get("del_key") is None


@pytest.mark.asyncio
async def test_working_memory_max_list_len():
    from brains.memory.working import WorkingMemory
    wm = WorkingMemory()
    for i in range(60):
        await wm.append_to_list("history", i, max_len=10)
    items = await wm.get_list("history")
    assert len(items) == 10
    assert items[-1] == 59


# ── Episodic Memory ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_episodic_memory_add_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ep_test.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from brains.memory import episodic as ep_mod
    importlib.reload(ep_mod)
    ep = ep_mod.EpisodicMemory()

    ep_id = await ep.add(
        agent="test_agent",
        task_description="search for python tutorials",
        action_taken="called duckduckgo",
        result="found 10 results",
        success=True,
        importance_score=0.8,
    )
    assert ep_id

    recent = await ep.recent(agent="test_agent", limit=5)
    assert len(recent) >= 1
    assert recent[0]["agent"] == "test_agent"

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_episodic_memory_search(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ep_search.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from brains.memory import episodic as ep_mod
    importlib.reload(ep_mod)
    ep = ep_mod.EpisodicMemory()

    await ep.add(
        agent="researcher",
        task_description="how to train a neural network",
        action_taken="searched",
        result="pytorch tutorial",
        success=True,
    )
    results = await ep.search_by_task("neural network")
    assert len(results) >= 1
    assert "neural" in results[0]["task_description"]

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── Semantic Memory ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_memory_store_and_search():
    from brains.memory.semantic import SemanticMemory, SimpleVectorStore
    # Use SimpleVectorStore directly to avoid Qdrant dependency
    sm = SemanticMemory.__new__(SemanticMemory)
    sm._store = SimpleVectorStore()
    sm._embedder = None

    # Mock embedder to return deterministic vectors
    import unittest.mock as mock
    sm._embed = mock.AsyncMock(side_effect=lambda text: [0.1] * 384 if "python" in text.lower() else [0.9] * 384)

    eid = await sm.store("Python is a programming language", metadata={"source": "test"})
    assert eid

    results = await sm.search("python programming")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_semantic_memory_simple_vector_store():
    from brains.memory.semantic import SimpleVectorStore, SemanticEntry
    store = SimpleVectorStore()

    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    vec_q = [1.0, 0.0, 0.0]  # identical to vec_a

    await store.upsert("id1", vec_a, {"text": "doc A"})
    await store.upsert("id2", vec_b, {"text": "doc B"})

    results = await store.search(vec_q, limit=2)
    assert results[0].id == "id1"   # most similar
    assert results[0].score > results[1].score

    await store.delete("id1")
    assert store.count == 1


# ── Procedural Memory ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_procedural_memory_register_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "proc.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from brains.memory import procedural as proc_mod
    importlib.reload(proc_mod)
    pm = proc_mod.ProceduralMemory()

    skill_id = await pm.register(
        name="web_search",
        description="Search the web using DuckDuckGo",
        trigger_pattern="search|find|look up",
        code="async def run(query): return await quick_search(query)",
    )
    assert skill_id

    skill = await pm.get("web_search")
    assert skill is not None
    assert skill["name"] == "web_search"

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_procedural_memory_usage_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "usage.db"))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from core import audit_log as al_mod
    importlib.reload(al_mod)
    await al_mod.AuditLog.init()

    from brains.memory import procedural as proc_mod
    importlib.reload(proc_mod)
    pm = proc_mod.ProceduralMemory()

    await pm.register(name="test_skill", description="a test", trigger_pattern="test")
    await pm.record_usage("test_skill", success=True)
    await pm.record_usage("test_skill", success=True)
    await pm.record_usage("test_skill", success=False)

    skill = await pm.get("test_skill")
    assert skill["usage_count"] == 3
    assert abs(skill["success_rate"] - 0.667) < 0.01

    await al_mod.AuditLog.close()
    cfg_mod.get_config.cache_clear()


# ── Knowledge Graph ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_graph_add_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEON_ROOT", str(tmp_path))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from brains.memory import knowledge_graph as kg_mod
    importlib.reload(kg_mod)

    kg = kg_mod.KnowledgeGraph()
    # Force NetworkX backend
    from brains.memory.knowledge_graph import NetworkXGraph
    kg._g = NetworkXGraph.__new__(NetworkXGraph)
    import networkx as nx
    from pathlib import Path
    kg._g._G = nx.DiGraph()
    kg._g._persist_path = tmp_path / "kg.json"

    cid = await kg.add_concept("Python", description="A programming language", confidence=0.9)
    assert cid

    node = await kg._g.get_node(cid)
    assert node is not None
    assert node.properties.get("name") == "Python"

    cfg_mod.get_config.cache_clear()


@pytest.mark.asyncio
async def test_knowledge_graph_relation(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEON_ROOT", str(tmp_path))
    from core import config as cfg_mod
    cfg_mod.get_config.cache_clear()

    import importlib
    from brains.memory import knowledge_graph as kg_mod
    importlib.reload(kg_mod)

    kg = kg_mod.KnowledgeGraph()
    from brains.memory.knowledge_graph import NetworkXGraph
    import networkx as nx
    kg._g = NetworkXGraph.__new__(NetworkXGraph)
    kg._g._G = nx.DiGraph()
    kg._g._persist_path = tmp_path / "kg2.json"

    aid = await kg.add_concept("AI", "Artificial Intelligence")
    mid = await kg.add_concept("ML", "Machine Learning")
    await kg.relate(mid, aid, "IS_SUBFIELD_OF", weight=1.0)

    neighbors = await kg._g.neighbors(mid)
    assert any(n.id == aid for n in neighbors)

    cfg_mod.get_config.cache_clear()


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_register_and_get():
    from core.registry import Registry, PluginType

    reg = Registry()

    class FakeBrain:
        pass

    reg.register("fake_brain", FakeBrain, PluginType.BRAIN, description="test")
    assert reg.is_registered("fake_brain")

    info = reg.get_info("fake_brain")
    assert info.plugin_type == PluginType.BRAIN


def test_registry_decorator():
    from core.registry import Registry, PluginType

    reg = Registry()

    @reg.brain("decorated_brain", description="decorated")
    class MyBrain:
        pass

    assert reg.is_registered("decorated_brain")
    assert reg.get_info("decorated_brain").description == "decorated"


def test_registry_instantiate():
    from core.registry import Registry, PluginType

    reg = Registry()

    class Counter:
        count = 0
        def __init__(self):
            Counter.count += 1

    reg.register("counter", Counter, PluginType.TOOL)
    inst1 = reg.get_instance("counter")
    inst2 = reg.get_instance("counter")
    assert inst1 is inst2        # singleton
    assert Counter.count == 1    # only created once


def test_registry_disable_enable():
    from core.registry import Registry, PluginType

    reg = Registry()

    class MyTool:
        pass

    reg.register("my_tool", MyTool, PluginType.TOOL)
    reg.disable("my_tool")

    with pytest.raises(RuntimeError):
        reg.get_instance("my_tool")

    reg.enable("my_tool")
    inst = reg.get_instance("my_tool")
    assert isinstance(inst, MyTool)
