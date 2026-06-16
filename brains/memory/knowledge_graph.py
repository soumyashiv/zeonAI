"""
Knowledge Graph Memory — Neo4j-backed graph with in-process NetworkX fallback.
Stores: Concepts, Entities, Skills, Goals, Experiences and their relationships.
Enables multi-hop reasoning that vector search cannot provide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


@dataclass
class GraphNode:
    id: str
    label: str          # Concept | Entity | Skill | Goal | Experience
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelation:
    from_id: str
    to_id: str
    rel_type: str       # RELATES_TO | USED | ACHIEVED | DEPENDS_ON | DERIVED_FROM
    properties: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# NetworkX in-process fallback
# ─────────────────────────────────────────────────────────────────────────────

class NetworkXGraph:
    """
    In-process directed graph using NetworkX.
    Persisted to a JSON file on disk between sessions.
    """

    def __init__(self) -> None:
        import networkx as nx
        from pathlib import Path
        import json

        self._G = nx.DiGraph()
        self._persist_path = Path(cfg.zeon_root) / "data" / "knowledge_graph.json"
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        log.info("knowledge_graph.backend", backend="networkx-in-process")

    def _load(self) -> None:
        import json
        if self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text())
                import networkx as nx
                self._G = nx.DiGraph()
                for n in data.get("nodes", []):
                    self._G.add_node(n["id"], **n)
                for e in data.get("edges", []):
                    self._G.add_edge(e["from_id"], e["to_id"], **e)
                log.info("knowledge_graph.loaded",
                         nodes=self._G.number_of_nodes(),
                         edges=self._G.number_of_edges())
            except Exception as ex:
                log.warning("knowledge_graph.load_failed", error=str(ex))

    def _save(self) -> None:
        import json
        nodes = [dict(d) for _, d in self._G.nodes(data=True)]
        edges = [{"from_id": u, "to_id": v, **dict(d)}
                 for u, v, d in self._G.edges(data=True)]
        self._persist_path.write_text(json.dumps({"nodes": nodes, "edges": edges}))

    async def add_node(self, node: GraphNode) -> None:
        self._G.add_node(node.id, id=node.id, label=node.label, **node.properties)
        self._save()

    async def add_relation(self, rel: GraphRelation) -> None:
        self._G.add_edge(
            rel.from_id, rel.to_id,
            from_id=rel.from_id, to_id=rel.to_id,
            rel_type=rel.rel_type, **rel.properties
        )
        self._save()

    async def get_node(self, node_id: str) -> GraphNode | None:
        if node_id not in self._G:
            return None
        d = self._G.nodes[node_id]
        return GraphNode(id=node_id, label=d.get("label", ""), properties=dict(d))

    async def neighbors(self, node_id: str, rel_type: str | None = None) -> list[GraphNode]:
        if node_id not in self._G:
            return []
        result = []
        for nbr_id in self._G.successors(node_id):
            edge_data = self._G.edges[node_id, nbr_id]
            if rel_type and edge_data.get("rel_type") != rel_type:
                continue
            d = self._G.nodes[nbr_id]
            result.append(GraphNode(id=nbr_id, label=d.get("label",""), properties=dict(d)))
        return result

    async def search_by_label(self, label: str, limit: int = 10) -> list[GraphNode]:
        results = []
        for nid, data in self._G.nodes(data=True):
            if data.get("label") == label:
                results.append(GraphNode(id=nid, label=label, properties=dict(data)))
            if len(results) >= limit:
                break
        return results

    async def find_path(self, from_id: str, to_id: str) -> list[str]:
        import networkx as nx
        try:
            return nx.shortest_path(self._G, from_id, to_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    async def stats(self) -> dict[str, int]:
        return {
            "nodes": self._G.number_of_nodes(),
            "edges": self._G.number_of_edges(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j backend
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGraph:
    SCHEMA_QUERIES = [
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT goal_id IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE",
        "CREATE CONSTRAINT experience_id IF NOT EXISTS FOR (e:Experience) REQUIRE e.id IS UNIQUE",
    ]

    def __init__(self) -> None:
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )
        self._driver.verify_connectivity()
        self._init_schema()
        log.info("knowledge_graph.backend", backend="neo4j")

    def _init_schema(self) -> None:
        with self._driver.session() as session:
            for q in self.SCHEMA_QUERIES:
                try:
                    session.run(q)
                except Exception:
                    pass  # constraint may already exist

    async def add_node(self, node: GraphNode) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        props = {**node.properties, "id": node.id}
        query = f"MERGE (n:{node.label} {{id: $id}}) SET n += $props"
        await loop.run_in_executor(
            None,
            lambda: self._driver.session().run(query, id=node.id, props=props)
        )

    async def add_relation(self, rel: GraphRelation) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        query = (
            f"MATCH (a {{id:$fid}}), (b {{id:$tid}}) "
            f"MERGE (a)-[r:{rel.rel_type}]->(b) SET r += $props"
        )
        await loop.run_in_executor(
            None,
            lambda: self._driver.session().run(
                query, fid=rel.from_id, tid=rel.to_id, props=rel.properties
            )
        )

    async def get_node(self, node_id: str) -> GraphNode | None:
        import asyncio
        loop = asyncio.get_event_loop()
        def _run():
            with self._driver.session() as s:
                r = s.run("MATCH (n {id:$id}) RETURN n LIMIT 1", id=node_id)
                rec = r.single()
                return rec
        rec = await loop.run_in_executor(None, _run)
        if not rec:
            return None
        n = rec["n"]
        labels = list(n.labels)
        return GraphNode(id=node_id, label=labels[0] if labels else "", properties=dict(n))

    async def neighbors(self, node_id: str, rel_type: str | None = None) -> list[GraphNode]:
        import asyncio
        loop = asyncio.get_event_loop()
        if rel_type:
            q = f"MATCH (a {{id:$id}})-[:{rel_type}]->(b) RETURN b"
        else:
            q = "MATCH (a {id:$id})-->(b) RETURN b"
        def _run():
            with self._driver.session() as s:
                return list(s.run(q, id=node_id))
        recs = await loop.run_in_executor(None, _run)
        result = []
        for rec in recs:
            n = rec["b"]
            labels = list(n.labels)
            result.append(GraphNode(
                id=n.get("id",""),
                label=labels[0] if labels else "",
                properties=dict(n)
            ))
        return result

    async def search_by_label(self, label: str, limit: int = 10) -> list[GraphNode]:
        import asyncio
        loop = asyncio.get_event_loop()
        def _run():
            with self._driver.session() as s:
                return list(s.run(f"MATCH (n:{label}) RETURN n LIMIT $lim", lim=limit))
        recs = await loop.run_in_executor(None, _run)
        result = []
        for rec in recs:
            n = rec["n"]
            labels = list(n.labels)
            result.append(GraphNode(
                id=n.get("id",""),
                label=labels[0] if labels else "",
                properties=dict(n)
            ))
        return result

    async def find_path(self, from_id: str, to_id: str) -> list[str]:
        import asyncio
        loop = asyncio.get_event_loop()
        def _run():
            with self._driver.session() as s:
                r = s.run(
                    "MATCH p=shortestPath((a {id:$fid})-[*]-(b {id:$tid})) RETURN p",
                    fid=from_id, tid=to_id
                )
                rec = r.single()
                if not rec:
                    return []
                return [n.get("id","") for n in rec["p"].nodes]
        return await loop.run_in_executor(None, _run)

    async def stats(self) -> dict[str, int]:
        import asyncio
        loop = asyncio.get_event_loop()
        def _run():
            with self._driver.session() as s:
                nc = s.run("MATCH (n) RETURN count(n) as c").single()["c"]
                ec = s.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                return {"nodes": nc, "edges": ec}
        return await loop.run_in_executor(None, _run)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeGraph facade
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Unified knowledge graph interface.
    Auto-selects Neo4j or NetworkX based on availability.
    """

    def __init__(self) -> None:
        self._g = self._init_backend()

    def _init_backend(self):
        try:
            return Neo4jGraph()
        except Exception as e:
            log.warning("knowledge_graph.neo4j_failed", error=str(e), fallback="networkx")
            return NetworkXGraph()

    async def add_concept(self, name: str, description: str = "", confidence: float = 1.0) -> str:
        nid = str(uuid.uuid4())
        await self._g.add_node(GraphNode(
            id=nid, label="Concept",
            properties={"name": name, "description": description, "confidence": confidence}
        ))
        return nid

    async def add_experience(self, summary: str, embedding_id: str = "") -> str:
        nid = str(uuid.uuid4())
        from datetime import datetime, timezone
        await self._g.add_node(GraphNode(
            id=nid, label="Experience",
            properties={
                "summary": summary,
                "embedding_id": embedding_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ))
        return nid

    async def relate(
        self, from_id: str, to_id: str, rel_type: str, **props
    ) -> None:
        await self._g.add_relation(GraphRelation(
            from_id=from_id, to_id=to_id,
            rel_type=rel_type, properties=props
        ))

    async def get_context_for(self, concept_name: str, hops: int = 2) -> list[GraphNode]:
        """Return all nodes reachable within `hops` from a concept."""
        matches = await self._g.search_by_label("Concept")
        root = next((n for n in matches if n.properties.get("name") == concept_name), None)
        if not root:
            return []
        visited = {root.id}
        frontier = [root]
        result = [root]
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                neighbors = await self._g.neighbors(node.id)
                for nbr in neighbors:
                    if nbr.id not in visited:
                        visited.add(nbr.id)
                        next_frontier.append(nbr)
                        result.append(nbr)
            frontier = next_frontier
        return result

    async def find_path(self, from_id: str, to_id: str) -> list[str]:
        return await self._g.find_path(from_id, to_id)

    async def stats(self) -> dict[str, int]:
        return await self._g.stats()


_kg: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph()
    return _kg
