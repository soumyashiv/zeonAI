"""
Semantic Memory — Qdrant vector store with sentence-transformers embeddings.
Stores factual knowledge, research results, and document chunks.
Falls back to in-process SimpleVectorStore when Qdrant is unavailable.
"""
from __future__ import annotations

import uuid
import json
from dataclasses import dataclass
from typing import Any

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


@dataclass
class SemanticEntry:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Simple in-process fallback (no Qdrant needed for Phase 2 dev)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVectorStore:
    """
    Naive cosine-similarity vector store using numpy.
    Used when Qdrant is not running. Sufficient for dev/testing.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []
        log.info("semantic_memory.backend", backend="simple-in-process")

    def _cosine(self, a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    async def upsert(
        self, entry_id: str, vector: list[float], payload: dict
    ) -> None:
        self._entries.append({"id": entry_id, "vector": vector, "payload": payload})

    async def search(
        self, query_vector: list[float], limit: int = 5, collection: str = ""
    ) -> list[SemanticEntry]:
        scored = []
        for e in self._entries:
            score = self._cosine(query_vector, e["vector"])
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SemanticEntry(
                id=e["id"],
                text=e["payload"].get("text", ""),
                metadata=e["payload"],
                score=s,
            )
            for s, e in scored[:limit]
        ]

    async def delete(self, entry_id: str) -> None:
        self._entries = [e for e in self._entries if e["id"] != entry_id]

    @property
    def count(self) -> int:
        return len(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant backend
# ─────────────────────────────────────────────────────────────────────────────

class QdrantStore:
    COLLECTIONS = [
        "zeon_semantic",
        "zeon_episodic",
        "zeon_skills",
        "zeon_web_cache",
    ]

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if cfg.qdrant_in_memory:
            self._client = QdrantClient(":memory:")
            log.info("semantic_memory.backend", backend="qdrant-in-memory")
        else:
            self._client = QdrantClient(
                host=cfg.qdrant_host, port=cfg.qdrant_port
            )
            log.info("semantic_memory.backend", backend="qdrant-server")

        for col in self.COLLECTIONS:
            self._client.recreate_collection(
                collection_name=col,
                vectors_config=VectorParams(
                    size=cfg.embedding_dimension, distance=Distance.COSINE
                ),
            )

    async def upsert(
        self,
        entry_id: str,
        vector: list[float],
        payload: dict,
        collection: str = "zeon_semantic",
    ) -> None:
        from qdrant_client.models import PointStruct
        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=entry_id, vector=vector, payload=payload)],
        )

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        collection: str = "zeon_semantic",
    ) -> list[SemanticEntry]:
        results = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
        )
        return [
            SemanticEntry(
                id=str(r.id),
                text=r.payload.get("text", ""),
                metadata=r.payload,
                score=r.score,
            )
            for r in results
        ]

    async def delete(
        self, entry_id: str, collection: str = "zeon_semantic"
    ) -> None:
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[entry_id]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Memory facade
# ─────────────────────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    High-level semantic memory. Handles embedding + storage.
    Auto-selects Qdrant or SimpleVectorStore based on availability.
    """

    def __init__(self) -> None:
        self._store = self._init_store()
        self._embedder = None   # lazy-loaded

    def _init_store(self):
        try:
            store = QdrantStore()
            return store
        except Exception as e:
            log.warning("semantic_memory.qdrant_failed", error=str(e), fallback="simple")
            return SimpleVectorStore()

    async def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(cfg.embedding_model)
                log.info("semantic_memory.embedder_loaded", model=cfg.embedding_model)
            except ImportError:
                log.warning("sentence_transformers not installed; using zero vector")
                return [0.0] * cfg.embedding_dimension
        import asyncio
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None, lambda: self._embedder.encode(text).tolist()
        )
        return vec

    async def store(
        self,
        text: str,
        *,
        metadata: dict | None = None,
        collection: str = "zeon_semantic",
        entry_id: str | None = None,
    ) -> str:
        eid = entry_id or str(uuid.uuid4())
        vector = await self._embed(text)
        payload = {"text": text, **(metadata or {})}
        if isinstance(self._store, QdrantStore):
            await self._store.upsert(eid, vector, payload, collection=collection)
        else:
            await self._store.upsert(eid, vector, payload)
        return eid

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        collection: str = "zeon_semantic",
    ) -> list[SemanticEntry]:
        vector = await self._embed(query)
        if isinstance(self._store, QdrantStore):
            return await self._store.search(vector, limit=limit, collection=collection)
        return await self._store.search(vector, limit=limit)

    async def delete(self, entry_id: str, collection: str = "zeon_semantic") -> None:
        if isinstance(self._store, QdrantStore):
            await self._store.delete(entry_id, collection=collection)
        else:
            await self._store.delete(entry_id)


_semantic: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    global _semantic
    if _semantic is None:
        _semantic = SemanticMemory()
    return _semantic
