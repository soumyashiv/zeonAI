"""
ZEON Core Configuration
Loads settings from .env and provides a singleton config object.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ZeonConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parents[1] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Mode ──────────────────────────────────────────────────────
    zeon_env: str = Field("development", alias="ZEON_ENV")
    dev_auto_approve: bool = Field(True, alias="ZEON_DEV_AUTO_APPROVE")

    # ── LLM ───────────────────────────────────────────────────────
    llm_backend: str = Field("ollama", alias="ZEON_LLM_BACKEND")
    llm_model: str = Field("qwen3.5-9b", alias="ZEON_LLM_MODEL")
    llm_host: str = Field("http://localhost:11434", alias="ZEON_LLM_HOST")
    llm_context_window: int = Field(32768, alias="ZEON_LLM_CONTEXT_WINDOW")
    llm_temperature: float = Field(0.7, alias="ZEON_LLM_TEMPERATURE")
    gguf_path: str = Field("", alias="ZEON_GGUF_PATH")

    # ── Qdrant ────────────────────────────────────────────────────
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_in_memory: bool = Field(True, alias="QDRANT_IN_MEMORY")

    # ── Neo4j ─────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("zeon_neo4j_pass", alias="NEO4J_PASSWORD")

    # ── Redis ─────────────────────────────────────────────────────
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_use_memory: bool = Field(True, alias="REDIS_USE_MEMORY")

    # ── SQLite ────────────────────────────────────────────────────
    sqlite_path: str = Field("data/sqlite/zeon.db", alias="SQLITE_PATH")

    # ── Voice ─────────────────────────────────────────────────────
    voice_enabled: bool = Field(False, alias="ZEON_VOICE_ENABLED")
    wake_word: str = Field("hey zeon", alias="ZEON_WAKE_WORD")
    voice_languages: str = Field("en,hi", alias="ZEON_VOICE_LANGUAGES")
    whisper_model_size: str = Field("base", alias="WHISPER_MODEL_SIZE")
    tts_engine: str = Field("piper", alias="ZEON_TTS_ENGINE")

    # ── Paths ─────────────────────────────────────────────────────
    zeon_root: str = Field(".", alias="ZEON_ROOT")
    logs_dir: str = Field("data/logs", alias="ZEON_LOGS_DIR")
    skills_dir: str = Field("skills_registry", alias="ZEON_SKILLS_DIR")
    improvements_dir: str = Field("improvements", alias="ZEON_IMPROVEMENTS_DIR")

    # ── Embeddings ────────────────────────────────────────────────
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(384, alias="EMBEDDING_DIMENSION")

    # ── Security / Quality ────────────────────────────────────────
    audit_all_actions: bool = Field(True, alias="ZEON_AUDIT_ALL_ACTIONS")
    max_plan_revision_loops: int = Field(3, alias="ZEON_MAX_PLAN_REVISION_LOOPS")
    critic_min_score: int = Field(7, alias="ZEON_CRITIC_MIN_SCORE")

    @property
    def is_dev(self) -> bool:
        return self.zeon_env == "development"

    @property
    def voice_language_list(self) -> list[str]:
        return [lang.strip() for lang in self.voice_languages.split(",")]

    @property
    def sqlite_path_resolved(self) -> Path:
        p = Path(self.sqlite_path)
        if not p.is_absolute():
            p = Path(self.zeon_root) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_config() -> ZeonConfig:
    """Return singleton config instance."""
    return ZeonConfig()
