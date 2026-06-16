"""
JARVIS LLM Backend
Unified interface for local LLM inference.
Supports: Ollama (primary) and llama-cpp-python (fallback).
Primary model: Qwen3.5-9B-Q6_K via Ollama.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def health_check(self) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """Uses the official ollama Python client."""

    def __init__(self) -> None:
        import ollama as _ollama
        self._client = _ollama.AsyncClient(host=cfg.llm_host)
        self._model = cfg.llm_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        opts: dict[str, Any] = {"temperature": temperature or cfg.llm_temperature}
        if max_tokens:
            opts["num_predict"] = max_tokens
        if stop:
            opts["stop"] = stop

        response = await self._client.chat(
            model=self._model,
            messages=messages,
            options=opts,
        )
        return response["message"]["content"]

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        opts = {"temperature": temperature or cfg.llm_temperature}
        async for chunk in await self._client.chat(
            model=self._model,
            messages=messages,
            options=opts,
            stream=True,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings(model=self._model, prompt=text)
        return resp["embedding"]

    async def health_check(self) -> bool:
        try:
            models = await self._client.list()
            names = [m["name"] for m in models.get("models", [])]
            ok = any(self._model in n for n in names)
            if not ok:
                log.warning(
                    "ollama.model_not_found",
                    model=self._model,
                    available=names,
                )
            return ok
        except Exception as e:
            log.error("ollama.health_check_failed", error=str(e))
            return False


# ─────────────────────────────────────────────────────────────────────────────
# llama-cpp-python backend (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class LlamaCppBackend(LLMBackend):
    """Direct GGUF loading via llama-cpp-python. Used when Ollama unavailable."""

    def __init__(self) -> None:
        from llama_cpp import Llama
        log.info("llamacpp.loading", path=cfg.gguf_path)
        self._llm = Llama(
            model_path=cfg.gguf_path,
            n_ctx=cfg.llm_context_window,
            n_threads=6,        # leave 2 threads for system
            verbose=False,
        )
        log.info("llamacpp.loaded")

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature or cfg.llm_temperature,
                max_tokens=max_tokens or 2048,
                stop=stop,
            ),
        )
        return result["choices"][0]["message"]["content"]

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            lambda: self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature or cfg.llm_temperature,
                stream=True,
            ),
        )
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm.create_embedding(text),
        )
        return result["data"][0]["embedding"]

    async def health_check(self) -> bool:
        from pathlib import Path
        return Path(cfg.gguf_path).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Factory + singleton
# ─────────────────────────────────────────────────────────────────────────────

_llm: LLMBackend | None = None


async def get_llm() -> LLMBackend:
    """
    Returns the active LLM backend.
    Tries Ollama first; falls back to llama-cpp-python if unavailable.
    """
    global _llm
    if _llm is not None:
        return _llm

    if cfg.llm_backend == "ollama":
        backend = OllamaBackend()
        if await backend.health_check():
            log.info("llm.backend", backend="ollama", model=cfg.llm_model)
            _llm = backend
            return _llm
        log.warning("ollama.unavailable", fallback="llama-cpp-python")

    # Fallback to llama-cpp-python
    if cfg.gguf_path:
        _llm = LlamaCppBackend()
        log.info("llm.backend", backend="llama-cpp-python")
        return _llm

    raise RuntimeError(
        "No LLM backend available. "
        "Start Ollama (`ollama serve`) or set JARVIS_GGUF_PATH in .env"
    )


async def chat(messages: list[dict], **kwargs) -> str:
    llm = await get_llm()
    return await llm.chat(messages, **kwargs)
