"""
ZEON Plugin Registry
Central registry for all Brain modules and Agents.
Supports dynamic registration, discovery, and health-checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Type

import structlog

log = structlog.get_logger(__name__)


class PluginType(str, Enum):
    BRAIN = "brain"
    AGENT = "agent"
    TOOL  = "tool"


@dataclass
class PluginInfo:
    name: str
    plugin_type: PluginType
    cls: Type
    description: str = ""
    version: str = "0.1.0"
    enabled: bool = True
    instance: Any = field(default=None, repr=False)


class Registry:
    """
    Singleton plugin registry.
    All brains, agents, and tools register here at startup.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}

    # ── Registration ──────────────────────────────────────────────

    def register(
        self,
        name: str,
        cls: Type,
        plugin_type: PluginType,
        *,
        description: str = "",
        version: str = "0.1.0",
    ) -> None:
        if name in self._plugins:
            log.warning("registry.duplicate", name=name)
            return
        self._plugins[name] = PluginInfo(
            name=name,
            plugin_type=plugin_type,
            cls=cls,
            description=description,
            version=version,
        )
        log.debug("registry.registered", name=name, type=plugin_type.value)

    def brain(self, name: str, description: str = ""):
        """Class decorator for auto-registering brain modules."""
        def decorator(cls):
            self.register(name, cls, PluginType.BRAIN, description=description)
            return cls
        return decorator

    def agent(self, name: str, description: str = ""):
        """Class decorator for auto-registering agents."""
        def decorator(cls):
            self.register(name, cls, PluginType.AGENT, description=description)
            return cls
        return decorator

    def tool(self, name: str, description: str = ""):
        """Class decorator for auto-registering tools."""
        def decorator(cls):
            self.register(name, cls, PluginType.TOOL, description=description)
            return cls
        return decorator

    # ── Instantiation ─────────────────────────────────────────────

    def get_instance(self, name: str, *args, **kwargs) -> Any:
        """Get or create a singleton instance of a registered plugin."""
        info = self._plugins.get(name)
        if not info:
            raise KeyError(f"Plugin '{name}' not registered.")
        if not info.enabled:
            raise RuntimeError(f"Plugin '{name}' is disabled.")
        if info.instance is None:
            info.instance = info.cls(*args, **kwargs)
            log.debug("registry.instantiated", name=name)
        return info.instance

    def reset_instance(self, name: str) -> None:
        """Force re-instantiation on next get_instance call."""
        if name in self._plugins:
            self._plugins[name].instance = None

    # ── Discovery ─────────────────────────────────────────────────

    def list_all(self, plugin_type: PluginType | None = None) -> list[PluginInfo]:
        plugins = list(self._plugins.values())
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        return plugins

    def get_info(self, name: str) -> PluginInfo | None:
        return self._plugins.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._plugins

    def enable(self, name: str) -> None:
        if name in self._plugins:
            self._plugins[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._plugins:
            self._plugins[name].enabled = False
            self._plugins[name].instance = None

    def summary(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for info in self._plugins.values():
            bucket = result.setdefault(info.plugin_type.value, [])
            status = "✓" if info.enabled else "✗"
            bucket.append(f"{status} {info.name} v{info.version}")
        return result


# ── Singleton ─────────────────────────────────────────────────────

_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
