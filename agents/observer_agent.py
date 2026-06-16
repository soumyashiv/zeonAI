"""
JARVIS Observer Agent
Monitors system health, detects anomalies, and reports on agent activity.
Runs a background heartbeat loop to keep all agents alive.
"""
from __future__ import annotations

import asyncio
import platform
from datetime import datetime, timezone
from typing import Any

import psutil
import structlog

from agents.base_agent import BaseAgent
from core.event_bus import AgentMessage, MessageType

log = structlog.get_logger(__name__)


class ObserverAgent(BaseAgent):
    name = "observer_agent"
    description = "System monitor. Tracks resource usage, agent health, anomalies."

    def __init__(self) -> None:
        super().__init__()
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_interval = 60  # seconds

    async def _on_start(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._log.info("observer.started", platform=platform.system())

    async def _on_stop(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                metrics = await self.collect_metrics()
                self._log.info(
                    "observer.heartbeat",
                    cpu=f"{metrics['cpu_percent']}%",
                    ram=f"{metrics['ram_used_gb']:.1f}GB/{metrics['ram_total_gb']:.1f}GB",
                    disk=f"{metrics['disk_free_gb']:.1f}GB free",
                )
                # Alert if resources are critically low
                if metrics["ram_percent"] > 90:
                    self._log.warning("observer.high_ram", percent=metrics["ram_percent"])
                if metrics["cpu_percent"] > 95:
                    self._log.warning("observer.high_cpu", percent=metrics["cpu_percent"])
            except Exception as e:
                self._log.error("observer.heartbeat_error", error=str(e))
            await asyncio.sleep(self._heartbeat_interval)

    async def collect_metrics(self) -> dict[str, Any]:
        loop = asyncio.get_event_loop()

        def _gather():
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return cpu, ram, disk

        cpu, ram, disk = await loop.run_in_executor(None, _gather)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": cpu,
            "ram_percent": ram.percent,
            "ram_used_gb": ram.used / 1e9,
            "ram_total_gb": ram.total / 1e9,
            "ram_available_gb": ram.available / 1e9,
            "disk_free_gb": disk.free / 1e9,
            "disk_total_gb": disk.total / 1e9,
        }

    async def execute(self, plan: dict[str, Any], original_message: AgentMessage) -> Any:
        action = original_message.payload.get("action", "metrics")

        if action == "metrics":
            metrics = await self.collect_metrics()
            await self.send(
                original_message.from_agent,
                MessageType.TASK_RESULT,
                {"metrics": metrics},
                priority=5,
            )
            return metrics

        return {"status": "observer running"}

    async def verify(self, result: Any, plan: dict) -> tuple[bool, str]:
        return True, "observer report generated"
