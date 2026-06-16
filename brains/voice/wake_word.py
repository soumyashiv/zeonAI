"""
JARVIS Voice — Wake Word Detection
Uses openWakeWord for "hey jarvis" detection.
Falls back to a simple energy-threshold trigger when openWakeWord unavailable.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Callable, Any

import numpy as np
import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()


class WakeWordDetector:
    """
    Listens on microphone for the configured wake word.
    Calls `on_detected` callback when wake word is heard.
    Runs in a background thread to not block the event loop.
    """

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 1280        # 80ms @ 16kHz (openWakeWord requirement)
    WAKE_THRESHOLD = 0.5        # confidence threshold

    def __init__(self, on_detected: Callable[[], None]) -> None:
        self._on_detected = on_detected
        self._running = False
        self._thread: threading.Thread | None = None
        self._model = None
        self._backend = "none"
        self._loop: asyncio.AbstractEventLoop | None = None

    def _load_model(self) -> None:
        try:
            from openwakeword.model import Model
            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            self._backend = "openwakeword"
            log.info("wake_word.model_loaded", backend="openwakeword")
        except Exception as e:
            log.warning("wake_word.oww_unavailable", error=str(e), fallback="energy")
            self._backend = "energy"

    def _audio_stream(self):
        """Return a sounddevice RawInputStream context manager."""
        import sounddevice as sd
        return sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK_SAMPLES,
        )

    def _trigger(self) -> None:
        """Signal wake word detected — runs in thread, schedules async callback."""
        log.info("wake_word.detected", word=cfg.wake_word)
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._async_trigger())
            )

    async def _async_trigger(self) -> None:
        self._on_detected()

    def _run_oww(self) -> None:
        """openWakeWord detection loop."""
        try:
            with self._audio_stream() as stream:
                log.info("wake_word.listening", word=cfg.wake_word, backend="openwakeword")
                while self._running:
                    raw, _ = stream.read(self.CHUNK_SAMPLES)
                    audio = np.frombuffer(raw, dtype=np.int16)
                    preds = self._model.predict(audio)
                    for score in preds.values():
                        if score >= self.WAKE_THRESHOLD:
                            self._trigger()
                            # Brief cooldown after trigger
                            import time; time.sleep(1.5)
                            break
        except Exception as e:
            if self._running:
                log.error("wake_word.stream_error", error=str(e))

    def _run_energy(self) -> None:
        """Energy-threshold fallback: trigger on loud spike (for testing)."""
        ENERGY_THRESHOLD = 3000
        try:
            with self._audio_stream() as stream:
                log.info("wake_word.listening", word=cfg.wake_word, backend="energy-threshold")
                while self._running:
                    raw, _ = stream.read(self.CHUNK_SAMPLES)
                    audio = np.frombuffer(raw, dtype=np.int16)
                    energy = np.abs(audio).mean()
                    if energy > ENERGY_THRESHOLD:
                        self._trigger()
                        import time; time.sleep(1.5)
        except Exception as e:
            if self._running:
                log.error("wake_word.energy_error", error=str(e))

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._running = True
        self._loop = loop
        self._load_model()

        target = self._run_oww if self._backend == "openwakeword" else self._run_energy
        self._thread = threading.Thread(target=target, daemon=True, name="wake-word")
        self._thread.start()
        log.info("wake_word.started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("wake_word.stopped")
