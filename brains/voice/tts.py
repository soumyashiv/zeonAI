"""
JARVIS Voice — Text-to-Speech (TTS)
Primary: Piper TTS (offline, neural, fast on CPU, ~50ms latency)
Fallback: pyttsx3 (system TTS, instant, lower quality)
Supports: EN and HI voices.
"""
from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()

# Piper voice model paths (download separately if needed)
# https://github.com/rhasspy/piper/releases
VOICE_MAP = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-google-medium",
}

PIPER_MODELS_DIR = Path(cfg.jarvis_root) / "models" / "piper"


class TTS:
    """Speaks text aloud using Piper TTS or pyttsx3 fallback."""

    def __init__(self) -> None:
        self._backend = "none"
        self._engine = None          # pyttsx3 engine
        self._lock = asyncio.Lock()
        self._detect_backend()

    def _detect_backend(self) -> None:
        # Check for Piper binary
        piper_bin = Path(cfg.jarvis_root) / "models" / "piper" / "piper.exe"
        if not piper_bin.exists():
            piper_bin = Path(cfg.jarvis_root) / "models" / "piper" / "piper"
        if piper_bin.exists():
            self._piper_bin = str(piper_bin)
            self._backend = "piper"
            log.info("tts.backend", backend="piper", bin=str(piper_bin))
            return

        # Try pyttsx3
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 165)
            self._engine.setProperty("volume", 1.0)
            self._backend = "pyttsx3"
            log.info("tts.backend", backend="pyttsx3")
            return
        except Exception:
            pass

        # Try Windows SAPI directly as last resort
        try:
            import win32com.client
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
            self._backend = "sapi"
            log.info("tts.backend", backend="sapi")
            return
        except Exception:
            pass

        log.warning("tts.no_backend", note="install pyttsx3 or place piper binary")
        self._backend = "none"

    async def speak(self, text: str, language: str = "en") -> None:
        """Speak text aloud. Non-blocking — waits for completion."""
        if not text.strip():
            return

        log.info("tts.speaking", text=text[:60], lang=language, backend=self._backend)

        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._speak_sync(text, language))

    def _speak_sync(self, text: str, language: str) -> None:
        if self._backend == "piper":
            self._piper_speak(text, language)
        elif self._backend == "pyttsx3":
            self._pyttsx3_speak(text)
        elif self._backend == "sapi":
            self._sapi_speak(text)
        else:
            log.warning("tts.no_backend_speak")

    def _piper_speak(self, text: str, language: str) -> None:
        """Use Piper TTS binary to generate and play audio."""
        voice_name = VOICE_MAP.get(language, VOICE_MAP["en"])
        model_path = PIPER_MODELS_DIR / f"{voice_name}.onnx"

        if not model_path.exists():
            log.warning("tts.piper_model_missing", model=str(model_path), fallback="pyttsx3")
            self._pyttsx3_speak(text)
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            subprocess.run(
                [self._piper_bin, "--model", str(model_path), "--output_file", out_path],
                input=text.encode(),
                capture_output=True,
                timeout=15,
                check=True,
            )
            # Play via sounddevice
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(out_path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            log.error("tts.piper_failed", error=str(e))
            self._pyttsx3_speak(text)
        finally:
            Path(out_path).unlink(missing_ok=True)

    def _pyttsx3_speak(self, text: str) -> None:
        try:
            import pyttsx3
            if self._engine is None:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 165)
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            log.error("tts.pyttsx3_failed", error=str(e))

    def _sapi_speak(self, text: str) -> None:
        try:
            self._sapi.Speak(text)
        except Exception as e:
            log.error("tts.sapi_failed", error=str(e))

    async def speak_async(self, text: str, language: str = "en") -> asyncio.Task:
        """Start speaking without blocking — returns the task."""
        return asyncio.create_task(self.speak(text, language))


_tts: TTS | None = None


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS()
    return _tts
