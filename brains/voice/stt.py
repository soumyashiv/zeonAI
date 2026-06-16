"""
JARVIS Voice — Speech-to-Text (STT)
Primary: faster-whisper (CTranslate2 — CPU-optimised, 4x faster than openai-whisper)
Fallback: vosk (if faster-whisper unavailable)
Supports: EN, HI, and auto-detection.
"""
from __future__ import annotations

import asyncio
import io
import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from core.config import get_config
from brains.voice.vad import AudioSegment

log = structlog.get_logger(__name__)
cfg = get_config()


class STT:
    """
    Transcribes audio segments to text.
    Automatically detects language or uses configured default.
    """

    # faster-whisper model size: tiny/base/small/medium
    # base is ~140 MB, fast on CPU. tiny is ~75 MB.
    DEFAULT_MODEL = cfg.whisper_model_size   # from .env (default: "base")

    def __init__(self) -> None:
        self._model = None
        self._backend = "none"
        self._lock = asyncio.Lock()
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.DEFAULT_MODEL,
                device="cpu",
                compute_type="int8",    # INT8 quantization — fastest on CPU
            )
            self._backend = "faster-whisper"
            log.info("stt.model_loaded", backend="faster-whisper", size=self.DEFAULT_MODEL)
        except ImportError:
            log.warning("stt.faster_whisper_unavailable", fallback="vosk")
            self._try_vosk()
        except Exception as e:
            log.error("stt.load_failed", error=str(e))
            self._backend = "none"

    def _try_vosk(self) -> None:
        try:
            from vosk import Model as VoskModel, KaldiRecognizer
            model_dir = Path(cfg.jarvis_root) / "models" / "vosk-model-en"
            if not model_dir.exists():
                log.warning("stt.vosk_model_missing", path=str(model_dir))
                self._backend = "none"
                return
            self._vosk_model = VoskModel(str(model_dir))
            self._backend = "vosk"
            log.info("stt.model_loaded", backend="vosk")
        except ImportError:
            log.warning("stt.vosk_unavailable")
            self._backend = "none"

    async def transcribe(
        self,
        segment: AudioSegment,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe an AudioSegment.
        Returns: {text, language, confidence, duration_ms}
        """
        if self._backend == "none":
            return {"text": "", "language": "en", "confidence": 0.0, "error": "no STT backend"}

        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._transcribe_sync(segment, language)
            )

    def _transcribe_sync(self, segment: AudioSegment, language: Optional[str]) -> dict:
        if self._backend == "faster-whisper":
            return self._fw_transcribe(segment, language)
        elif self._backend == "vosk":
            return self._vosk_transcribe(segment)
        return {"text": "", "language": "en", "confidence": 0.0}

    def _fw_transcribe(self, segment: AudioSegment, language: Optional[str]) -> dict:
        """faster-whisper transcription."""
        # Convert raw bytes → float32 numpy array
        audio_int16 = np.frombuffer(segment.data, dtype=np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        lang = language or (cfg.voice_language_list[0] if cfg.voice_language_list else None)

        try:
            segments_iter, info = self._model.transcribe(
                audio_f32,
                language=lang,
                beam_size=3,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            texts = []
            for seg in segments_iter:
                texts.append(seg.text.strip())

            full_text = " ".join(texts).strip()
            detected_lang = info.language if hasattr(info, "language") else (lang or "en")
            confidence = float(info.language_probability) if hasattr(info, "language_probability") else 1.0

            log.info("stt.transcribed",
                     text=full_text[:60],
                     lang=detected_lang,
                     duration_ms=segment.duration_ms)

            return {
                "text": full_text,
                "language": detected_lang,
                "confidence": round(confidence, 3),
                "duration_ms": segment.duration_ms,
            }
        except Exception as e:
            log.error("stt.fw_transcribe_failed", error=str(e))
            return {"text": "", "language": "en", "confidence": 0.0, "error": str(e)}

    def _vosk_transcribe(self, segment: AudioSegment) -> dict:
        """Vosk offline transcription."""
        from vosk import KaldiRecognizer
        import json as jsonlib

        rec = KaldiRecognizer(self._vosk_model, segment.sample_rate)
        rec.AcceptWaveform(segment.data)
        result = jsonlib.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        log.info("stt.transcribed", text=text[:60], backend="vosk")
        return {"text": text, "language": "en", "confidence": 1.0, "duration_ms": segment.duration_ms}


_stt: STT | None = None


def get_stt() -> STT:
    global _stt
    if _stt is None:
        _stt = STT()
    return _stt
