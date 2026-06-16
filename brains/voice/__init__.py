"""
JARVIS Voice Brain — Unified Voice API
Facade combining wake word, VAD, STT, TTS, and multilingual routing.
"""
from __future__ import annotations

from brains.voice.wake_word import WakeWordDetector
from brains.voice.vad import VAD, AudioSegment, get_vad
from brains.voice.stt import STT, get_stt
from brains.voice.tts import TTS, get_tts
from brains.voice.multilingual import MultilingualRouter, detect_language

__all__ = [
    "WakeWordDetector",
    "VAD", "AudioSegment", "get_vad",
    "STT", "get_stt",
    "TTS", "get_tts",
    "MultilingualRouter", "detect_language",
    "get_voice_brain",
]


class VoiceBrain:
    """Top-level voice subsystem. Manage all voice components from one place."""

    def __init__(self) -> None:
        self.vad = get_vad()
        self.stt = get_stt()
        self.tts = get_tts()
        self.router = MultilingualRouter()
        self._wake_detector: WakeWordDetector | None = None

    async def speak(self, text: str, language: str | None = None) -> None:
        """Detect language if not given, normalise, and speak."""
        routing = self.router.route(text, transcription_lang=language)
        tts_lang = routing["tts_language"]
        clean = routing["normalised_text"]
        await self.tts.speak(clean, language=tts_lang)

    async def listen(self) -> dict:
        """Record one utterance (after wake word or direct call) and transcribe it."""
        segment = await self.vad.record_utterance()
        if segment is None:
            return {"text": "", "language": "en", "confidence": 0.0}

        lang_hint = self.router.get_stt_language()
        result = await self.stt.transcribe(segment, language=lang_hint)
        return result

    def attach_wake_word(self, on_detected) -> WakeWordDetector:
        """Create and return a WakeWordDetector (caller must call .start())."""
        self._wake_detector = WakeWordDetector(on_detected=on_detected)
        return self._wake_detector

    async def stop(self) -> None:
        if self._wake_detector:
            self._wake_detector.stop()


_brain: VoiceBrain | None = None


def get_voice_brain() -> VoiceBrain:
    global _brain
    if _brain is None:
        _brain = VoiceBrain()
    return _brain
