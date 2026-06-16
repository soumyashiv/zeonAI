"""
JARVIS Voice — Voice Activity Detection (VAD)
Records from microphone until speech ends.
Primary: webrtcvad (fast, rule-based, no model)
Fallback: energy-threshold silence detection
"""
from __future__ import annotations

import asyncio
import collections
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class AudioSegment:
    data: bytes           # raw PCM int16
    sample_rate: int      # always 16000
    duration_ms: int


class VAD:
    """
    Records speech from mic after wake word triggers.
    Stops when silence > `silence_duration_ms` milliseconds.
    Returns raw PCM bytes of the utterance.
    """

    SAMPLE_RATE   = 16000
    FRAME_MS      = 30          # webrtcvad supports 10, 20, or 30ms
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480
    MAX_RECORD_S  = 30          # safety cap

    def __init__(
        self,
        aggressiveness: int = 2,          # 0–3; higher = more aggressive filtering
        silence_duration_ms: int = 800,   # stop after this much silence
    ) -> None:
        self._aggressiveness = aggressiveness
        self._silence_ms = silence_duration_ms
        self._vad = self._load_vad()
        log.info("vad.initialized", backend=self._backend)

    def _load_vad(self):
        try:
            import webrtcvad
            vad = webrtcvad.Vad(self._aggressiveness)
            self._backend = "webrtcvad"
            return vad
        except ImportError:
            log.warning("vad.webrtcvad_unavailable", fallback="energy")
            self._backend = "energy"
            return None

    async def record_utterance(self) -> Optional[AudioSegment]:
        """
        Block until an utterance is captured.
        Returns None if nothing was said within MAX_RECORD_S.
        """
        loop = asyncio.get_event_loop()
        segment = await loop.run_in_executor(None, self._record_sync)
        return segment

    def _record_sync(self) -> Optional[AudioSegment]:
        try:
            import sounddevice as sd
        except ImportError:
            log.error("vad.sounddevice_missing")
            return None

        voiced_frames = []
        ring_buffer = collections.deque(maxlen=self._silence_frames)
        triggered = False
        recording = []

        log.debug("vad.recording_start")

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.FRAME_SAMPLES,
            ) as stream:
                frames_recorded = 0
                max_frames = int(self.MAX_RECORD_S * 1000 / self.FRAME_MS)

                while frames_recorded < max_frames:
                    raw, _ = stream.read(self.FRAME_SAMPLES)
                    frame = bytes(raw)
                    frames_recorded += 1

                    is_speech = self._is_speech(frame)

                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        num_voiced = sum(1 for _, s in ring_buffer if s)
                        if num_voiced > 0.8 * ring_buffer.maxlen:
                            triggered = True
                            recording.extend(f for f, _ in ring_buffer)
                            ring_buffer.clear()
                            log.debug("vad.speech_start")
                    else:
                        recording.append(frame)
                        ring_buffer.append((frame, is_speech))
                        num_unvoiced = sum(1 for _, s in ring_buffer if not s)
                        if num_unvoiced > 0.9 * ring_buffer.maxlen:
                            log.debug("vad.speech_end", frames=len(recording))
                            break

        except Exception as e:
            log.error("vad.record_error", error=str(e))
            return None

        if not recording:
            return None

        audio_bytes = b"".join(recording)
        duration_ms = int(len(recording) * self.FRAME_MS)
        log.info("vad.captured", duration_ms=duration_ms, bytes=len(audio_bytes))
        return AudioSegment(data=audio_bytes, sample_rate=self.SAMPLE_RATE, duration_ms=duration_ms)

    @property
    def _silence_frames(self) -> int:
        return max(1, self._silence_ms // self.FRAME_MS)

    def _is_speech(self, frame: bytes) -> bool:
        if self._backend == "webrtcvad" and self._vad:
            try:
                return self._vad.is_speech(frame, self.SAMPLE_RATE)
            except Exception:
                pass
        # Energy fallback
        audio = np.frombuffer(frame, dtype=np.int16)
        return float(np.abs(audio).mean()) > 500


_vad: VAD | None = None


def get_vad() -> VAD:
    global _vad
    if _vad is None:
        _vad = VAD()
    return _vad
