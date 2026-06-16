"""
JARVIS Voice Shell
Full voice interaction loop:
  Wake Word → VAD → STT → LangGraph → TTS → (repeat)

Run with: python main.py --voice
"""
from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import Optional

import structlog

from core.config import get_config
from core.event_bus import get_bus, MessageType
from brains.voice import get_voice_brain

log = structlog.get_logger("jarvis.voice_shell")
cfg = get_config()


class VoiceState(Enum):
    IDLE          = auto()   # waiting for wake word
    LISTENING     = auto()   # recording utterance
    PROCESSING    = auto()   # running through LangGraph
    SPEAKING      = auto()   # TTS output
    INTERRUPTED   = auto()   # interrupted mid-speech


class VoiceShell:
    """
    Full voice pipeline controller.
    Manages state machine: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
    """

    READY_SOUND   = "Ready."
    ERROR_SOUND   = "Sorry, I didn't catch that."
    THINKING_SOUND = "Thinking."

    def __init__(self) -> None:
        self._state = VoiceState.IDLE
        self._voice = get_voice_brain()
        self._bus = get_bus()
        self._running = False
        self._interrupt_event = asyncio.Event()

    # ── Wake word callback ────────────────────────────────────────────────────

    def _on_wake_word(self) -> None:
        """Called from wake word thread when wake word detected."""
        if self._state == VoiceState.IDLE:
            log.info("voice_shell.wake_word_triggered")
            self._state = VoiceState.LISTENING
            asyncio.ensure_future(self._handle_utterance())
        elif self._state == VoiceState.SPEAKING:
            # Interrupt JARVIS mid-speech
            log.info("voice_shell.interrupted")
            self._state = VoiceState.INTERRUPTED
            self._interrupt_event.set()

    # ── Main utterance handler ────────────────────────────────────────────────

    async def _handle_utterance(self) -> None:
        try:
            # 1. Confirmation chime / ready signal
            await self._voice.speak(self.READY_SOUND)

            # 2. Record utterance
            log.info("voice_shell.listening")
            result = await self._voice.listen()
            text = result.get("text", "").strip()
            lang = result.get("language", "en")

            if not text:
                await self._voice.speak(self.ERROR_SOUND)
                self._state = VoiceState.IDLE
                return

            log.info("voice_shell.heard", text=text, lang=lang)

            # 3. Process
            self._state = VoiceState.PROCESSING
            await self._voice.speak(self.THINKING_SOUND)
            response = await self._process(text, lang)

            if not response:
                await self._voice.speak(self.ERROR_SOUND)
                self._state = VoiceState.IDLE
                return

            # 4. Speak response (sentence by sentence for low latency)
            self._state = VoiceState.SPEAKING
            self._interrupt_event.clear()
            sentences = self._voice.router.route(response, lang).get("sentences", [response])

            for sentence in sentences:
                if self._interrupt_event.is_set():
                    log.info("voice_shell.speech_interrupted")
                    break
                await self._voice.speak(sentence, language=lang)

        except Exception as e:
            log.error("voice_shell.error", error=str(e))
            try:
                await self._voice.speak("I encountered an error. Please try again.")
            except Exception:
                pass
        finally:
            self._state = VoiceState.IDLE

    # ── LangGraph processing ──────────────────────────────────────────────────

    async def _process(self, text: str, language: str) -> Optional[str]:
        """Send utterance through LangGraph and return the response text."""
        try:
            from orchestration.graph import run_task
            state = await run_task(text)

            result = state.get("result", {})
            verified = state.get("verified", False)

            # Extract spoken response from result
            if isinstance(result, dict):
                # Pull from step outputs
                outputs = []
                for step in result.get("results", []):
                    out = step.get("output", "")
                    if out and not str(out).startswith("Error"):
                        outputs.append(str(out))
                if outputs:
                    return " ".join(outputs)

            # Fallback: use plan summary
            plan = state.get("plan", {})
            goal = plan.get("goal", text)
            return f"I processed your request: {goal}"

        except Exception as e:
            log.error("voice_shell.process_failed", error=str(e))
            # Direct LLM fallback
            try:
                from core.llm import chat
                messages = [
                    {"role": "system", "content":
                     "You are JARVIS. Answer concisely in 1-3 sentences. "
                     f"Language: {language}"},
                    {"role": "user", "content": text},
                ]
                return await chat(messages, max_tokens=200)
            except Exception as e2:
                log.error("voice_shell.llm_fallback_failed", error=str(e2))
                return None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not cfg.voice_enabled:
            log.warning("voice_shell.disabled",
                        hint="Set JARVIS_VOICE_ENABLED=true in .env")
            return

        log.info("voice_shell.starting", wake_word=cfg.wake_word)
        self._running = True

        loop = asyncio.get_event_loop()
        detector = self._voice.attach_wake_word(self._on_wake_word)
        detector.start(loop)

        await self._voice.speak(
            f"JARVIS online. Say '{cfg.wake_word}' to activate."
        )
        log.info("voice_shell.ready")

        # Keep alive — the wake word detector runs in its own thread
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        self._running = False
        await self._voice.stop()
        log.info("voice_shell.stopped")


# ── Standalone voice loop for --voice flag in main.py ─────────────────────

async def run_voice_loop() -> None:
    shell = VoiceShell()
    try:
        await shell.start()
    finally:
        await shell.stop()
