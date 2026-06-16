"""
Phase 4 — Voice System Tests
All audio I/O and model loading mocked — no microphone required.
"""
from __future__ import annotations

import asyncio
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch


# ── Multilingual (no external deps) ─────────────────────────────────────────

def test_detect_language_english():
    from brains.voice.multilingual import detect_language
    assert detect_language("Hello, how are you today?") == "en"


def test_detect_language_hindi_devanagari():
    from brains.voice.multilingual import detect_language
    assert detect_language("नमस्ते, आप कैसे हैं?") == "hi"


def test_detect_language_empty():
    from brains.voice.multilingual import detect_language
    assert detect_language("") == "en"


def test_normalise_strips_markdown():
    from brains.voice.multilingual import normalise_for_tts
    text = "**Hello** `world` [click here](https://example.com)"
    result = normalise_for_tts(text, "en")
    assert "**" not in result
    assert "`" not in result
    assert "click here" in result


def test_normalise_expands_abbreviations():
    from brains.voice.multilingual import normalise_for_tts
    result = normalise_for_tts("The AI uses an API", "en")
    assert "A.I." in result
    assert "A.P.I." in result


def test_normalise_caps_word_length():
    from brains.voice.multilingual import normalise_for_tts
    long_text = " ".join(["word"] * 300)
    result = normalise_for_tts(long_text, "en")
    assert len(result.split()) <= 255   # 250 words + ellipsis phrase


def test_split_into_sentences():
    from brains.voice.multilingual import split_into_sentences
    text = "Hello world. How are you? I am fine!"
    sents = split_into_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Hello world."


def test_multilingual_router_single_lang():
    from brains.voice.multilingual import MultilingualRouter
    import unittest.mock as mock
    with mock.patch("brains.voice.multilingual.cfg") as mc:
        mc.voice_language_list = ["en"]
        router = MultilingualRouter()
        assert router.get_stt_language() == "en"


def test_multilingual_router_multi_lang_returns_none():
    from brains.voice.multilingual import MultilingualRouter
    import unittest.mock as mock
    with mock.patch("brains.voice.multilingual.cfg") as mc:
        mc.voice_language_list = ["en", "hi"]
        router = MultilingualRouter()
        assert router.get_stt_language() is None   # let Whisper auto-detect


def test_multilingual_router_route_hindi():
    from brains.voice.multilingual import MultilingualRouter
    router = MultilingualRouter()
    result = router.route("नमस्ते जार्विस")
    assert result["language"] == "hi"
    assert result["tts_language"] == "hi"
    assert result["is_supported"] is True


def test_multilingual_router_route_english():
    from brains.voice.multilingual import MultilingualRouter
    router = MultilingualRouter()
    result = router.route("Hello ZEON, what is the time?")
    assert result["language"] == "en"
    assert len(result["sentences"]) >= 1


# ── VAD (mocked audio) ────────────────────────────────────────────────────────

def test_vad_is_speech_energy_english():
    from brains.voice.vad import VAD
    vad = VAD()
    vad._backend = "energy"   # force fallback
    import numpy as np
    # Loud audio = speech
    loud = (np.ones(480, dtype=np.int16) * 1000).tobytes()
    assert vad._is_speech(loud) is True
    # Quiet audio = silence
    quiet = (np.zeros(480, dtype=np.int16)).tobytes()
    assert vad._is_speech(quiet) is False


@pytest.mark.asyncio
async def test_vad_record_utterance_no_sounddevice():
    from brains.voice.vad import VAD
    vad = VAD()
    # If sounddevice missing, should return None gracefully
    with patch.dict("sys.modules", {"sounddevice": None}):
        result = await vad.record_utterance()
        # No crash — returns None or AudioSegment
        assert result is None or hasattr(result, "data")


# ── STT (mocked model) ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stt_no_backend_returns_empty():
    from brains.voice.stt import STT
    from brains.voice.vad import AudioSegment
    stt = STT.__new__(STT)
    stt._backend = "none"
    stt._lock = asyncio.Lock()

    seg = AudioSegment(data=b"\x00" * 1600, sample_rate=16000, duration_ms=100)
    result = await stt.transcribe(seg)
    assert result["text"] == ""
    assert "error" in result


@pytest.mark.asyncio
async def test_stt_faster_whisper_mocked():
    from brains.voice.stt import STT
    from brains.voice.vad import AudioSegment
    import numpy as np

    stt = STT.__new__(STT)
    stt._backend = "faster-whisper"
    stt._lock = asyncio.Lock()

    # Mock WhisperModel
    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_segment = MagicMock()
    mock_segment.text = " Hello ZEON"
    mock_model.transcribe = MagicMock(return_value=(iter([mock_segment]), mock_info))
    stt._model = mock_model

    seg = AudioSegment(data=b"\x00" * 3200, sample_rate=16000, duration_ms=100)
    result = await stt.transcribe(seg)

    assert result["text"] == "Hello ZEON"
    assert result["language"] == "en"
    assert result["confidence"] == 0.99


# ── TTS (mocked output) ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tts_no_backend_no_crash():
    from brains.voice.tts import TTS
    tts = TTS.__new__(TTS)
    tts._backend = "none"
    tts._lock = asyncio.Lock()
    # Should not raise
    await tts.speak("Hello world")


@pytest.mark.asyncio
async def test_tts_pyttsx3_mocked():
    from brains.voice.tts import TTS

    mock_engine = MagicMock()
    mock_pyttsx3 = MagicMock()
    mock_pyttsx3.init = MagicMock(return_value=mock_engine)

    with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
        tts = TTS.__new__(TTS)
        tts._backend = "pyttsx3"
        tts._lock = asyncio.Lock()
        tts._engine = mock_engine

        await tts.speak("Testing one two three")

    mock_engine.say.assert_called_once()
    mock_engine.runAndWait.assert_called_once()


@pytest.mark.asyncio
async def test_tts_speak_empty_string():
    from brains.voice.tts import TTS
    tts = TTS.__new__(TTS)
    tts._backend = "pyttsx3"
    tts._lock = asyncio.Lock()
    tts._engine = MagicMock()
    # Empty string should be a no-op
    await tts.speak("")
    tts._engine.say.assert_not_called()


# ── VoiceBrain facade ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_brain_speak_routes_language():
    from brains.voice import VoiceBrain
    brain = VoiceBrain.__new__(VoiceBrain)

    from brains.voice.multilingual import MultilingualRouter
    brain.router = MultilingualRouter()

    mock_tts = MagicMock()
    mock_tts.speak = AsyncMock()
    brain.tts = mock_tts
    brain._wake_detector = None

    await brain.speak("नमस्ते दुनिया")
    mock_tts.speak.assert_called_once()
    # Should have passed language="hi"
    call_kwargs = mock_tts.speak.call_args
    assert call_kwargs[1].get("language") == "hi" or call_kwargs[0][1] == "hi"


@pytest.mark.asyncio
async def test_voice_brain_listen_returns_dict():
    from brains.voice import VoiceBrain
    from brains.voice.vad import AudioSegment
    brain = VoiceBrain.__new__(VoiceBrain)

    from brains.voice.multilingual import MultilingualRouter
    brain.router = MultilingualRouter()

    mock_vad = MagicMock()
    seg = AudioSegment(data=b"\x00" * 3200, sample_rate=16000, duration_ms=200)
    mock_vad.record_utterance = AsyncMock(return_value=seg)
    brain.vad = mock_vad

    mock_stt = MagicMock()
    mock_stt.transcribe = AsyncMock(return_value={
        "text": "what time is it", "language": "en", "confidence": 0.9
    })
    brain.stt = mock_stt
    brain._wake_detector = None

    result = await brain.listen()
    assert result["text"] == "what time is it"
    assert result["language"] == "en"


# ── Voice Shell state machine ─────────────────────────────────────────────────

def test_voice_state_idle_on_init():
    from interfaces.voice_shell import VoiceShell, VoiceState
    with patch("interfaces.voice_shell.get_voice_brain", return_value=MagicMock()):
        with patch("interfaces.voice_shell.get_bus", return_value=MagicMock()):
            shell = VoiceShell()
    assert shell._state == VoiceState.IDLE


def test_voice_shell_wake_word_transitions_to_listening():
    from interfaces.voice_shell import VoiceShell, VoiceState
    with patch("interfaces.voice_shell.get_voice_brain", return_value=MagicMock()):
        with patch("interfaces.voice_shell.get_bus", return_value=MagicMock()):
            shell = VoiceShell()

    shell._state = VoiceState.IDLE
    with patch.object(shell, "_handle_utterance", return_value=None):
        with patch("asyncio.ensure_future"):
            shell._on_wake_word()
    assert shell._state == VoiceState.LISTENING


def test_voice_shell_wake_word_during_speech_interrupts():
    from interfaces.voice_shell import VoiceShell, VoiceState
    with patch("interfaces.voice_shell.get_voice_brain", return_value=MagicMock()):
        with patch("interfaces.voice_shell.get_bus", return_value=MagicMock()):
            shell = VoiceShell()

    shell._state = VoiceState.SPEAKING
    shell._on_wake_word()
    assert shell._state == VoiceState.INTERRUPTED
    assert shell._interrupt_event.is_set()
