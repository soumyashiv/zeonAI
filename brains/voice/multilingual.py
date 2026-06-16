"""
ZEON Voice — Multilingual Support
Detects language of transcribed text, normalises, routes to correct TTS voice.
Supported: EN (English), HI (Hindi). Extensible to any language.
"""
from __future__ import annotations

import re
from typing import Optional

import structlog

from core.config import get_config

log = structlog.get_logger(__name__)
cfg = get_config()

# Hindi Unicode range: \u0900-\u097F (Devanagari script)
_HINDI_RE = re.compile(r"[\u0900-\u097F]")

# Supported language codes
SUPPORTED_LANGS = {"en", "hi"}


def detect_language(text: str) -> str:
    """
    Fast heuristic language detection.
    Falls back to langdetect for ambiguous Latin-script text.
    """
    if not text.strip():
        return "en"

    # Devanagari present → Hindi
    if _HINDI_RE.search(text):
        return "hi"

    # Try langdetect for ambiguous text
    try:
        from langdetect import detect
        lang = detect(text)
        # Map to supported languages
        if lang in SUPPORTED_LANGS:
            return lang
        # Map common variants
        if lang.startswith("hi"):
            return "hi"
        return "en"
    except Exception:
        pass

    return "en"


def normalise_for_tts(text: str, language: str) -> str:
    """
    Clean text before passing to TTS.
    - Remove markdown formatting
    - Expand common abbreviations
    - Limit length for natural speech
    """
    # Strip markdown
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "code block", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Expand common abbreviations (EN)
    if language == "en":
        abbrevs = {
            "AI": "A.I.",
            "LLM": "L.L.M.",
            "CPU": "C.P.U.",
            "GPU": "G.P.U.",
            "API": "A.P.I.",
            "e.g.": "for example",
            "i.e.": "that is",
            "etc.": "and so on",
            "vs.": "versus",
        }
        for short, expanded in abbrevs.items():
            text = text.replace(short, expanded)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Cap at ~250 words to keep TTS snappy
    words = text.split()
    if len(words) > 250:
        text = " ".join(words[:250]) + "… and more."

    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Split long response into sentences for streaming TTS
    (speak first sentence while generating rest).
    """
    # Simple split on .!? followed by space/end
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


class MultilingualRouter:
    """Routes text to correct STT language and TTS voice."""

    def __init__(self) -> None:
        self._configured_langs = cfg.voice_language_list   # e.g. ["en", "hi"]

    def get_stt_language(self) -> Optional[str]:
        """
        If only one language configured, pass it to Whisper.
        Otherwise pass None (auto-detect).
        """
        if len(self._configured_langs) == 1:
            return self._configured_langs[0]
        return None   # Whisper auto-detect

    def route(self, text: str, transcription_lang: Optional[str] = None) -> dict:
        """
        Given transcribed text, return routing info.
        Returns: {language, tts_language, is_supported, normalised_text}
        """
        lang = transcription_lang or detect_language(text)
        if lang not in SUPPORTED_LANGS:
            lang = "en"   # graceful fallback

        normalised = normalise_for_tts(text, lang)
        sentences = split_into_sentences(normalised)

        return {
            "language": lang,
            "tts_language": lang,
            "is_supported": lang in SUPPORTED_LANGS,
            "normalised_text": normalised,
            "sentences": sentences,
        }
