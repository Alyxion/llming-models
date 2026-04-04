"""ElevenLabs media provider — TTS and STT via a single BaseMediaProvider."""

from __future__ import annotations

import base64
import logging
import os
from typing import TYPE_CHECKING

import httpx

from llming_models.media.base import (
    STTResult,
    TTSResult,
    TranscriptSegment,
    VoiceInfo,
    WordTiming,
)
from llming_models.media.provider import BaseMediaProvider, MediaModelInfo, MediaType
from llming_models.media.registry import register_media_provider

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.elevenlabs.io/v1"


def _aggregate_character_timings(
    characters: list[str],
    character_start_times: list[float],
    character_end_times: list[float],
) -> list[WordTiming]:
    """Aggregate character-level alignment data into word-level timings.

    ElevenLabs returns per-character timestamps. We split on spaces to
    reconstruct words with their start/end boundaries.
    """
    if not characters:
        return []

    words: list[WordTiming] = []
    current_word_chars: list[str] = []
    word_start: float | None = None
    word_end: float = 0.0

    for char, start, end in zip(characters, character_start_times, character_end_times):
        if char == " ":
            # Space terminates the current word
            if current_word_chars:
                words.append(
                    WordTiming(
                        text="".join(current_word_chars),
                        start=word_start or 0.0,
                        end=word_end,
                    )
                )
                current_word_chars = []
                word_start = None
        else:
            if word_start is None:
                word_start = start
            current_word_chars.append(char)
            word_end = end

    # Flush the last word
    if current_word_chars:
        words.append(
            WordTiming(
                text="".join(current_word_chars),
                start=word_start or 0.0,
                end=word_end,
            )
        )

    return words


@register_media_provider("elevenlabs")
class ElevenLabsMediaProvider(BaseMediaProvider):
    """ElevenLabs TTS (eleven_v3 / eleven_flash_v2_5) and STT (scribe_v1)."""

    # -- Male voices (10) ---------------------------------------------------
    VOICES_MALE: list[VoiceInfo] = [
        VoiceInfo(id="Gfpl8Yo74Is0W6cPUWWT", name="Max", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="JBFqnCBsd6RMkjVDRZzb", name="George", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="nPczCjzI2devNBz1zQrb", name="Brian", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="onwK4e9ZLuTAKqWW03F9", name="Daniel", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="cjVigY5qzO86Huf0OWal", name="Eric", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="pNInz6obpgDQGcFmaJgB", name="Adam", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="iP95p4xoKVk53GoZ742B", name="Chris", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="rDmv3mOhK6TnhYWckFaD", name="Felix Serenitas", gender="male", language="de", provider="elevenlabs"),
        VoiceInfo(id="kmSVBPu7loj4ayNinwWM", name="Archie", gender="male", language="en", provider="elevenlabs"),
        VoiceInfo(id="TX3LPaxmHKxFdv7VOQHJ", name="Liam", gender="male", language="en", provider="elevenlabs"),
    ]

    # -- Female voices (10) -------------------------------------------------
    VOICES_FEMALE: list[VoiceInfo] = [
        VoiceInfo(id="AnvlJBAqSLDzEevYr9Ap", name="Ava", gender="female", language="de", provider="elevenlabs"),
        VoiceInfo(id="EXAVITQu4vr4xnSDxMaL", name="Sarah", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="Xb7hH8MSUJpSbSDYk0k2", name="Alice", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="XrExE9yKIg1WjnnlVkGX", name="Matilda", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="cgSgspJ2msm6clMCkdW9", name="Jessica", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="hpp4J3VqNfWAUOO0d1Us", name="Bella", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="pFZP5JQG7iQjIQuC4Bku", name="Lily", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="FGY2WhTYpPnrIDTdsKH5", name="Laura", gender="female", language="en", provider="elevenlabs"),
        VoiceInfo(id="uvysWDLbKpA4XvpD3GI6", name="Leonie", gender="female", language="de", provider="elevenlabs"),
        VoiceInfo(id="McVZB9hVxVSk3Equu8EH", name="Audrey", gender="female", language="fr", provider="elevenlabs"),
    ]

    # Combined list for list_voices()
    VOICES: list[VoiceInfo] = VOICES_MALE + VOICES_FEMALE

    MODELS = [
        "eleven_v3",
        "eleven_multilingual_v2",
        "eleven_turbo_v2_5",
        "eleven_flash_v2_5",
    ]

    # Default voice per language prefix
    _LANGUAGE_DEFAULTS: dict[str, str] = {
        "de": "AnvlJBAqSLDzEevYr9Ap",  # Ava
        "en": "Gfpl8Yo74Is0W6cPUWWT",  # Max
        "fr": "McVZB9hVxVSk3Equu8EH",  # Audrey
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "eleven_v3",
        *,
        credentials: ProviderCredentials | None = None,
    ):
        super().__init__(name="elevenlabs", label="ElevenLabs", credentials=credentials)
        self._api_key = api_key or self._resolve_api_key()
        self._model = model

    def _resolve_api_key(self) -> str:
        if self._credentials and self._credentials.api_key.get_secret_value():
            return self._credentials.api_key.get_secret_value()
        return os.environ.get("ELEVENLABS_API_KEY", "")

    # -- BaseMediaProvider interface ------------------------------------------

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_tts_models(self) -> list[MediaModelInfo]:
        return [
            MediaModelInfo(
                provider="elevenlabs",
                name="eleven_v3",
                label="ElevenLabs TTS (eleven_v3)",
                media_type=MediaType.TTS,
                price_per_1m_chars=300.0,  # $0.30/1K chars = $300/1M chars
                supports_word_timings=True,
                supports_streaming=True,
                languages=["en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh", "pl", "hi", "ar"],
                max_chars=5000,
                speed=7,
                quality=9,
            ),
            MediaModelInfo(
                provider="elevenlabs",
                name="eleven_multilingual_v2",
                label="ElevenLabs TTS (eleven_multilingual_v2)",
                media_type=MediaType.TTS,
                price_per_1m_chars=300.0,
                supports_word_timings=True,
                supports_streaming=True,
                languages=["en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh", "pl", "hi", "ar"],
                max_chars=5000,
                speed=6,
                quality=8,
            ),
            MediaModelInfo(
                provider="elevenlabs",
                name="eleven_turbo_v2_5",
                label="ElevenLabs TTS (eleven_turbo_v2_5)",
                media_type=MediaType.TTS,
                price_per_1m_chars=150.0,  # roughly half of v3
                supports_word_timings=True,
                supports_streaming=True,
                languages=["en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh"],
                max_chars=5000,
                speed=9,
                quality=7,
            ),
            MediaModelInfo(
                provider="elevenlabs",
                name="eleven_flash_v2_5",
                label="ElevenLabs TTS (eleven_flash_v2_5)",
                media_type=MediaType.TTS,
                price_per_1m_chars=75.0,  # $0.075/1K chars = $75/1M chars
                supports_word_timings=True,
                supports_streaming=True,
                languages=["en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh"],
                max_chars=5000,
                speed=10,
                quality=6,
            ),
        ]

    def get_stt_models(self) -> list[MediaModelInfo]:
        return [
            MediaModelInfo(
                provider="elevenlabs",
                name="scribe_v1",
                label="ElevenLabs STT (scribe_v1)",
                media_type=MediaType.STT,
                price_per_minute=0.0,  # Free in most plans
                supports_word_timings=True,
                languages=["en", "de", "fr", "es", "it", "pt", "ja", "ko", "zh", "pl", "hi", "ar"],
                speed=7,
                quality=8,
            ),
        ]

    def list_voices(self) -> list[VoiceInfo]:
        return list(self.VOICES)

    async def synthesize(
        self,
        text: str,
        *,
        model: str = "",
        voice: str = "",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult:
        """Synthesize text via ElevenLabs API.

        When *with_timings* is True, uses the ``/with-timestamps`` endpoint
        to retrieve character-level alignment and aggregates it into words.
        """
        voice_id = self._resolve_voice(voice, language)
        tts_model = model or self._model
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        body: dict = {
            "text": text,
            "model_id": tts_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }
        if language:
            body["language_code"] = language

        output_format = "mp3_44100_128"

        if with_timings:
            return await self._synthesize_with_timings(
                voice_id, headers, body, output_format
            )

        return await self._synthesize_plain(voice_id, headers, body, output_format)

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        model: str = "",
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str = "",
        with_timings: bool = False,
    ) -> STTResult:
        """Transcribe audio via ElevenLabs Scribe API."""
        headers = {"xi-api-key": self._api_key}
        files = {"file": (filename, audio_bytes, content_type)}
        data: dict[str, str] = {"model_id": model or "scribe_v1"}
        if language:
            data["language_code"] = language[:2]

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers=headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()
            result = resp.json()

        # Capture ALL segment types — words, spacing, punctuation, audio events
        segments: list[TranscriptSegment] = []
        if "words" in result:
            for w in result["words"]:
                segments.append(TranscriptSegment(
                    text=w["text"],
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    type=w.get("type", "word"),
                    confidence=w.get("logprob"),
                ))

        return STTResult(
            text=result.get("text", ""),
            language=result.get("language_code", language) or None,
            segments=segments,
            confidence=result.get("language_probability"),
        )

    # -- Backwards-compatible aliases -----------------------------------------

    @property
    def provider_name(self) -> str:
        return self.name

    # -- Private helpers ------------------------------------------------------

    def _resolve_voice(self, voice: str, language: str) -> str:
        """Pick a voice ID — explicit, by language, or fallback to Max."""
        if voice:
            return voice
        # Try language-based default
        lang_prefix = language.split("-")[0].lower() if language else ""
        if lang_prefix in self._LANGUAGE_DEFAULTS:
            return self._LANGUAGE_DEFAULTS[lang_prefix]
        # Fallback: Max (English)
        return self.VOICES[0].id

    async def _synthesize_plain(
        self,
        voice_id: str,
        headers: dict,
        body: dict,
        output_format: str,
    ) -> TTSResult:
        """Standard synthesis returning raw audio bytes."""
        url = f"{_BASE_URL}/text-to-speech/{voice_id}?output_format={output_format}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return TTSResult(
                audio_bytes=resp.content,
                content_type="audio/mpeg",
            )

    async def _synthesize_with_timings(
        self,
        voice_id: str,
        headers: dict,
        body: dict,
        output_format: str,
    ) -> TTSResult:
        """Synthesis via /with-timestamps endpoint for character alignment."""
        url = (
            f"{_BASE_URL}/text-to-speech/{voice_id}"
            f"/with-timestamps?output_format={output_format}"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        audio_b64: str = data.get("audio_base64", "")
        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

        alignment = data.get("alignment", {})
        characters: list[str] = alignment.get("characters", [])
        char_starts: list[float] = alignment.get("character_start_times_seconds", [])
        char_ends: list[float] = alignment.get("character_end_times_seconds", [])

        word_timings = _aggregate_character_timings(characters, char_starts, char_ends)

        return TTSResult(
            audio_bytes=audio_bytes,
            content_type="audio/mpeg",
            word_timings=word_timings,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

class ElevenLabsTTSProvider(ElevenLabsMediaProvider):
    """Deprecated — use ``ElevenLabsMediaProvider`` or ``get_media_provider('elevenlabs')``."""

    def __init__(self, api_key: str | None = None, model: str = "eleven_v3", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)


class ElevenLabsSTTProvider(ElevenLabsMediaProvider):
    """Deprecated — use ``ElevenLabsMediaProvider`` or ``get_media_provider('elevenlabs')``."""

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
