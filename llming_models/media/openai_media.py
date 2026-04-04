"""OpenAI media provider — TTS and STT via a single BaseMediaProvider."""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

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


@register_media_provider("openai")
class OpenAIMediaProvider(BaseMediaProvider):
    """OpenAI TTS (gpt-4o-mini-tts) and STT (gpt-4o-transcribe / whisper-1)."""

    TTS_MODEL = "gpt-4o-mini-tts"
    STT_MODEL = "gpt-4o-transcribe"
    WHISPER_MODEL = "whisper-1"

    VOICES = [
        VoiceInfo(id="cedar", name="Cedar", gender="male", provider="openai"),
        VoiceInfo(id="marin", name="Marin", gender="female", provider="openai"),
        VoiceInfo(id="ash", name="Ash", gender="male", provider="openai"),
        VoiceInfo(id="ballad", name="Ballad", gender="male", provider="openai"),
        VoiceInfo(id="coral", name="Coral", gender="female", provider="openai"),
        VoiceInfo(id="echo", name="Echo", gender="male", provider="openai"),
        VoiceInfo(id="fable", name="Fable", gender="male", provider="openai"),
        VoiceInfo(id="nova", name="Nova", gender="female", provider="openai"),
        VoiceInfo(id="onyx", name="Onyx", gender="male", provider="openai"),
        VoiceInfo(id="sage", name="Sage", gender="female", provider="openai"),
        VoiceInfo(id="shimmer", name="Shimmer", gender="female", provider="openai"),
        VoiceInfo(id="verse", name="Verse", gender="female", provider="openai"),
        VoiceInfo(id="alloy", name="Alloy", gender="neutral", provider="openai"),
    ]

    # Locale code -> language name for TTS instructions
    _LOCALE_NAMES: dict[str, str] = {
        "en": "English",
        "en-us": "English",
        "en-in": "English",
        "de": "German",
        "de-de": "German",
        "de-swg": "Swabian German",
        "fr": "French",
        "fr-fr": "French",
        "it": "Italian",
        "it-it": "Italian",
        "hi": "Hindi",
        "hi-in": "Hindi",
    }

    # Models that support the 'instructions' parameter
    _INSTRUCTION_MODELS = {"gpt-4o-mini-tts"}

    def __init__(
        self,
        api_key: str | None = None,
        *,
        credentials: ProviderCredentials | None = None,
    ):
        super().__init__(name="openai", label="OpenAI", credentials=credentials)
        self._api_key = api_key or self._resolve_api_key()

    def _resolve_api_key(self) -> str:
        if self._credentials and self._credentials.api_key.get_secret_value():
            return self._credentials.api_key.get_secret_value()
        return os.environ.get("OPENAI_API_KEY", "")

    # -- BaseMediaProvider interface ------------------------------------------

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_tts_models(self) -> list[MediaModelInfo]:
        return [
            MediaModelInfo(
                provider="openai",
                name="gpt-4o-mini-tts",
                label="OpenAI TTS (gpt-4o-mini-tts)",
                media_type=MediaType.TTS,
                price_per_1m_chars=15.0,  # $15 / 1M chars
                supports_word_timings=True,  # via whisper round-trip
                supports_streaming=True,
                languages=["en", "de", "fr", "it", "hi", "es", "pt", "ja", "ko", "zh"],
                max_chars=4096,
                speed=8,
                quality=8,
            ),
        ]

    def get_stt_models(self) -> list[MediaModelInfo]:
        return [
            MediaModelInfo(
                provider="openai",
                name="gpt-4o-transcribe",
                label="OpenAI STT (gpt-4o-transcribe)",
                media_type=MediaType.STT,
                price_per_minute=0.006,  # $0.006 / min
                supports_word_timings=False,
                languages=["en", "de", "fr", "it", "hi", "es", "pt", "ja", "ko", "zh"],
                speed=8,
                quality=9,
            ),
            MediaModelInfo(
                provider="openai",
                name="whisper-1",
                label="OpenAI STT (whisper-1)",
                media_type=MediaType.STT,
                price_per_minute=0.006,  # $0.006 / min
                supports_word_timings=True,
                languages=["en", "de", "fr", "it", "hi", "es", "pt", "ja", "ko", "zh"],
                speed=7,
                quality=7,
            ),
        ]

    def list_voices(self) -> list[VoiceInfo]:
        return list(self.VOICES)

    async def synthesize(
        self,
        text: str,
        *,
        model: str = "",
        voice: str = "cedar",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult:
        """Synthesize text to speech via OpenAI.

        If *with_timings* is True, the generated audio is transcribed back
        through whisper-1 to obtain word-level timestamps.
        """
        client = AsyncOpenAI(api_key=self._api_key)
        tts_model = model or self.TTS_MODEL
        kwargs = self._build_tts_kwargs(text, voice, language, tts_model)

        tts_response = await client.audio.speech.create(**kwargs)
        mp3_bytes: bytes = tts_response.content

        word_timings: list[WordTiming] = []
        if with_timings:
            audio_file = io.BytesIO(mp3_bytes)
            audio_file.name = "speech.mp3"
            stt_result = await client.audio.transcriptions.create(
                model=self.WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
            word_timings = [
                WordTiming(text=w.word, start=w.start, end=w.end)
                for w in (stt_result.words or [])
            ]

        return TTSResult(
            audio_bytes=mp3_bytes,
            content_type="audio/mpeg",
            word_timings=word_timings,
        )

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
        """Transcribe audio to text via OpenAI.

        If *with_timings* is True, whisper-1 is used with verbose_json and
        word-level timestamp granularities.
        """
        if len(audio_bytes) < 100:
            logger.warning("[STT] Audio too small (%d bytes), skipping", len(audio_bytes))
            return STTResult(text="")

        client = AsyncOpenAI(api_key=self._api_key)

        if with_timings:
            return await self._transcribe_with_timings(client, audio_bytes, filename, language)

        return await self._transcribe_plain(client, audio_bytes, filename, language, model)

    # -- Backwards-compatible aliases -----------------------------------------

    @property
    def provider_name(self) -> str:
        return self.name

    # -- Private helpers ------------------------------------------------------

    def _build_tts_kwargs(self, text: str, voice: str, language: str, tts_model: str = "") -> dict:
        """Build kwargs for the OpenAI TTS API call."""
        used_model = tts_model or self.TTS_MODEL
        kwargs: dict = {
            "model": used_model,
            "voice": voice or "cedar",
            "input": text,
            "response_format": "mp3",
        }
        if used_model in self._INSTRUCTION_MODELS:
            lang_name = self._LOCALE_NAMES.get(language, "")
            if lang_name:
                kwargs["instructions"] = f"Speak in {lang_name}."
        return kwargs

    async def _transcribe_plain(
        self,
        client: AsyncOpenAI,
        audio_bytes: bytes,
        filename: str,
        language: str,
        model: str = "",
    ) -> STTResult:
        """Standard transcription without word timings."""
        stt_model = model or self.STT_MODEL
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        kwargs: dict = {
            "model": stt_model,
            "file": audio_file,
            "response_format": "json",
        }
        if language:
            kwargs["language"] = language

        try:
            result = await client.audio.transcriptions.create(**kwargs)
            return STTResult(text=result.text)
        except Exception as exc:
            err_str = str(exc).lower()
            format_errors = (
                "corrupted",
                "unsupported",
                "invalid file format",
                "invalid_file_format",
                "could not process",
            )
            if any(s in err_str for s in format_errors):
                logger.warning(
                    "[STT] %s format error (%s), retrying with %s",
                    stt_model,
                    exc,
                    self.WHISPER_MODEL,
                )
                audio_file.seek(0)
                kwargs["model"] = self.WHISPER_MODEL
                try:
                    result = await client.audio.transcriptions.create(**kwargs)
                    return STTResult(text=result.text)
                except Exception as exc2:
                    logger.warning(
                        "[STT] Fallback also failed (%s), returning empty", exc2
                    )
                    return STTResult(text="")
            logger.error("[STT] Transcription failed: %s", exc)
            raise

    async def _transcribe_with_timings(
        self,
        client: AsyncOpenAI,
        audio_bytes: bytes,
        filename: str,
        language: str,
    ) -> STTResult:
        """Transcription with word-level timestamps via whisper-1."""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        kwargs: dict = {
            "model": self.WHISPER_MODEL,
            "file": audio_file,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            kwargs["language"] = language

        result = await client.audio.transcriptions.create(**kwargs)
        segments = [
            TranscriptSegment(text=w.word, start=w.start, end=w.end, type="word")
            for w in (result.words or [])
        ]
        return STTResult(
            text=result.text,
            language=language or None,
            segments=segments,
        )


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------

class OpenAITTSProvider(OpenAIMediaProvider):
    """Deprecated — use ``OpenAIMediaProvider`` or ``get_media_provider('openai')``."""

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)


class OpenAISTTProvider(OpenAIMediaProvider):
    """Deprecated — use ``OpenAIMediaProvider`` or ``get_media_provider('openai')``."""

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
