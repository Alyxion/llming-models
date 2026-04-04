"""Abstract base classes for TTS and STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A segment of a transcription with timing and type info.

    Types:
        word         — a spoken word
        spacing      — whitespace between words
        punctuation  — punctuation marks
        audio_event  — non-speech sound, e.g. "(laughter)", "(clears throat)", "(music)"
    """

    text: str
    start: float  # seconds
    end: float  # seconds
    type: str = "word"  # "word", "spacing", "punctuation", "audio_event"
    confidence: float | None = None  # log probability or confidence score


# Backwards-compatible alias
WordTiming = TranscriptSegment


class VoiceInfo(BaseModel):
    """Metadata for a TTS voice."""

    id: str
    name: str
    gender: str = ""  # "male", "female", "neutral"
    language: str = ""  # e.g. "en", "de", "multilingual"
    provider: str = ""


class TTSResult(BaseModel):
    """Result of a text-to-speech synthesis."""

    audio_bytes: bytes
    content_type: str = "audio/mpeg"
    word_timings: list[WordTiming] = Field(default_factory=list)
    duration: float | None = None  # seconds


class STTResult(BaseModel):
    """Result of a speech-to-text transcription.

    ``text`` is the clean transcription (words only).
    ``segments`` is the full detailed transcript including non-speech events,
    spacing, and punctuation with per-segment timing and confidence.
    ``word_timings`` is a filtered view containing only word-type segments
    (backwards-compatible alias).
    """

    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    confidence: float | None = None

    @property
    def word_timings(self) -> list[TranscriptSegment]:
        """Return only word-type segments (backwards compatible)."""
        return [s for s in self.segments if s.type == "word"]

    @property
    def audio_events(self) -> list[TranscriptSegment]:
        """Return non-speech audio events like (laughter), (clears throat)."""
        return [s for s in self.segments if s.type == "audio_event"]

    @property
    def rich_text(self) -> str:
        """Return text with audio events inline, e.g. 'Hello (laughter) world'."""
        parts = []
        for s in self.segments:
            if s.type in ("word", "audio_event"):
                parts.append(s.text)
            elif s.type == "spacing":
                parts.append(s.text)
            elif s.type == "punctuation":
                parts.append(s.text)
        return "".join(parts).strip()


class TTSProvider(ABC):
    """Abstract base class for text-to-speech providers.

    .. deprecated::
        Use :class:`~llming_models.media.provider.BaseMediaProvider` instead.
        This ABC is kept for backwards compatibility only.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]: ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult: ...


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers.

    .. deprecated::
        Use :class:`~llming_models.media.provider.BaseMediaProvider` instead.
        This ABC is kept for backwards compatibility only.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str = "",
        with_timings: bool = False,
    ) -> STTResult: ...
