"""Base media provider and model info — mirrors BaseProvider / LLMInfo for TTS/STT/image."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from llming_models.media.base import STTResult, TTSResult, VoiceInfo

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials


class MediaType(str, Enum):
    """Type of media service a model provides."""
    TTS = "tts"
    STT = "stt"
    IMAGE = "image"


@dataclass
class MediaModelInfo:
    """Model info for a media service — mirrors LLMInfo for TTS/STT/image."""
    provider: str              # "openai", "elevenlabs"
    name: str                  # "gpt-4o-mini-tts", "eleven_v3"
    label: str                 # "OpenAI TTS (gpt-4o-mini-tts)"
    media_type: MediaType      # TTS, STT, IMAGE
    # Pricing
    price_per_1m_chars: float = 0.0     # TTS: cost per 1M characters
    price_per_minute: float = 0.0       # STT: cost per minute of audio
    price_per_image: float = 0.0        # Image: cost per image
    # Capabilities
    supports_word_timings: bool = False
    supports_streaming: bool = False
    languages: list[str] = field(default_factory=list)  # ["en", "de", "fr", ...]
    max_chars: int = 5000              # Max input chars per request
    # UI hints
    speed: int = 5                      # 1-10
    quality: int = 5                    # 1-10


class BaseMediaProvider(ABC):
    """Abstract base for media providers — mirrors BaseProvider for LLMs."""

    def __init__(self, name: str, label: str, *, credentials: ProviderCredentials | None = None):
        self.name = name
        self.label = label
        self._credentials = credentials

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is usable (has valid API key)."""
        ...

    @abstractmethod
    def get_tts_models(self) -> list[MediaModelInfo]:
        """Return TTS model descriptors supported by this provider."""
        ...

    @abstractmethod
    def get_stt_models(self) -> list[MediaModelInfo]:
        """Return STT model descriptors supported by this provider."""
        ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """Return available TTS voices."""
        ...

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        model: str = "",
        voice: str = "",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult:
        """Synthesize *text* into audio."""
        ...

    @abstractmethod
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
        """Transcribe *audio_bytes* to text."""
        ...

    # -- Cost helpers ---------------------------------------------------------

    def estimate_tts_cost(self, text: str, model: str = "") -> float:
        """Estimate cost for TTS in currency units."""
        models = self.get_tts_models()
        m = next((m for m in models if m.name == model), models[0] if models else None)
        if not m:
            return 0.0
        return len(text) * m.price_per_1m_chars / 1_000_000

    def estimate_stt_cost(self, duration_seconds: float, model: str = "") -> float:
        """Estimate cost for STT in currency units."""
        models = self.get_stt_models()
        m = next((m for m in models if m.name == model), models[0] if models else None)
        if not m:
            return 0.0
        return (duration_seconds / 60) * m.price_per_minute
