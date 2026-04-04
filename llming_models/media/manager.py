"""MediaManager — manages media providers, mirrors LLMManager for LLMs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llming_models.media.base import STTResult, TTSResult, VoiceInfo
from llming_models.media.provider import BaseMediaProvider, MediaModelInfo
from llming_models.media.registry import MEDIA_PROVIDERS

if TYPE_CHECKING:
    from llming_models.budget.budget_manager import LLMBudgetManager
    from llming_models.credentials import LLMCredentials

logger = logging.getLogger(__name__)


class MediaManager:
    """Manages media providers — mirrors LLMManager for LLMs.

    Usage::

        manager = MediaManager()  # auto-discovers from env
        manager = MediaManager(credentials=creds, budget_manager=budget)

        # Get provider directly (typed, not string)
        provider = manager.get_provider("elevenlabs")
        result = await provider.synthesize("Hello", voice="...")

        # Or use convenience methods with budget tracking
        result = await manager.synthesize("Hello", provider="openai")
        cost = manager.estimate_tts_cost("Hello", provider="openai")
    """

    def __init__(
        self,
        *,
        credentials: LLMCredentials | None = None,
        budget_manager: LLMBudgetManager | None = None,
        provider_cascade: list[str] | None = None,
    ) -> None:
        self._credentials = credentials
        self._budget_manager = budget_manager
        self._cascade = provider_cascade or ["openai", "elevenlabs"]
        self._providers: dict[str, BaseMediaProvider] = {}
        self._auto_discover()

    # -- Discovery ------------------------------------------------------------

    def _auto_discover(self) -> None:
        """Instantiate all registered providers and keep the available ones."""
        # Import provider modules to trigger @register_media_provider decorators.
        # These must be imported *after* the registry module exists.
        from llming_models.media import openai_media as _openai  # noqa: F401
        from llming_models.media import elevenlabs_media as _elevenlabs  # noqa: F401

        for name, cls in MEDIA_PROVIDERS.items():
            creds = None
            if self._credentials:
                creds = self._credentials.for_provider(name)
            provider = cls(credentials=creds)  # type: ignore[call-arg]
            if provider.is_available:
                self._providers[name] = provider
                logger.debug("Media provider %r available", name)
            else:
                logger.debug("Media provider %r not available (no key)", name)

    # -- Provider access ------------------------------------------------------

    def get_provider(self, name: str = "") -> BaseMediaProvider:
        """Get a provider by name, or the first available from the cascade."""
        if name:
            if name not in self._providers:
                available = list(self._providers.keys())
                raise ValueError(f"Media provider {name!r} not available (available: {available})")
            return self._providers[name]
        # Return first available from cascade
        for pname in self._cascade:
            if pname in self._providers:
                return self._providers[pname]
        raise RuntimeError(
            "No media provider available — set OPENAI_API_KEY or ELEVENLABS_API_KEY"
        )

    # -- Backwards-compatible TTS/STT accessors -------------------------------

    @property
    def tts(self) -> BaseMediaProvider:
        """Return the preferred TTS provider (first in cascade)."""
        return self.get_provider()

    @property
    def stt(self) -> BaseMediaProvider:
        """Return the preferred STT provider (first in cascade)."""
        return self.get_provider()

    def get_tts(self, provider: str = "") -> BaseMediaProvider:
        """Get a specific TTS provider by name, or the preferred one."""
        return self.get_provider(provider)

    def get_stt(self, provider: str = "") -> BaseMediaProvider:
        """Get a specific STT provider by name, or the preferred one."""
        return self.get_provider(provider)

    # -- Backwards-compatible provider list accessors -------------------------

    @property
    def _tts_providers(self) -> list[BaseMediaProvider]:
        """Backwards-compatible list of TTS providers (ordered by cascade)."""
        result: list[BaseMediaProvider] = []
        for pname in self._cascade:
            if pname in self._providers:
                result.append(self._providers[pname])
        # Add any providers not in cascade
        for pname, prov in self._providers.items():
            if prov not in result:
                result.append(prov)
        return result

    @property
    def _stt_providers(self) -> list[BaseMediaProvider]:
        """Backwards-compatible list of STT providers (ordered by cascade)."""
        return self._tts_providers  # Same providers serve both TTS and STT

    # -- Convenience methods with budget tracking -----------------------------

    async def synthesize(self, text: str, *, provider: str = "", **kwargs) -> TTSResult:
        """Synthesize with optional budget tracking."""
        p = self.get_provider(provider)
        cost = p.estimate_tts_cost(text, kwargs.get("model", ""))
        if self._budget_manager and cost > 0:
            logger.debug("TTS estimated cost: %.6f", cost)
        result = await p.synthesize(text, **kwargs)
        return result

    async def transcribe(self, audio_bytes: bytes, *, provider: str = "", **kwargs) -> STTResult:
        """Transcribe with optional budget tracking."""
        p = self.get_provider(provider)
        result = await p.transcribe(audio_bytes, **kwargs)
        return result

    def estimate_tts_cost(self, text: str, *, provider: str = "", model: str = "") -> float:
        """Estimate TTS cost in currency units."""
        p = self.get_provider(provider)
        return p.estimate_tts_cost(text, model)

    def estimate_stt_cost(
        self, duration_seconds: float, *, provider: str = "", model: str = ""
    ) -> float:
        """Estimate STT cost in currency units."""
        p = self.get_provider(provider)
        return p.estimate_stt_cost(duration_seconds, model)

    # -- Model / voice listing ------------------------------------------------

    def get_all_models(self) -> list[MediaModelInfo]:
        """All TTS + STT models from all available providers."""
        models: list[MediaModelInfo] = []
        for p in self._providers.values():
            models.extend(p.get_tts_models())
            models.extend(p.get_stt_models())
        return models

    def list_all_voices(self) -> list[VoiceInfo]:
        """All voices from all available providers."""
        voices: list[VoiceInfo] = []
        for pname in self._cascade:
            if pname in self._providers:
                voices.extend(self._providers[pname].list_voices())
        # Add any providers not in cascade
        for pname, prov in self._providers.items():
            if pname not in [cn for cn in self._cascade if cn in self._providers]:
                voices.extend(prov.list_voices())
        return voices
