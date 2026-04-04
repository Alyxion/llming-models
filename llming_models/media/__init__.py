"""Unified TTS/STT media providers.

Supports multiple backends (OpenAI, ElevenLabs) with automatic
provider selection, model info with pricing, and word-level timestamp support.
"""

from llming_models.media.base import (
    STTProvider,
    STTResult,
    TTSProvider,
    TTSResult,
    TranscriptSegment,
    VoiceInfo,
    WordTiming,
)
from llming_models.media.provider import BaseMediaProvider, MediaModelInfo, MediaType
from llming_models.media.registry import MEDIA_PROVIDERS, register_media_provider, get_media_provider
from llming_models.media.manager import MediaManager

# Import provider modules to trigger @register_media_provider decorators
from llming_models.media import openai_media as _openai_media  # noqa: F401
from llming_models.media import elevenlabs_media as _elevenlabs_media  # noqa: F401

__all__ = [
    # Data models
    "TranscriptSegment",
    "VoiceInfo",
    "TTSResult",
    "STTResult",
    "WordTiming",
    # Provider base + info
    "BaseMediaProvider",
    "MediaModelInfo",
    "MediaType",
    # Registry
    "MEDIA_PROVIDERS",
    "register_media_provider",
    "get_media_provider",
    # Manager
    "MediaManager",
    # Deprecated ABCs (backwards compat)
    "TTSProvider",
    "STTProvider",
]
