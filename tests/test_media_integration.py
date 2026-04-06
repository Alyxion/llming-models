"""Integration tests for TTS/STT media providers.

These tests hit real APIs (OpenAI direct or Azure OpenAI) and require
at least one of OPENAI_API_KEY or AZURE_OPENAI_API_KEY to be set.

Run with: pytest tests/test_media_integration.py -v
"""
from __future__ import annotations

import os

import pytest

from llming_models.media import MediaManager, OpenAIMediaProvider

# Skip entire module if no API key is available
_has_openai = bool(os.environ.get("OPENAI_API_KEY"))
_has_azure = bool(os.environ.get("AZURE_OPENAI_API_KEY")) and bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))
pytestmark = pytest.mark.skipif(
    not (_has_openai or _has_azure),
    reason="No OPENAI_API_KEY or AZURE_OPENAI_API_KEY+ENDPOINT set",
)


@pytest.fixture
def provider() -> OpenAIMediaProvider:
    return OpenAIMediaProvider()


@pytest.fixture
def manager() -> MediaManager:
    return MediaManager()


class TestOpenAIMediaProviderInit:
    def test_is_available(self, provider: OpenAIMediaProvider):
        assert provider.is_available

    def test_azure_detection(self, provider: OpenAIMediaProvider):
        if _has_openai:
            assert not provider.is_azure
        elif _has_azure:
            assert provider.is_azure

    def test_tts_model_set(self, provider: OpenAIMediaProvider):
        if provider.is_azure:
            assert provider.tts_model == os.environ.get("AZURE_TTS_DEPLOYMENT", "tts-hd")
        else:
            assert provider.tts_model == "gpt-4o-mini-tts"

    def test_stt_model_set(self, provider: OpenAIMediaProvider):
        if provider.is_azure:
            assert provider.stt_model == os.environ.get("AZURE_STT_DEPLOYMENT", "gpt-4o-transcribe")
        else:
            assert provider.stt_model == "gpt-4o-transcribe"

    def test_voices(self, provider: OpenAIMediaProvider):
        voices = provider.list_voices()
        assert len(voices) > 0
        ids = [v.id for v in voices]
        assert "nova" in ids
        assert "cedar" in ids

    def test_get_tts_models(self, provider: OpenAIMediaProvider):
        models = provider.get_tts_models()
        assert len(models) >= 1
        assert models[0].name == provider.tts_model

    def test_get_stt_models(self, provider: OpenAIMediaProvider):
        models = provider.get_stt_models()
        assert len(models) >= 1
        assert models[0].name == provider.stt_model


class TestTTS:
    async def test_synthesize_basic(self, provider: OpenAIMediaProvider):
        result = await provider.synthesize("Hello, this is a test.", voice="nova")
        assert len(result.audio_bytes) > 1000
        assert result.content_type == "audio/mpeg"

    async def test_synthesize_german(self, provider: OpenAIMediaProvider):
        result = await provider.synthesize("Hallo, wie geht es dir?", voice="nova", language="de-de")
        assert len(result.audio_bytes) > 1000

    async def test_synthesize_with_timings(self, provider: OpenAIMediaProvider):
        result = await provider.synthesize(
            "Hello world, this is a timing test.",
            voice="nova",
            with_timings=True,
        )
        assert len(result.audio_bytes) > 1000
        if not provider.is_azure:
            assert len(result.word_timings) > 0
        else:
            # Azure tts-hd doesn't support word timings
            assert result.word_timings == []

    async def test_synthesize_empty_text(self, provider: OpenAIMediaProvider):
        """Empty text should still return valid audio (silence or error)."""
        with pytest.raises(Exception):
            await provider.synthesize("", voice="nova")


class TestSTT:
    async def test_transcribe_from_tts_roundtrip(self, provider: OpenAIMediaProvider):
        """Generate audio via TTS, then transcribe it back."""
        tts_result = await provider.synthesize("The quick brown fox jumps over the lazy dog.", voice="nova")
        assert len(tts_result.audio_bytes) > 1000

        stt_result = await provider.transcribe(
            tts_result.audio_bytes,
            filename="test.mp3",
            content_type="audio/mpeg",
            language="en",
        )
        text = stt_result.text.lower()
        assert "fox" in text or "dog" in text

    async def test_transcribe_too_small(self, provider: OpenAIMediaProvider):
        result = await provider.transcribe(b"tiny", filename="tiny.webm")
        assert result.text == ""


class TestMediaManager:
    def test_provider_available(self, manager: MediaManager):
        provider = manager.get_provider("openai")
        assert provider.is_available

    async def test_manager_synthesize(self, manager: MediaManager):
        result = await manager.synthesize("Testing the manager.", provider="openai", voice="nova")
        assert len(result.audio_bytes) > 1000

    async def test_manager_transcribe_roundtrip(self, manager: MediaManager):
        tts = await manager.synthesize("Integration test audio.", provider="openai", voice="nova")
        stt = await manager.transcribe(
            tts.audio_bytes,
            provider="openai",
            filename="test.mp3",
            content_type="audio/mpeg",
            language="en",
        )
        assert len(stt.text) > 0

    def test_all_models(self, manager: MediaManager):
        models = manager.get_all_models()
        assert len(models) >= 2  # at least TTS + STT

    def test_all_voices(self, manager: MediaManager):
        voices = manager.list_all_voices()
        assert len(voices) > 0

    def test_estimate_tts_cost(self, manager: MediaManager):
        cost = manager.estimate_tts_cost("Hello world", provider="openai")
        assert cost > 0
