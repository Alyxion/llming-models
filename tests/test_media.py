"""Tests for the TTS/STT media provider system."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from llming_models.media.base import STTResult, TTSResult, VoiceInfo, WordTiming
from llming_models.media.provider import BaseMediaProvider, MediaModelInfo, MediaType
from llming_models.media.registry import MEDIA_PROVIDERS, get_media_provider
from llming_models.media.elevenlabs_media import (
    ElevenLabsMediaProvider,
    ElevenLabsSTTProvider,
    ElevenLabsTTSProvider,
    _aggregate_character_timings,
)
from llming_models.media.manager import MediaManager
from llming_models.media.openai_media import (
    OpenAIMediaProvider,
    OpenAISTTProvider,
    OpenAITTSProvider,
)


# ---------------------------------------------------------------------------
# Model / dataclass tests
# ---------------------------------------------------------------------------


class TestWordTiming:
    def test_basic(self):
        wt = WordTiming(text="hello", start=0.0, end=0.5)
        assert wt.text == "hello"
        assert wt.start == 0.0
        assert wt.end == 0.5

    def test_serialization(self):
        wt = WordTiming(text="test", start=1.2, end=1.8)
        data = wt.model_dump()
        assert data["text"] == "test"
        assert data["start"] == 1.2
        assert data["end"] == 1.8
        assert data["type"] == "word"


class TestVoiceInfo:
    def test_defaults(self):
        vi = VoiceInfo(id="v1", name="Voice One")
        assert vi.gender == ""
        assert vi.language == ""
        assert vi.provider == ""

    def test_full(self):
        vi = VoiceInfo(
            id="abc", name="Alice", gender="female", language="en", provider="openai"
        )
        assert vi.id == "abc"
        assert vi.name == "Alice"
        assert vi.gender == "female"


class TestTTSResult:
    def test_defaults(self):
        result = TTSResult(audio_bytes=b"\x00\x01")
        assert result.content_type == "audio/mpeg"
        assert result.word_timings == []
        assert result.duration is None

    def test_with_timings(self):
        timings = [WordTiming(text="hi", start=0.0, end=0.3)]
        result = TTSResult(audio_bytes=b"audio", word_timings=timings, duration=0.3)
        assert len(result.word_timings) == 1
        assert result.duration == 0.3


class TestSTTResult:
    def test_defaults(self):
        result = STTResult(text="hello world")
        assert result.language is None
        assert result.word_timings == []
        assert result.confidence is None

    def test_full(self):
        result = STTResult(
            text="hello",
            language="en",
            segments=[WordTiming(text="hello", start=0.0, end=0.5)],
            confidence=0.99,
        )
        assert result.confidence == 0.99
        assert result.language == "en"


# ---------------------------------------------------------------------------
# MediaModelInfo tests
# ---------------------------------------------------------------------------


class TestMediaModelInfo:
    def test_tts_model(self):
        info = MediaModelInfo(
            provider="openai",
            name="gpt-4o-mini-tts",
            label="OpenAI TTS",
            media_type=MediaType.TTS,
            price_per_1m_chars=15.0,
        )
        assert info.media_type == MediaType.TTS
        assert info.price_per_1m_chars == 15.0
        assert info.price_per_minute == 0.0

    def test_stt_model(self):
        info = MediaModelInfo(
            provider="openai",
            name="whisper-1",
            label="OpenAI STT",
            media_type=MediaType.STT,
            price_per_minute=0.006,
        )
        assert info.media_type == MediaType.STT
        assert info.price_per_minute == 0.006

    def test_defaults(self):
        info = MediaModelInfo(
            provider="test", name="test", label="Test", media_type=MediaType.TTS,
        )
        assert info.supports_word_timings is False
        assert info.supports_streaming is False
        assert info.languages == []
        assert info.max_chars == 5000
        assert info.speed == 5
        assert info.quality == 5


# ---------------------------------------------------------------------------
# Provider registration tests
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_openai_registered(self):
        assert "openai" in MEDIA_PROVIDERS
        assert MEDIA_PROVIDERS["openai"] is OpenAIMediaProvider

    def test_elevenlabs_registered(self):
        assert "elevenlabs" in MEDIA_PROVIDERS
        assert MEDIA_PROVIDERS["elevenlabs"] is ElevenLabsMediaProvider

    def test_get_media_provider(self):
        cls = get_media_provider("openai")
        assert cls is OpenAIMediaProvider

    def test_get_media_provider_unknown(self):
        with pytest.raises(ValueError, match="not found"):
            get_media_provider("nonexistent")


# ---------------------------------------------------------------------------
# Cost estimation tests
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_openai_tts_cost(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        # 1000 chars at $15/1M chars = $0.015
        cost = provider.estimate_tts_cost("x" * 1000)
        assert abs(cost - 0.015) < 0.0001

    def test_openai_stt_cost(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        # 60 seconds at $0.006/min = $0.006
        cost = provider.estimate_stt_cost(60.0)
        assert abs(cost - 0.006) < 0.0001

    def test_elevenlabs_tts_cost_v3(self):
        provider = ElevenLabsMediaProvider(api_key="test-key", model="eleven_v3")
        # 1000 chars at $300/1M = $0.30
        cost = provider.estimate_tts_cost("x" * 1000, "eleven_v3")
        assert abs(cost - 0.30) < 0.001

    def test_elevenlabs_tts_cost_flash(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        # 1000 chars at $75/1M = $0.075
        cost = provider.estimate_tts_cost("x" * 1000, "eleven_flash_v2_5")
        assert abs(cost - 0.075) < 0.001

    def test_zero_cost_empty_text(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        assert provider.estimate_tts_cost("") == 0.0

    def test_zero_cost_zero_duration(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        assert provider.estimate_stt_cost(0.0) == 0.0


# ---------------------------------------------------------------------------
# Voice listing
# ---------------------------------------------------------------------------


class TestOpenAIVoiceListing:
    def test_list_voices(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        voices = provider.list_voices()
        assert len(voices) == 13
        ids = {v.id for v in voices}
        assert "cedar" in ids
        assert "marin" in ids
        assert "alloy" in ids
        for v in voices:
            assert v.provider == "openai"

    def test_provider_name(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        assert provider.provider_name == "openai"
        assert provider.name == "openai"

    def test_is_available_with_key(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        assert provider.is_available is True

    def test_is_available_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIMediaProvider(api_key="")
            assert provider.is_available is False

    def test_get_tts_models(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        models = provider.get_tts_models()
        assert len(models) == 1
        assert models[0].name == "gpt-4o-mini-tts"
        assert models[0].media_type == MediaType.TTS
        assert models[0].price_per_1m_chars == 15.0

    def test_get_stt_models(self):
        provider = OpenAIMediaProvider(api_key="test-key")
        models = provider.get_stt_models()
        assert len(models) == 2
        names = {m.name for m in models}
        assert "gpt-4o-transcribe" in names
        assert "whisper-1" in names


class TestElevenLabsVoiceListing:
    def test_list_voices(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        voices = provider.list_voices()
        assert len(voices) == 20  # 10 male + 10 female
        names = {v.name for v in voices}
        assert "Max" in names
        assert "Ava" in names
        assert "George" in names
        assert "Sarah" in names
        assert "Leonie" in names
        assert "Audrey" in names
        for v in voices:
            assert v.provider == "elevenlabs"

    def test_voice_gender_split(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        assert len(provider.VOICES_MALE) == 10
        assert len(provider.VOICES_FEMALE) == 10
        for v in provider.VOICES_MALE:
            assert v.gender == "male"
        for v in provider.VOICES_FEMALE:
            assert v.gender == "female"

    def test_provider_name(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        assert provider.provider_name == "elevenlabs"
        assert provider.name == "elevenlabs"

    def test_is_available_with_key(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        assert provider.is_available is True

    def test_get_tts_models(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        models = provider.get_tts_models()
        assert len(models) == 4
        names = {m.name for m in models}
        assert "eleven_v3" in names
        assert "eleven_flash_v2_5" in names

    def test_get_stt_models(self):
        provider = ElevenLabsMediaProvider(api_key="test-key")
        models = provider.get_stt_models()
        assert len(models) == 1
        assert models[0].name == "scribe_v1"


# ---------------------------------------------------------------------------
# Backwards compatibility — old class names still work
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_openai_tts_provider_alias(self):
        provider = OpenAITTSProvider(api_key="test-key")
        assert isinstance(provider, OpenAIMediaProvider)
        assert isinstance(provider, BaseMediaProvider)
        assert provider.provider_name == "openai"

    def test_openai_stt_provider_alias(self):
        provider = OpenAISTTProvider(api_key="test-key")
        assert isinstance(provider, OpenAIMediaProvider)
        assert provider.provider_name == "openai"

    def test_elevenlabs_tts_provider_alias(self):
        provider = ElevenLabsTTSProvider(api_key="test-key")
        assert isinstance(provider, ElevenLabsMediaProvider)
        assert isinstance(provider, BaseMediaProvider)
        assert provider.provider_name == "elevenlabs"

    def test_elevenlabs_stt_provider_alias(self):
        provider = ElevenLabsSTTProvider(api_key="test-key")
        assert isinstance(provider, ElevenLabsMediaProvider)
        assert provider.provider_name == "elevenlabs"


# ---------------------------------------------------------------------------
# Word timing aggregation (ElevenLabs character -> word)
# ---------------------------------------------------------------------------


class TestCharacterTimingAggregation:
    def test_simple_sentence(self):
        chars = list("hi there")
        starts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ends = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        words = _aggregate_character_timings(chars, starts, ends)
        assert len(words) == 2
        assert words[0].text == "hi"
        assert words[0].start == 0.0
        assert words[0].end == 0.2
        assert words[1].text == "there"
        assert words[1].start == 0.3
        assert words[1].end == 0.8

    def test_single_word(self):
        chars = list("hello")
        starts = [0.0, 0.1, 0.2, 0.3, 0.4]
        ends = [0.1, 0.2, 0.3, 0.4, 0.5]
        words = _aggregate_character_timings(chars, starts, ends)
        assert len(words) == 1
        assert words[0].text == "hello"
        assert words[0].start == 0.0
        assert words[0].end == 0.5

    def test_empty(self):
        words = _aggregate_character_timings([], [], [])
        assert words == []

    def test_multiple_spaces(self):
        # "a  b" -- two consecutive spaces
        chars = ["a", " ", " ", "b"]
        starts = [0.0, 0.1, 0.2, 0.3]
        ends = [0.1, 0.2, 0.3, 0.4]
        words = _aggregate_character_timings(chars, starts, ends)
        assert len(words) == 2
        assert words[0].text == "a"
        assert words[1].text == "b"

    def test_trailing_space(self):
        chars = ["h", "i", " "]
        starts = [0.0, 0.1, 0.2]
        ends = [0.1, 0.2, 0.3]
        words = _aggregate_character_timings(chars, starts, ends)
        assert len(words) == 1
        assert words[0].text == "hi"


# ---------------------------------------------------------------------------
# MediaManager auto-discovery
# ---------------------------------------------------------------------------


class TestMediaManagerAutoDiscovery:
    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            mgr = MediaManager()
            assert mgr._tts_providers == []
            assert mgr._stt_providers == []

    def test_no_keys_raises_on_access(self):
        with patch.dict(os.environ, {}, clear=True):
            mgr = MediaManager()
            with pytest.raises(RuntimeError, match="No media provider"):
                _ = mgr.tts
            with pytest.raises(RuntimeError, match="No media provider"):
                _ = mgr.stt

    def test_openai_only(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            assert len(mgr._tts_providers) == 1
            assert len(mgr._stt_providers) == 1
            assert mgr.tts.provider_name == "openai"
            assert mgr.stt.provider_name == "openai"

    def test_elevenlabs_only(self):
        env = {"ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            assert len(mgr._tts_providers) == 1
            assert mgr.tts.provider_name == "elevenlabs"
            assert len(mgr._stt_providers) == 1
            assert mgr.stt.provider_name == "elevenlabs"

    def test_both_providers(self):
        env = {"OPENAI_API_KEY": "sk-test", "ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            assert len(mgr._tts_providers) == 2
            # OpenAI should be first (preferred via cascade)
            assert mgr.tts.provider_name == "openai"

    def test_get_tts_by_name(self):
        env = {"OPENAI_API_KEY": "sk-test", "ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            assert mgr.get_tts("elevenlabs").provider_name == "elevenlabs"
            assert mgr.get_tts("openai").provider_name == "openai"

    def test_get_tts_unknown_raises(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            with pytest.raises(ValueError, match="not available"):
                mgr.get_tts("elevenlabs")

    def test_list_all_voices(self):
        env = {"OPENAI_API_KEY": "sk-test", "ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            voices = mgr.list_all_voices()
            # 13 OpenAI + 20 ElevenLabs
            assert len(voices) == 33
            providers = {v.provider for v in voices}
            assert providers == {"openai", "elevenlabs"}

    def test_get_all_models(self):
        env = {"OPENAI_API_KEY": "sk-test", "ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            models = mgr.get_all_models()
            # OpenAI: 1 TTS + 2 STT; ElevenLabs: 4 TTS + 1 STT = 8 total
            assert len(models) == 8
            tts_models = [m for m in models if m.media_type == MediaType.TTS]
            stt_models = [m for m in models if m.media_type == MediaType.STT]
            assert len(tts_models) == 5  # 1 openai + 4 elevenlabs
            assert len(stt_models) == 3  # 2 openai + 1 elevenlabs

    def test_estimate_tts_cost_via_manager(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            cost = mgr.estimate_tts_cost("x" * 1000, provider="openai")
            assert abs(cost - 0.015) < 0.0001

    def test_estimate_stt_cost_via_manager(self):
        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            cost = mgr.estimate_stt_cost(60.0, provider="openai")
            assert abs(cost - 0.006) < 0.0001

    def test_get_provider(self):
        env = {"OPENAI_API_KEY": "sk-test", "ELEVENLABS_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            mgr = MediaManager()
            p = mgr.get_provider("openai")
            assert isinstance(p, OpenAIMediaProvider)
            p2 = mgr.get_provider("elevenlabs")
            assert isinstance(p2, ElevenLabsMediaProvider)


# ---------------------------------------------------------------------------
# ElevenLabs voice resolution
# ---------------------------------------------------------------------------


class TestElevenLabsVoiceResolution:
    def test_explicit_voice(self):
        provider = ElevenLabsMediaProvider(api_key="test")
        assert provider._resolve_voice("custom-id", "en") == "custom-id"

    def test_german_default(self):
        provider = ElevenLabsMediaProvider(api_key="test")
        resolved = provider._resolve_voice("", "de")
        assert resolved == "AnvlJBAqSLDzEevYr9Ap"  # Ava

    def test_english_default(self):
        provider = ElevenLabsMediaProvider(api_key="test")
        resolved = provider._resolve_voice("", "en")
        assert resolved == "Gfpl8Yo74Is0W6cPUWWT"  # Max

    def test_french_default(self):
        provider = ElevenLabsMediaProvider(api_key="test")
        resolved = provider._resolve_voice("", "fr")
        assert resolved == "McVZB9hVxVSk3Equu8EH"  # Audrey

    def test_fallback(self):
        provider = ElevenLabsMediaProvider(api_key="test")
        resolved = provider._resolve_voice("", "")
        assert resolved == "Gfpl8Yo74Is0W6cPUWWT"  # Max (first voice)


# ---------------------------------------------------------------------------
# OpenAI TTS kwargs building
# ---------------------------------------------------------------------------


class TestOpenAITTSKwargs:
    def test_default_kwargs(self):
        provider = OpenAIMediaProvider(api_key="test")
        kwargs = provider._build_tts_kwargs("Hello", "cedar", "")
        assert kwargs["model"] == "gpt-4o-mini-tts"
        assert kwargs["voice"] == "cedar"
        assert kwargs["input"] == "Hello"
        assert kwargs["response_format"] == "mp3"
        assert "instructions" not in kwargs

    def test_language_instruction(self):
        provider = OpenAIMediaProvider(api_key="test")
        kwargs = provider._build_tts_kwargs("Hallo", "cedar", "de")
        assert kwargs["instructions"] == "Speak in German."

    def test_unknown_language(self):
        provider = OpenAIMediaProvider(api_key="test")
        kwargs = provider._build_tts_kwargs("Hi", "cedar", "xx-unknown")
        assert "instructions" not in kwargs


# ===========================================================================
# Integration tests (skipped without API keys)
# ===========================================================================

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
_has_elevenlabs_key = bool(os.environ.get("ELEVENLABS_API_KEY"))
_has_both_keys = _has_openai_key and _has_elevenlabs_key


# ---------------------------------------------------------------------------
# OpenAI TTS integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestOpenAITTSIntegration:
    async def test_synthesize_basic(self):
        """Basic TTS: generate audio, verify non-empty MP3 bytes."""
        provider = OpenAIMediaProvider()
        result = await provider.synthesize("Hello, this is a test.", voice="cedar")
        assert len(result.audio_bytes) > 100
        assert result.content_type == "audio/mpeg"
        assert result.word_timings == []

    async def test_synthesize_with_timings(self):
        """TTS with word timings: verify timestamps are ordered."""
        provider = OpenAIMediaProvider()
        result = await provider.synthesize(
            "Hello, this is a test.",
            voice="cedar",
            with_timings=True,
        )
        assert len(result.audio_bytes) > 100
        assert len(result.word_timings) > 0
        # Each timing should have start <= end
        for wt in result.word_timings:
            assert wt.start <= wt.end

    async def test_openai_tts_all_voices(self):
        """Synthesize a short phrase with each of the 13 OpenAI voices."""
        provider = OpenAIMediaProvider()
        voices = provider.list_voices()
        assert len(voices) == 13
        for voice_info in voices:
            result = await provider.synthesize(
                "Hi.",
                voice=voice_info.id,
            )
            assert len(result.audio_bytes) > 50, (
                f"Voice {voice_info.name!r} returned too few bytes"
            )
            assert result.content_type == "audio/mpeg"

    async def test_openai_tts_german(self):
        """Synthesize German text and verify audio is produced."""
        provider = OpenAIMediaProvider()
        result = await provider.synthesize(
            "Guten Tag, das ist ein Test.",
            voice="cedar",
            language="de",
        )
        assert len(result.audio_bytes) > 100
        assert result.content_type == "audio/mpeg"


# ---------------------------------------------------------------------------
# OpenAI STT integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestOpenAISTTIntegration:
    async def test_transcribe_generated_audio(self):
        """Generate audio with TTS, then transcribe it back."""
        provider = OpenAIMediaProvider()
        tts_result = await provider.synthesize("The quick brown fox.")
        stt_result = await provider.transcribe(
            tts_result.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        assert "quick" in stt_result.text.lower() or "fox" in stt_result.text.lower()

    async def test_transcribe_too_small(self):
        """Tiny audio payload should return empty text, not crash."""
        provider = OpenAIMediaProvider()
        result = await provider.transcribe(b"\x00" * 10)
        assert result.text == ""

    async def test_openai_stt_roundtrip(self):
        """Full round-trip: TTS -> STT, verify transcription matches original."""
        original = "Hello world, this is a test."
        provider = OpenAIMediaProvider()
        audio = await provider.synthesize(original, voice="cedar")
        result = await provider.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        text_lower = result.text.lower()
        assert "hello" in text_lower
        assert "test" in text_lower

    async def test_openai_stt_with_word_timings(self):
        """STT with word timings: verify start < end and words are present."""
        provider = OpenAIMediaProvider()
        audio = await provider.synthesize("The sun is shining today.", voice="cedar")
        result = await provider.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
            with_timings=True,
        )
        assert len(result.word_timings) > 0
        for wt in result.word_timings:
            assert wt.start <= wt.end, f"Bad timing: {wt}"
        # Verify timings are in order (non-decreasing starts)
        starts = [wt.start for wt in result.word_timings]
        assert starts == sorted(starts), "Word timings not in chronological order"
        # At least one expected word should be present
        words_lower = [wt.text.lower().strip(".,!?") for wt in result.word_timings]
        assert "sun" in words_lower or "shining" in words_lower


# ---------------------------------------------------------------------------
# ElevenLabs TTS integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_elevenlabs_key, reason="ELEVENLABS_API_KEY not set")
class TestElevenLabsTTSIntegration:
    async def test_synthesize_basic(self):
        """Basic ElevenLabs TTS with default voice (Max)."""
        provider = ElevenLabsMediaProvider()
        result = await provider.synthesize("Hello from ElevenLabs.", language="en")
        assert len(result.audio_bytes) > 100
        assert result.content_type == "audio/mpeg"

    async def test_synthesize_with_timings(self):
        """ElevenLabs TTS with character-to-word timing aggregation."""
        provider = ElevenLabsMediaProvider()
        result = await provider.synthesize(
            "Hello from ElevenLabs.",
            language="en",
            with_timings=True,
        )
        assert len(result.audio_bytes) > 100
        assert len(result.word_timings) > 0
        for wt in result.word_timings:
            assert wt.start <= wt.end

    async def test_elevenlabs_tts_german_voice(self):
        """Synthesize German text with Ava (German default voice)."""
        provider = ElevenLabsMediaProvider()
        result = await provider.synthesize(
            "Guten Tag, das ist ein Test.",
            language="de",
        )
        assert len(result.audio_bytes) > 100
        assert result.content_type == "audio/mpeg"

    async def test_elevenlabs_tts_different_model(self):
        """Test with eleven_flash_v2_5 (faster, cheaper model)."""
        provider = ElevenLabsMediaProvider(model="eleven_flash_v2_5")
        result = await provider.synthesize(
            "Hello world.",
            language="en",
        )
        assert len(result.audio_bytes) > 100
        assert result.content_type == "audio/mpeg"


# ---------------------------------------------------------------------------
# ElevenLabs STT integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_elevenlabs_key, reason="ELEVENLABS_API_KEY not set")
class TestElevenLabsSTTIntegration:
    async def test_elevenlabs_stt_basic(self):
        """Generate audio with ElevenLabs TTS, transcribe back with STT."""
        provider = ElevenLabsMediaProvider()
        audio = await provider.synthesize(
            "Hello world, this is a test.",
            language="en",
        )
        result = await provider.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        assert "hello" in result.text.lower()

    async def test_elevenlabs_stt_with_word_timings(self):
        """ElevenLabs STT with word-level timings from Scribe API."""
        provider = ElevenLabsMediaProvider()
        audio = await provider.synthesize(
            "The weather is nice today.",
            language="en",
        )
        result = await provider.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
            with_timings=True,
        )
        assert len(result.text) > 0
        assert len(result.word_timings) > 0
        for wt in result.word_timings:
            assert wt.start <= wt.end, f"Bad timing: {wt}"
        # Timings should be chronologically ordered
        starts = [wt.start for wt in result.word_timings]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Manager integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestManagerIntegration:
    async def test_manager_tts_default_is_openai(self):
        """When both keys exist, OpenAI should be the default TTS provider."""
        mgr = MediaManager()
        assert mgr.tts.provider_name == "openai"

    async def test_manager_synthesize(self):
        """Synthesize through the manager convenience method."""
        mgr = MediaManager()
        result = await mgr.synthesize("Hello from the manager.", voice="cedar")
        assert len(result.audio_bytes) > 100

    async def test_manager_transcribe(self):
        """Round-trip through manager: synthesize then transcribe."""
        mgr = MediaManager()
        audio = await mgr.synthesize("Testing the manager.", voice="cedar")
        result = await mgr.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        assert len(result.text) > 0


@pytest.mark.skipif(not _has_both_keys, reason="Both OPENAI_API_KEY and ELEVENLABS_API_KEY required")
class TestManagerSwitchProvider:
    async def test_manager_switch_provider(self):
        """Synthesize with both providers via the manager, compare results."""
        mgr = MediaManager()
        openai_p = mgr.get_provider("openai")
        el_p = mgr.get_provider("elevenlabs")
        openai_result = await openai_p.synthesize(
            "Hello world.", voice="cedar",
        )
        el_result = await el_p.synthesize(
            "Hello world.", language="en",
        )
        assert len(openai_result.audio_bytes) > 100
        assert len(el_result.audio_bytes) > 100
        # Different providers should produce different bytes
        assert openai_result.audio_bytes != el_result.audio_bytes

    async def test_manager_stt_both_providers(self):
        """Transcribe the same audio with both STT providers."""
        mgr = MediaManager()
        openai_p = mgr.get_provider("openai")
        el_p = mgr.get_provider("elevenlabs")
        # Generate audio with OpenAI
        audio = await openai_p.synthesize(
            "Good morning everyone.", voice="cedar",
        )

        openai_text = await openai_p.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        el_text = await el_p.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        # Both should recognize the core content
        assert "morning" in openai_text.text.lower() or "good" in openai_text.text.lower()
        assert "morning" in el_text.text.lower() or "good" in el_text.text.lower()


# ---------------------------------------------------------------------------
# Cross-provider integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_both_keys, reason="Both OPENAI_API_KEY and ELEVENLABS_API_KEY required")
class TestCrossProviderIntegration:
    async def test_elevenlabs_tts_openai_stt(self):
        """Generate audio with ElevenLabs, transcribe with OpenAI."""
        el = get_media_provider("elevenlabs")(api_key=os.environ["ELEVENLABS_API_KEY"])
        audio = await el.synthesize(
            "Cross provider test one.",
            language="en",
        )
        assert len(audio.audio_bytes) > 100

        oai = get_media_provider("openai")(api_key=os.environ["OPENAI_API_KEY"])
        result = await oai.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        assert "cross" in result.text.lower() or "provider" in result.text.lower()

    async def test_openai_tts_elevenlabs_stt(self):
        """Generate audio with OpenAI, transcribe with ElevenLabs."""
        oai = get_media_provider("openai")(api_key=os.environ["OPENAI_API_KEY"])
        audio = await oai.synthesize(
            "Cross provider test two.",
            voice="cedar",
        )
        assert len(audio.audio_bytes) > 100

        el = get_media_provider("elevenlabs")(api_key=os.environ["ELEVENLABS_API_KEY"])
        result = await el.transcribe(
            audio.audio_bytes,
            filename="speech.mp3",
            content_type="audio/mpeg",
        )
        assert "cross" in result.text.lower() or "provider" in result.text.lower()
