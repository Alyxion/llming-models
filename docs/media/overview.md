# Media Provider Architecture

llming-models includes a unified TTS (text-to-speech) and STT (speech-to-text) system that mirrors the LLM provider architecture. Multiple backends are supported with automatic provider selection and word-level timestamp support.

## Design

The media system follows the same pattern as LLM providers:

- **Abstract base classes** -- `TTSProvider` and `STTProvider` define the interface
- **Concrete implementations** -- OpenAI and ElevenLabs providers
- **Manager** -- `MediaManager` auto-discovers available providers and selects the best one

```
llming_models.media.base        # TTSProvider, STTProvider, VoiceInfo, TTSResult, STTResult
llming_models.media.manager     # MediaManager (auto-discovery + selection)
llming_models.media.openai_media     # OpenAI TTS/STT
llming_models.media.elevenlabs_media # ElevenLabs TTS/STT
```

## Unified API

The `MediaManager` provides a single entry point for all media operations:

```python
from llming_models.media import MediaManager

media = MediaManager()  # auto-discovers providers from env vars

# Text-to-speech
result = await media.synthesize("Hello, world!", voice="cedar")
# result.audio_bytes, result.content_type, result.word_timings

# Speech-to-text
result = await media.transcribe(audio_bytes)
# result.text, result.segments, result.word_timings
```

## Provider Selection

`MediaManager` checks environment variables on initialization and registers available providers:

| Priority | Provider | Env Variable |
|---|---|---|
| 1 (preferred) | OpenAI | `OPENAI_API_KEY` |
| 2 | ElevenLabs | `ELEVENLABS_API_KEY` |

The first available provider is used by default. You can also request a specific provider:

```python
tts = media.get_tts("elevenlabs")
stt = media.get_stt("openai")
```

## Core Data Models

### VoiceInfo

```python
class VoiceInfo(BaseModel):
    id: str           # Voice identifier
    name: str         # Human-readable name
    gender: str       # "male", "female", "neutral"
    language: str     # e.g. "en", "de"
    provider: str     # "openai" or "elevenlabs"
```

### TTSResult

```python
class TTSResult(BaseModel):
    audio_bytes: bytes                    # Raw audio data
    content_type: str = "audio/mpeg"      # MIME type
    word_timings: list[TranscriptSegment] # Word-level timestamps
    duration: float | None                # Total duration in seconds
```

### STTResult

```python
class STTResult(BaseModel):
    text: str                              # Clean transcription
    language: str | None                   # Detected language
    segments: list[TranscriptSegment]      # Full detailed transcript
    confidence: float | None               # Overall confidence

    @property
    def word_timings(self) -> list[TranscriptSegment]: ...  # Word segments only
    @property
    def audio_events(self) -> list[TranscriptSegment]: ...  # Non-speech events
    @property
    def rich_text(self) -> str: ...  # Text with audio events inline
```

### TranscriptSegment

```python
class TranscriptSegment(BaseModel):
    text: str          # Segment text
    start: float       # Start time (seconds)
    end: float         # End time (seconds)
    type: str          # "word", "spacing", "punctuation", "audio_event"
    confidence: float | None
```

!!! tip "Word-level highlighting"
    Use `with_timings=True` on both TTS and STT calls to get word-level timestamps for synchronized text highlighting in chat UIs.
