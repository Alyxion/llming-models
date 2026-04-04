# API Reference: Media

## MediaManager

`llming_models.media.manager.MediaManager`

Auto-discovers and manages TTS/STT providers.

### Constructor

```python
MediaManager()  # auto-discovers from environment variables
```

### Properties and Methods

```python
# Preferred providers
media.tts -> TTSProvider       # preferred TTS provider
media.stt -> STTProvider       # preferred STT provider

# Specific provider
media.get_tts(provider: str = "") -> TTSProvider
media.get_stt(provider: str = "") -> STTProvider

# All voices
media.list_all_voices() -> list[VoiceInfo]

# Convenience methods
await media.synthesize(text, **kwargs) -> TTSResult
await media.transcribe(audio_bytes, **kwargs) -> STTResult
```

## TTSProvider

`llming_models.media.base.TTSProvider`

Abstract base class for text-to-speech providers.

```python
class TTSProvider(ABC):
    @property
    def provider_name(self) -> str: ...

    def list_voices(self) -> list[VoiceInfo]: ...

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult: ...
```

## STTProvider

`llming_models.media.base.STTProvider`

Abstract base class for speech-to-text providers.

```python
class STTProvider(ABC):
    @property
    def provider_name(self) -> str: ...

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str = "",
        with_timings: bool = False,
    ) -> STTResult: ...
```

## Data Models

### VoiceInfo

```python
class VoiceInfo(BaseModel):
    id: str           # Voice identifier
    name: str         # Human-readable name
    gender: str       # "male", "female", "neutral"
    language: str     # Language code
    provider: str     # Provider name
```

### TTSResult

```python
class TTSResult(BaseModel):
    audio_bytes: bytes                          # Raw audio
    content_type: str = "audio/mpeg"            # MIME type
    word_timings: list[TranscriptSegment] = []  # Timestamps
    duration: float | None = None               # Seconds
```

### STTResult

```python
class STTResult(BaseModel):
    text: str                                    # Clean text
    language: str | None = None                  # Detected language
    segments: list[TranscriptSegment] = []       # All segments
    confidence: float | None = None              # Overall confidence

    @property
    def word_timings(self) -> list[TranscriptSegment]: ...  # Word segments
    @property
    def audio_events(self) -> list[TranscriptSegment]: ...  # Non-speech
    @property
    def rich_text(self) -> str: ...  # Text with audio events
```

### TranscriptSegment

```python
class TranscriptSegment(BaseModel):
    text: str                    # Segment content
    start: float                 # Start time (seconds)
    end: float                   # End time (seconds)
    type: str = "word"           # "word", "spacing", "punctuation", "audio_event"
    confidence: float | None     # Confidence score
```

## Implementations

| Class | Module | Provider |
|---|---|---|
| `OpenAITTSProvider` | `llming_models.media.openai_media` | OpenAI |
| `OpenAISTTProvider` | `llming_models.media.openai_media` | OpenAI |
| `ElevenLabsTTSProvider` | `llming_models.media.elevenlabs_media` | ElevenLabs |
| `ElevenLabsSTTProvider` | `llming_models.media.elevenlabs_media` | ElevenLabs |
