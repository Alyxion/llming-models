# Adding Custom Media Providers

You can extend the media system by implementing the `TTSProvider` and/or `STTProvider` base classes.

## Implementing a TTS Provider

```python
from llming_models.media.base import TTSProvider, TTSResult, VoiceInfo


class MyTTSProvider(TTSProvider):
    """Custom TTS provider."""

    @property
    def provider_name(self) -> str:
        return "my_provider"

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id="voice-1",
                name="Default",
                gender="neutral",
                language="en",
                provider="my_provider",
            ),
        ]

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "",
        language: str = "",
        with_timings: bool = False,
    ) -> TTSResult:
        # Your synthesis logic here
        audio_bytes = await self._call_my_api(text, voice)

        return TTSResult(
            audio_bytes=audio_bytes,
            content_type="audio/mpeg",
            word_timings=[],  # populate if with_timings=True
        )
```

## Implementing an STT Provider

```python
from llming_models.media.base import STTProvider, STTResult, TranscriptSegment


class MySTTProvider(STTProvider):
    """Custom STT provider."""

    @property
    def provider_name(self) -> str:
        return "my_provider"

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str = "",
        with_timings: bool = False,
    ) -> STTResult:
        # Your transcription logic here
        text = await self._call_my_api(audio_bytes)

        segments = []
        if with_timings:
            # Populate word-level segments
            segments = [
                TranscriptSegment(text="Hello", start=0.0, end=0.5, type="word"),
                TranscriptSegment(text=" ", start=0.5, end=0.5, type="spacing"),
                TranscriptSegment(text="world", start=0.5, end=1.0, type="word"),
            ]

        return STTResult(
            text=text,
            language=language or None,
            segments=segments,
        )
```

## Registering with MediaManager

The `MediaManager` currently auto-discovers OpenAI and ElevenLabs providers. To add a custom provider, register it after initialization:

```python
from llming_models.media import MediaManager

media = MediaManager()

# Register custom providers
my_tts = MyTTSProvider(api_key="...")
my_stt = MySTTProvider(api_key="...")

media._tts_providers.append(my_tts)
media._stt_providers.append(my_stt)

# Use it
result = await media.get_tts("my_provider").synthesize("Hello!")
```

!!! tip "Provider priority"
    Providers are selected by list order. Append to add a lower-priority fallback, or insert at index 0 to make your provider the default.

## Segment Types

When implementing `with_timings`, use these segment types for rich transcripts:

| Type | Description | Example |
|---|---|---|
| `word` | A spoken word | `"Hello"` |
| `spacing` | Whitespace between words | `" "` |
| `punctuation` | Punctuation marks | `"."`, `","` |
| `audio_event` | Non-speech sound | `"(laughter)"` |

The `STTResult` class provides convenience properties that filter segments:

- `word_timings` -- Only `word` type segments
- `audio_events` -- Only `audio_event` type segments
- `rich_text` -- Full text with audio events inline
