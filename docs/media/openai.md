# OpenAI TTS/STT

OpenAI provides both text-to-speech (`gpt-4o-mini-tts`) and speech-to-text (`gpt-4o-transcribe`, `whisper-1`) models.

## Text-to-Speech

### Model

- **gpt-4o-mini-tts** -- Supports custom voice instructions and language control

### Voices

| Voice | Gender | Style |
|---|---|---|
| Cedar | Male | Default, versatile |
| Marin | Female | Warm |
| Ash | Male | Calm |
| Ballad | Male | Storytelling |
| Coral | Female | Conversational |
| Echo | Male | Clear |
| Fable | Male | Animated |
| Nova | Female | Energetic |
| Onyx | Male | Deep |
| Sage | Female | Wise |
| Shimmer | Female | Bright |
| Verse | Female | Poetic |
| Alloy | Neutral | Balanced |

### Usage

```python
from llming_models.media.openai_media import OpenAITTSProvider

tts = OpenAITTSProvider()  # uses OPENAI_API_KEY from env

# Basic synthesis
result = await tts.synthesize("Hello, world!", voice="cedar")
# result.audio_bytes -> MP3 audio
# result.content_type -> "audio/mpeg"

# With word-level timestamps (uses whisper-1 for alignment)
result = await tts.synthesize(
    "Hello, world!",
    voice="nova",
    with_timings=True,
)
for timing in result.word_timings:
    print(f"{timing.text}: {timing.start:.2f}s - {timing.end:.2f}s")
```

### Language Support

The `gpt-4o-mini-tts` model accepts language instructions:

```python
result = await tts.synthesize(
    "Guten Tag, wie geht es Ihnen?",
    voice="cedar",
    language="de",  # adds "Speak in German." instruction
)
```

Supported language codes: `en`, `de`, `fr`, `it`, `hi` (and regional variants).

## Speech-to-Text

### Models

- **gpt-4o-transcribe** -- Primary STT model, higher accuracy
- **whisper-1** -- Fallback model, supports word-level timestamps

### Usage

```python
from llming_models.media.openai_media import OpenAISTTProvider

stt = OpenAISTTProvider()

# Basic transcription
result = await stt.transcribe(audio_bytes, filename="recording.webm")
print(result.text)

# With word-level timestamps (uses whisper-1)
result = await stt.transcribe(
    audio_bytes,
    filename="recording.webm",
    with_timings=True,
)
for segment in result.word_timings:
    print(f"{segment.text}: {segment.start:.2f}s - {segment.end:.2f}s")
```

### Fallback Behavior

If `gpt-4o-transcribe` encounters a format error, the provider automatically retries with `whisper-1`. This handles edge cases with unusual audio formats.

!!! warning "Timing source"
    When requesting word timings from TTS, the audio is transcribed back through `whisper-1` to extract timestamps. This adds latency and a small additional cost.

!!! note "Audio formats"
    OpenAI STT accepts common audio formats: WebM, MP3, WAV, M4A. The `filename` parameter helps the API detect the format.
