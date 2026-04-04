# ElevenLabs TTS/STT

ElevenLabs provides high-quality text-to-speech with character-level timestamps and speech-to-text via the Scribe API.

## Text-to-Speech

### Models

| Model | Description |
|---|---|
| `eleven_v3` | Latest, highest quality (default) |
| `eleven_multilingual_v2` | Strong multilingual support |
| `eleven_turbo_v2_5` | Optimized for speed |
| `eleven_flash_v2_5` | Fastest, lowest latency |

### Curated Voices

ElevenLabs ships with 20 curated voices (10 male, 10 female) available in every account:

**Male voices:**

| Name | Language | Style |
|---|---|---|
| Max | English | Educational, documentary |
| George | English (GB) | Warm storyteller |
| Brian | English | Deep, narration |
| Daniel | English (GB) | Steady broadcaster |
| Eric | English | Smooth, conversational |
| Adam | English | Dominant, firm |
| Chris | English | Charming |
| Felix Serenitas | German | Calm, trustworthy |
| Archie | English (GB) | Social media narrator |
| Liam | English | Energetic |

**Female voices:**

| Name | Language | Style |
|---|---|---|
| Ava | German | Youthful, narrative |
| Sarah | English | Mature, reassuring |
| Alice | English (GB) | Clear, educational |
| Matilda | English | Professional |
| Jessica | English | Playful, warm |
| Bella | English | Informative |
| Lily | English (GB) | Velvety |
| Laura | English | Enthusiastic |
| Leonie | German | Clear, engaging |
| Audrey | French | Energetic |

### Usage

```python
from llming_models.media.elevenlabs_media import ElevenLabsTTSProvider

tts = ElevenLabsTTSProvider()  # uses ELEVENLABS_API_KEY from env

# Basic synthesis (auto-selects voice by language)
result = await tts.synthesize("Hello, world!")

# With specific voice
result = await tts.synthesize("Hello!", voice="Gfpl8Yo74Is0W6cPUWWT")  # Max

# With character-level timing (aggregated to words)
result = await tts.synthesize("Hello, world!", with_timings=True)
for timing in result.word_timings:
    print(f"{timing.text}: {timing.start:.2f}s - {timing.end:.2f}s")
```

### Custom Voice IDs

Pass any ElevenLabs voice ID directly:

```python
result = await tts.synthesize(
    "Custom voice test",
    voice="your-custom-voice-id",
)
```

### Language-Based Voice Selection

When no voice is specified, the provider selects a default voice by language:

| Language | Default Voice |
|---|---|
| `en` | Max |
| `de` | Ava |
| `fr` | Audrey |

## Speech-to-Text (Scribe)

ElevenLabs STT uses the `scribe_v1` model with rich transcript output.

### Usage

```python
from llming_models.media.elevenlabs_media import ElevenLabsSTTProvider

stt = ElevenLabsSTTProvider()

result = await stt.transcribe(audio_bytes, filename="recording.webm")
print(result.text)
```

### Rich Transcripts

Scribe returns detailed segments with types beyond just words:

```python
result = await stt.transcribe(audio_bytes, with_timings=True)

# All segment types: word, spacing, punctuation, audio_event
for seg in result.segments:
    print(f"[{seg.type}] {seg.text!r} ({seg.start:.2f}s-{seg.end:.2f}s)")

# Audio events (laughter, music, etc.)
for event in result.audio_events:
    print(f"Non-speech: {event.text} at {event.start:.2f}s")

# Rich text with audio events inline
print(result.rich_text)
# "Hello (laughter) world"
```

!!! tip "Audio events"
    ElevenLabs Scribe detects non-speech sounds like `(laughter)`, `(clears throat)`, and `(music)`. Access them via `result.audio_events` or `result.rich_text`.

!!! note "Environment"
    Set `ELEVENLABS_API_KEY` in your environment. The provider is auto-discovered by `MediaManager` when the key is present.
