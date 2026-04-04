# Chat Playground

llming-models ships with an interactive chat playground for testing models, streaming, TTS/STT, and tool use.

## Running the Playground

```bash
# Set up environment
cp .env.template .env   # fill in your API keys

# Launch the playground
python samples/chat_app.py

# Open http://localhost:8000
```

## Features

The playground provides a full-featured chat interface:

- **Model selection** -- Switch between all configured providers and models
- **Streaming responses** -- Real-time token-by-token output
- **TTS / STT** -- Text-to-speech and speech-to-text with OpenAI and ElevenLabs
- **Push-to-talk** -- Hold-to-record voice input
- **Word-level highlighting** -- Synchronized text highlighting during TTS playback
- **Token cost tracking** -- Per-message and session-level cost display
- **Tool support** -- Web search, image generation, and MCP tools

## Requirements

The playground requires API keys for at least one LLM provider. For voice features, set `OPENAI_API_KEY` or `ELEVENLABS_API_KEY`.

!!! tip "Multiple providers"
    Configure multiple provider keys to compare model outputs side by side. The playground lets you switch models on the fly.

## Screenshot

<p align="center"><img src="https://raw.githubusercontent.com/Alyxion/llming-models/main/docs/sample_screenshot_small.png" alt="llming playground" width="800"></p>

## Configuration

The playground reads configuration from environment variables and uses sensible defaults. Customize behavior by modifying `samples/chat_app.py`:

```python
# Example: change default model and system prompt
session = ChatSession(
    config=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        temperature=0.5,
    ),
    system_prompt="You are a creative writing assistant.",
)
```

!!! note
    The playground is a sample application for development and testing. For production use, build your own UI on top of the `ChatSession` API.
