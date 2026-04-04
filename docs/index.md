# llming-models

**LLM execution engine -- multi-provider streaming, MCP tools, and budget management.**

llming-models provides the core runtime for building multi-provider LLM applications. It handles model metadata, streaming chat sessions with tool support, MCP integration, per-user configuration, and monetary budget tracking -- all behind a unified API.

---

## Key Features

- **Multi-Provider Streaming** -- Unified async/sync streaming across OpenAI, Anthropic, Google Gemini, Azure, Mistral, and Together AI
- **Model Metadata** -- Rich descriptors with pricing, context windows, capability flags, and UI hints
- **Budget Management** -- Track and enforce monetary limits per time period with reservation/rollback semantics
- **MCP Tools** -- First-class Model Context Protocol support with tool registries and toolbox adapters
- **Media (TTS/STT)** -- Text-to-speech and speech-to-text with OpenAI and ElevenLabs, including word-level timestamps
- **Chat Sessions** -- High-level `ChatSession` with automatic tool dispatch, conversation history, and image support
- **Configuration** -- Global and per-user model selection with category mappings and provider cascade

---

## Quick Example

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
    system_prompt="You are a helpful assistant.",
)

async for chunk in session.stream("Explain async generators in Python"):
    print(chunk.content, end="")
```

---

## Navigation

| Section | What you'll find |
|---|---|
| [Installation](getting-started/installation.md) | pip install, Poetry setup, environment variables |
| [Quick Start](getting-started/quickstart.md) | First chat, streaming, model selection |
| [Providers](providers/overview.md) | OpenAI, Anthropic, Google, Mistral, Together, Azure |
| [Media](media/overview.md) | TTS and STT with OpenAI and ElevenLabs |
| [Core Concepts](concepts/sessions.md) | Sessions, budget, tools, MCP, configuration |

---

## Supported Providers

| Provider | Models | Features |
|---|---|---|
| OpenAI | GPT-5.4, GPT-5.2, GPT-5-mini, GPT-5-nano | Streaming, vision, reasoning, web search |
| Anthropic | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 | Streaming, vision, reasoning, web search |
| Google | Gemini 3 Pro, Gemini 3 Flash | 1M context, streaming, vision |
| Mistral | Large, Medium, Small | EU-hosted, multilingual |
| Together AI | DeepSeek models | Cost-effective inference |
| Azure | OpenAI + Anthropic models | Enterprise hosting |

---

## License

MIT License. Copyright (c) 2026 [Michael Ikemann](https://github.com/Alyxion).
