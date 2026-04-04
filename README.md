<p align="center"><img src="https://raw.githubusercontent.com/Alyxion/llming-models/main/docs/logo-small.png" alt="LLMing Models" width="400"></p>

# llming-models

[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Alyxion/llming-models/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/llming-models.svg)](https://pypi.org/project/llming-models/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-purple.svg)](https://github.com/astral-sh/ruff)

**LLM execution engine -- multi-provider streaming, MCP tools, and budget management.**

llming-models provides the core runtime for building multi-provider LLM applications. It handles model metadata and capabilities, streaming chat sessions with tool support, MCP (Model Context Protocol) integration, per-user configuration, and monetary budget tracking -- all behind a unified API that works across OpenAI, Anthropic, Google, Azure, Mistral, and Together/DeepSeek.

## Features

- **Model Metadata** -- Rich model descriptors with pricing, context windows, capability flags (vision, reasoning), and UI hints (speed/quality ratings)
- **Configuration** -- Global and per-user model selection with category mappings, provider cascade priority, and include/exclude filters
- **Budget Management** -- Track and enforce monetary limits per time period with reservation/rollback semantics and pluggable backends (memory, MongoDB)
- **Multi-Provider Streaming** -- Unified async/sync streaming across OpenAI, Anthropic, Google Gemini, Azure OpenAI, Azure Anthropic, Mistral, and Together AI
- **MCP Tools** -- First-class Model Context Protocol support with tool registries, toolbox adapters, and built-in MCP servers (math, image generation)
- **Chat Sessions** -- High-level `ChatSession` with automatic tool dispatch, conversation history, image support, and reasoning effort control
- **Conversation Persistence** -- IndexedDB-compatible conversation storage with metadata, avatars, and file references

## Chat Playground

<p align="center"><img src="https://raw.githubusercontent.com/Alyxion/llming-models/main/docs/sample_screenshot_small.png" alt="llming playground" width="800"></p>

Interactive chat UI with model selection, streaming responses, TTS/STT (OpenAI + ElevenLabs), push-to-talk, word-level highlighting, and token cost tracking. Run it with:

```bash
cp .env.template .env   # fill in API keys
python samples/chat_app.py
# Open http://localhost:8000
```

## Quick Start

### Installation

```bash
pip install llming-models
```

Or from source with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/Alyxion/llming-models.git
cd llming-models
poetry install
```

### Basic Usage

```python
from llming_models import LLMInfo, ModelSize, ReasoningEffort

model = LLMInfo(
    provider="anthropic",
    name="claude_sonnet",
    label="Claude Sonnet 4.6",
    model="claude-sonnet-4-6",
    description="Fast, capable model for most tasks",
    input_token_price=3.0,
    output_token_price=15.0,
    size=ModelSize.MEDIUM,
    max_input_tokens=200_000,
    max_output_tokens=64_000,
    supports_image_input=True,
    reasoning=True,
    default_reasoning_effort=ReasoningEffort.MEDIUM,
    speed=8,
    quality=8,
    best_use="Code & analysis",
    highlights=["Fast", "Code", "Vision"],
)

print(f"{model.label}: ${model.input_token_price}/1M in, ${model.output_token_price}/1M out")
```

### Configuration

```python
from llming_models import LLMGlobalConfig, LLMUserConfig, ModelCategories

global_config = LLMGlobalConfig(
    default_models={
        ModelCategories.SMALL: ["claude_haiku", "gpt-4o-mini"],
        ModelCategories.LARGE: ["claude_sonnet", "gpt-4o"],
    },
    provider_cascade=["anthropic", "openai"],
)

user_config = LLMUserConfig(
    global_config=global_config,
    default_models={ModelCategories.LARGE: "claude_sonnet"},
)

print(user_config.get_default_model(ModelCategories.LARGE))  # "claude_sonnet"
```

### Budget Management

```python
import asyncio
from llming_models import MemoryBudgetLimit, LLMBudgetManager, LimitPeriod

async def main():
    limits = [
        MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY),
        MemoryBudgetLimit(name="monthly", amount=100.0, period=LimitPeriod.MONTHLY),
    ]
    manager = LLMBudgetManager(limits)

    available = await manager.available_budget_async()
    print(f"Available: {available:.2f}")

    await manager.reserve_budget_async(
        input_tokens=1000,
        max_output_tokens=2000,
        input_token_price=3.0,    # per 1M tokens
        output_token_price=15.0,  # per 1M tokens
    )

asyncio.run(main())
```

## Project Structure

```
llming-models/
├── llming_models/          # Models, config, sessions, budget, providers, tools
│   ├── budget/             # Cost tracking with time-period limits (memory + MongoDB)
│   ├── providers/          # OpenAI, Anthropic, Google, Azure, Mistral, Together
│   ├── tools/              # Tool system, MCP integration, math server, image gen
│   └── utils/              # Image encoding utilities
├── tests/                  # 1255 tests (unit + integration with live APIs)
├── samples/                # Example scripts
└── docs/                   # Logo and assets
```

## Development

```bash
poetry install
poetry run pytest             # 1255 tests
poetry run ruff check         # lint
poetry run mypy llming_models # type check
```

## License

This project is licensed under the [MIT License](https://github.com/Alyxion/llming-models/blob/main/LICENSE). Copyright (c) 2026 [Michael Ikemann](https://github.com/Alyxion).
