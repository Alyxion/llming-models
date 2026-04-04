# Quick Start

## Basic Chat Session

Create a chat session and send a message:

```python
import asyncio
from llming_models import ChatSession, LLMConfig

async def main():
    session = ChatSession(
        config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
        system_prompt="You are a helpful assistant.",
    )

    async for chunk in session.stream("What is the capital of France?"):
        print(chunk.content, end="")
    print()

asyncio.run(main())
```

## Choosing a Model

Models are specified by provider and model name:

=== "Anthropic"

    ```python
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-6")
    ```

=== "OpenAI"

    ```python
    config = LLMConfig(provider="openai", model="gpt-5.2")
    ```

=== "Google"

    ```python
    config = LLMConfig(provider="google", model="gemini-3-pro-preview")
    ```

## Model Discovery

Use `LLMManager` to discover available models across all configured providers:

```python
from llming_models import LLMManager

manager = LLMManager()

for model in manager.get_available_models():
    print(f"{model.label}: ${model.input_token_price}/1M in, "
          f"${model.output_token_price}/1M out")
```

## Streaming with Budget Tracking

Combine streaming with budget management:

```python
import asyncio
from llming_models import (
    ChatSession, LLMConfig,
    LLMBudgetManager, MemoryBudgetLimit, LimitPeriod,
)

async def main():
    limits = [
        MemoryBudgetLimit(name="daily", amount=5.0, period=LimitPeriod.DAILY),
    ]
    budget = LLMBudgetManager(limits)

    session = ChatSession(
        config=LLMConfig(provider="openai", model="gpt-5-mini"),
        system_prompt="Be concise.",
        budget_manager=budget,
    )

    async for chunk in session.stream("Summarize quantum computing in 3 sentences"):
        print(chunk.content, end="")
    print()

    available = await budget.available_budget_async()
    print(f"\nRemaining budget: {available:.4f}")

asyncio.run(main())
```

## LLMConfig Options

Key configuration fields:

| Field | Default | Description |
|---|---|---|
| `provider` | required | Provider name (`openai`, `anthropic`, `google`, etc.) |
| `model` | required | Model identifier |
| `temperature` | 0.7 | Response randomness (0.0 - 1.0) |
| `max_tokens` | 4096 | Maximum output tokens |
| `max_input_tokens` | 64000 | Maximum input context tokens |
| `reasoning_effort` | None | Reasoning effort level (LOW, MEDIUM, HIGH) |
| `mcp_servers` | None | List of MCP server configurations |

!!! note "Provider availability"
    Only providers with valid API keys in the environment are available at runtime. Set the appropriate `*_API_KEY` environment variable for each provider you want to use.
