# OpenAI Provider

The OpenAI provider supports GPT-5.x models with streaming, vision, reasoning, and native tool support.

## Models

| Name | Model ID | Context | Input $/1M | Output $/1M | Features |
|---|---|---|---|---|---|
| GPT-5.4 (Preview) | `gpt-5.4` | 272K | $2.50 | $15.00 | Reasoning, vision, web search |
| GPT-5.4 Large Ctx | `gpt-5.4` | 1M | $5.00 | $22.50 | 1M context window |
| GPT-5.2 | `gpt-5.2` | 128K | $1.25 | $5.00 | Reasoning, vision |
| GPT-5-mini | `gpt-5-mini` | 128K | $0.15 | $0.60 | Fast, cost-effective |
| GPT-5-nano | `gpt-5-nano` | 128K | $0.075 | $0.30 | Fastest, cheapest |

## Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="openai",
        model="gpt-5.2",
        temperature=0.7,
        max_tokens=4096,
    ),
)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |

## Native Tools

OpenAI provides built-in tools that are handled server-side:

### Web Search

```python
from llming_models.tools.tool_definition import get_web_search_tool_for_provider

tool = get_web_search_tool_for_provider("openai")
# Automatically enabled for models with default_tools=["web_search"]
```

### Image Generation

GPT Image support is available as a built-in tool with configurable size and quality:

```python
from llming_models.tools.tool_definition import DEFAULT_IMAGE_GENERATION_TOOL
# Fixed cost: ~$0.042 per generation (medium quality, 1024x1024)
```

## Reasoning

GPT-5.x models support reasoning with configurable effort levels:

```python
from llming_models import LLMConfig, ReasoningEffort

config = LLMConfig(
    provider="openai",
    model="gpt-5.4",
    reasoning_effort=ReasoningEffort.HIGH,  # LOW, MEDIUM, HIGH
)
```

!!! note "Temperature"
    GPT-5.4 enforces `temperature=1.0` regardless of the configured value. Other GPT-5.x models respect the temperature setting.

## Cached Input Pricing

OpenAI offers reduced pricing for cached input tokens (repeated prefixes):

| Model | Standard $/1M | Cached $/1M |
|---|---|---|
| GPT-5.4 | $2.50 | $0.25 |
| GPT-5.2 | $1.25 | $0.125 |
| GPT-5-mini | $0.15 | $0.015 |
