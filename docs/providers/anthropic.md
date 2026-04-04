# Anthropic Provider

The Anthropic provider supports Claude models with streaming, vision, reasoning, and web search.

## Models

| Name | Model ID | Context | Input $/1M | Output $/1M | Features |
|---|---|---|---|---|---|
| Claude Opus 4.6 | `claude-opus-4-6` | 200K | $5.00 | $25.00 | Best reasoning, code, analysis |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 200K | $3.00 | $15.00 | Flagship performance, balanced |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K | $1.00 | $5.00 | Fastest, near-frontier |

## Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=4096,
    ),
)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

## Native Web Search

Anthropic provides built-in web search powered by Brave Search:

```python
from llming_models.tools.tool_definition import get_web_search_tool_for_provider

tool = get_web_search_tool_for_provider("anthropic")
# Pricing: $10 per 1,000 searches ($0.01 per search)
# Default limit: 5 searches per request
```

## Reasoning

All Claude models support reasoning. Claude Opus 4.6 is the strongest reasoning model in the lineup:

```python
from llming_models import LLMConfig, ReasoningEffort

config = LLMConfig(
    provider="anthropic",
    model="claude-opus-4-6",
    reasoning_effort=ReasoningEffort.HIGH,
)
```

## Cached Input Pricing

Anthropic offers prompt caching for reduced costs on repeated prefixes:

| Model | Standard $/1M | Cached $/1M |
|---|---|---|
| Claude Opus 4.6 | $5.00 | $0.50 |
| Claude Sonnet 4.6 | $3.00 | $0.30 |
| Claude Haiku 4.5 | $1.00 | $0.10 |

## Vision

All Claude models support image input:

```python
session = ChatSession(
    config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
)

# Images can be passed in chat messages
# The session handles encoding and format conversion automatically
```

!!! tip "Model selection"
    Use **Sonnet 4.6** for most tasks -- it offers the best price/performance ratio. Reserve **Opus 4.6** for complex reasoning tasks. Use **Haiku 4.5** for high-volume, latency-sensitive workloads.
