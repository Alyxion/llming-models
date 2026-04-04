# Google Gemini Provider

The Google provider supports Gemini 3 models with 1M token context windows, streaming, and multimodal input.

## Models

| Name | Model ID | Context | Input $/1M | Output $/1M | Features |
|---|---|---|---|---|---|
| Gemini 3 Pro | `gemini-3-pro-preview` | 1M | $2.00 | $12.00 | Reasoning, multimodal, code |
| Gemini 3 Flash | `gemini-3-flash-preview` | 1M | $0.50 | $3.00 | Fast, low cost |

## Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="google",
        model="gemini-3-pro-preview",
        temperature=0.7,
        max_tokens=4096,
    ),
)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Your Google AI API key |

## Key Features

### 1M Token Context

Both Gemini 3 models support 1M token input context -- the largest among supported providers. This makes them ideal for:

- Analyzing large codebases
- Processing long documents
- Multi-document Q&A

```python
config = LLMConfig(
    provider="google",
    model="gemini-3-pro-preview",
    max_input_tokens=1_000_000,  # use the full context
)
```

### Vision

Both models support image input alongside text:

```python
session = ChatSession(
    config=LLMConfig(provider="google", model="gemini-3-flash-preview"),
)
# Image support is built into the chat message format
```

### Reasoning

Both Gemini 3 models support reasoning mode for complex analytical tasks.

## Cached Input Pricing

| Model | Standard $/1M | Cached $/1M |
|---|---|---|
| Gemini 3 Pro | $2.00 | $0.20 |
| Gemini 3 Flash | $0.50 | $0.05 |

!!! tip "Cost optimization"
    Gemini 3 Flash at $0.50/1M input tokens is one of the most cost-effective models with a 1M context window. Use it for bulk processing and document analysis.

!!! note "Preview models"
    Gemini 3 models are currently in preview. Model IDs may change when they reach general availability.
