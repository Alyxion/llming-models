# Mistral Provider

The Mistral provider offers EU-hosted models with strong multilingual capabilities.

## Models

| Name | Model ID | Context | Input $/1M | Output $/1M | Features |
|---|---|---|---|---|---|
| Mistral Large | `mistral-large-latest` | 128K | $2.00 | $6.00 | Reasoning, multilingual |
| Mistral Medium | `mistral-medium-latest` | 32K | $0.50 | $1.50 | Balanced |
| Mistral Small | `mistral-small-latest` | 16K | $0.20 | $0.60 | Fast, low cost |

## Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="mistral",
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=4096,
    ),
)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Your Mistral API key |

## Key Features

### EU Hosting

All Mistral models are hosted in the EU, making them suitable for applications with data residency requirements.

### Multilingual Support

Mistral models excel at multilingual tasks, particularly European languages.

### Model Selection Guide

| Use Case | Recommended Model |
|---|---|
| Complex reasoning, analysis | Mistral Large |
| General-purpose tasks | Mistral Medium |
| Classification, extraction | Mistral Small |
| High-volume processing | Mistral Small |

## Output Pricing Comparison

Mistral offers competitive output pricing compared to frontier models:

| Model | Output $/1M |
|---|---|
| Mistral Large | $6.00 |
| Claude Sonnet 4.6 | $15.00 |
| GPT-5.2 | $5.00 |
| Claude Opus 4.6 | $25.00 |

!!! tip "When to use Mistral"
    Choose Mistral when you need EU data residency, strong multilingual support, or cost-effective output pricing ($6/1M for the large model vs. $15-25 for frontier competitors).

!!! note "Context limits"
    Mistral context windows (16K-128K) are smaller than some competitors. For large document processing, consider Google Gemini (1M context) instead.
