# Together AI Provider

The Together AI provider hosts open-weight models, currently featuring DeepSeek models for cost-effective inference.

## Models

Together AI serves DeepSeek models:

| Name | Model ID | Context | Input $/1M | Output $/1M | Features |
|---|---|---|---|---|---|
| DeepSeek | varies | varies | varies | varies | Cost-effective |

## Configuration

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="together",
        model="deepseek-chat",  # check available models
        temperature=0.7,
    ),
)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TOGETHER_API_KEY` | Yes | Your Together AI API key |

## Architecture

Together AI uses an OpenAI-compatible API, so the provider extends the generic OpenAI-compatible client:

```
llming_models.providers.together.together_provider
  -> llming_models.providers.openai_compat_client
```

This means Together AI supports the same streaming and tool-calling interface as OpenAI.

## Provider Cascade Position

Together AI is typically placed last in the provider cascade as a fallback:

```python
from llming_models import LLMGlobalConfig

config = LLMGlobalConfig(
    provider_cascade=[
        "anthropic", "openai", "google", "mistral", "together",
    ],
)
```

## When to Use Together AI

- **Cost optimization** -- Open-weight models at lower per-token prices
- **DeepSeek access** -- Run DeepSeek models via a managed API
- **Fallback provider** -- Last resort when primary providers are unavailable

!!! tip "Cost savings"
    Together AI can be significantly cheaper than first-party providers for open-weight models. Use it for high-volume workloads where cost matters more than maximum capability.

!!! note "Model availability"
    Together AI's model catalog changes frequently. Check their documentation for the latest available models and pricing.
