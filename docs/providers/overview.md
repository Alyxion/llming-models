# Provider Architecture

llming-models uses a pluggable provider system. Each provider implements the `BaseProvider` interface and registers itself for automatic discovery.

## BaseProvider Interface

Every provider extends `llming_models.providers.llm_provider_base.BaseProvider`:

```python
from llming_models.providers.llm_provider_base import BaseProvider

class BaseProvider(ABC):
    def __init__(self, name: str, label: str, credentials=None): ...

    @property
    def is_available(self) -> bool:
        """True when the provider has valid API credentials."""
        ...

    def get_models(self) -> list[LLMInfo]:
        """Return all models this provider supports."""
        ...

    def create_client(
        self, model: str, temperature: float = 0.7,
        max_tokens: int | None = None, streaming: bool = False,
        base_url: str | None = None, toolboxes: list | None = None,
    ) -> LlmClient:
        """Create a configured LLM client instance."""
        ...
```

## Available Providers

| Provider | Module | Env Variable | Models |
|---|---|---|---|
| OpenAI | `llming_models.providers.openai` | `OPENAI_API_KEY` | GPT-5.4, GPT-5.2, GPT-5-mini, GPT-5-nano |
| Anthropic | `llming_models.providers.anthropic` | `ANTHROPIC_API_KEY` | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| Google | `llming_models.providers.google` | `GOOGLE_API_KEY` | Gemini 3 Pro, Gemini 3 Flash |
| Mistral | `llming_models.providers.mistral` | `MISTRAL_API_KEY` | Large, Medium, Small |
| Together | `llming_models.providers.together` | `TOGETHER_API_KEY` | DeepSeek models |
| Azure OpenAI | `llming_models.providers.azure_openai` | `AZURE_OPENAI_API_KEY` | Azure-hosted OpenAI models |
| Azure Anthropic | `llming_models.providers.azure_anthropic` | `AZURE_ANTHROPIC_API_KEY` | Azure-hosted Claude models |

## Provider Cascade

The `LLMGlobalConfig.provider_cascade` defines the priority order for model resolution. When a model name appears across multiple providers, the first available provider wins:

```python
from llming_models import LLMGlobalConfig

config = LLMGlobalConfig(
    provider_cascade=["anthropic", "openai", "google", "mistral", "together"],
)
```

## Credential Resolution

Credentials are resolved in this order:

1. **Explicit credentials** -- Passed via `ProviderCredentials` to the provider constructor
2. **Environment variables** -- Each provider checks its standard env variable (e.g., `OPENAI_API_KEY`)

```python
from llming_models import LLMCredentials, ProviderCredentials

# Explicit credentials (useful for multi-tenant apps)
creds = LLMCredentials(providers={
    "openai": ProviderCredentials(api_key="sk-..."),
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
})

manager = LLMManager(credentials=creds)
```

## LLMManager

The `LLMManager` orchestrates providers and sessions:

```python
from llming_models import LLMManager

manager = LLMManager()

# List all available models across providers
models = manager.get_available_models()

# Create a chat session
session = manager.create_session(
    provider="anthropic",
    model="claude-sonnet-4-6",
    system_prompt="You are helpful.",
)
```

!!! tip "Provider compatibility"
    Azure providers are API-compatible with their base providers. Tools targeting `openai` automatically work with `azure_openai`, and `anthropic` tools work with `azure_anthropic`.
