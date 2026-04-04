# API Reference: Providers

## BaseProvider

`llming_models.providers.llm_provider_base.BaseProvider`

Abstract base class for all LLM providers.

### Methods

```python
class BaseProvider(ABC):
    def __init__(self, name: str, label: str, credentials=None): ...

    @property
    def is_available(self) -> bool:
        """True if the provider has valid API credentials."""

    def get_models(self) -> list[LLMInfo]:
        """Return all models supported by this provider."""

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        streaming: bool = False,
        base_url: str | None = None,
        toolboxes: list[LlmToolbox] | None = None,
        **kwargs,
    ) -> LlmClient:
        """Create a configured LLM client."""
```

## LLMManager

`llming_models.llm_provider_manager.LLMManager`

Orchestrates providers, model discovery, and session creation.

### Constructor

```python
LLMManager(
    user_config: LLMUserConfig | None = None,
    budget_manager: LLMBudgetManager | None = None,
    credentials: LLMCredentials | None = None,
)
```

### Key Methods

```python
# Get all available models across configured providers
manager.get_available_models() -> list[LLMInfo]

# Create a chat session
manager.create_session(
    provider: str,
    model: str,
    system_prompt: str | None = None,
) -> ChatSession
```

### Example

```python
from llming_models import LLMManager, LLMUserConfig, LLMGlobalConfig

manager = LLMManager(
    user_config=LLMUserConfig(
        global_config=LLMGlobalConfig(
            provider_cascade=["anthropic", "openai"],
        ),
    ),
)

for model in manager.get_available_models():
    print(f"{model.provider}:{model.name} - {model.label}")
```

## ChatSession

`llming_models.session.ChatSession`

High-level chat interface with streaming, tools, and budget tracking.

### Constructor

```python
ChatSession(
    config: LLMConfig,
    system_prompt: str | None = None,
    budget_manager: LLMBudgetManager | None = None,
    user_id: str | None = None,
    credentials: ProviderCredentials | None = None,
)
```

### Key Methods

```python
# Stream a response
async for chunk in session.stream(message: str):
    print(chunk.content, end="")
```

## LLMConfig

`llming_models.session.LLMConfig`

Pydantic model for session configuration.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | required | Provider name |
| `model` | `str` | required | Model identifier |
| `base_url` | `str \| None` | `None` | Custom API base URL |
| `temperature` | `float` | `0.7` | Response temperature |
| `max_tokens` | `int \| None` | `4096` | Max output tokens |
| `max_input_tokens` | `int \| None` | `64000` | Max input context |
| `reasoning_effort` | `ReasoningEffort \| None` | `None` | Reasoning level |
| `max_history_images` | `int` | `20` | Max active images in history |
| `condense_threshold_pct` | `float` | `0.80` | Context condensation trigger |
| `condense_model` | `str \| None` | `None` | Model for condensation |
| `condense_max_tokens` | `int` | `5000` | Max condensation output |
| `tools` | `list[str] \| None` | `None` | Enabled tool names |
| `tool_config` | `dict \| None` | `None` | Per-tool configuration |
| `mcp_servers` | `list[MCPServerConfig] \| None` | `None` | MCP server configs |

## Credentials

### ProviderCredentials

`llming_models.credentials.ProviderCredentials`

Credentials for a single provider.

### LLMCredentials

`llming_models.credentials.LLMCredentials`

Container for multi-provider credentials:

```python
from llming_models import LLMCredentials, ProviderCredentials

creds = LLMCredentials(providers={
    "openai": ProviderCredentials(api_key="sk-..."),
    "anthropic": ProviderCredentials(api_key="sk-ant-..."),
})

# Get credentials for a specific provider
openai_creds = creds.for_provider("openai")
```
