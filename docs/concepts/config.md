# Configuration

llming-models uses a layered configuration system with global defaults and per-user overrides.

## Configuration Hierarchy

```
LLMBaseConfig           # Base: include/exclude model filters
  -> LLMGlobalConfig    # Global: default models, provider cascade, budgets
  -> LLMUserConfig      # Per-user: overrides, personal defaults, budgets
```

## LLMBaseConfig

Base class with model filtering:

```python
from llming_models import LLMBaseConfig

config = LLMBaseConfig(
    included_models=["anthropic:*", "openai:gpt-5*"],  # glob patterns
    excluded_models=["openai:gpt-5-nano"],
)

config.is_model_supported("anthropic:claude_sonnet")  # True
config.is_model_supported("openai:gpt-5-nano")        # False
config.is_model_supported("mistral:mistral_large")     # False
```

## LLMGlobalConfig

Defines system-wide defaults:

```python
from llming_models import LLMGlobalConfig, ModelCategories

global_config = LLMGlobalConfig(
    # Default model per category (first available wins)
    default_models={
        ModelCategories.SMALL: ["claude_haiku", "gpt-5-nano"],
        ModelCategories.MEDIUM: ["claude_sonnet", "gpt-5-mini"],
        ModelCategories.LARGE: ["claude_sonnet", "gpt-5.2"],
        ModelCategories.REASONING_SMALL: ["claude_haiku", "gpt-5-mini"],
        ModelCategories.REASONING_MEDIUM: ["claude_sonnet", "gpt-5.2"],
        ModelCategories.REASONING_LARGE: ["claude_sonnet", "gpt-5.2"],
    },

    # Provider priority order
    provider_cascade=[
        "azure_openai", "openai",
        "azure_anthropic", "anthropic",
        "mistral", "google", "together",
    ],

    # Global budget limits
    budgets=[],

    # Bump to invalidate client-side cached preferences
    model_defaults_version=2,
)
```

## ModelCategories

Pre-defined model size categories:

| Category | Constant | Typical Use |
|---|---|---|
| Small | `ModelCategories.SMALL` | Quick tasks, classification |
| Medium | `ModelCategories.MEDIUM` | General purpose |
| Large | `ModelCategories.LARGE` | Complex analysis |
| Reasoning Small | `ModelCategories.REASONING_SMALL` | Light reasoning |
| Reasoning Medium | `ModelCategories.REASONING_MEDIUM` | Standard reasoning |
| Reasoning Large | `ModelCategories.REASONING_LARGE` | Deep reasoning |

## LLMUserConfig

Per-user overrides layered on top of global config:

```python
from llming_models import LLMUserConfig, LLMGlobalConfig

user_config = LLMUserConfig(
    global_config=global_config,

    # Override defaults for this user
    default_models={
        "large": "claude_opus",
    },

    # Additional filters
    included_models=["*"],
    excluded_models=["openai:gpt-5-nano"],

    # User-specific budgets
    budgets=[],

    # Template parameters for system prompts
    prompt_parameters={"username": "Alice"},
)

# Resolves through both user and global filters
user_config.is_model_supported("anthropic:claude_opus")  # True
user_config.get_default_model("large")                    # "claude_opus"
```

## Provider Cascade

The cascade determines which provider serves a model when multiple providers offer it:

```python
global_config = LLMGlobalConfig(
    provider_cascade=["anthropic", "openai", "google"],
)
```

When resolving `claude_sonnet`, the system tries `anthropic` first. If unavailable (no API key), it falls back through the cascade.

!!! tip "Pinning providers"
    Use `provider:model` format in `default_models` to pin a specific provider:

    ```python
    default_models={
        "large": "anthropic:claude_sonnet",  # always Anthropic
    }
    ```

## Using with LLMManager

The configuration flows into `LLMManager` for full provider orchestration:

```python
from llming_models import LLMManager

manager = LLMManager(user_config=user_config)

# Only available models that pass all filters are shown
models = manager.get_available_models()
```
