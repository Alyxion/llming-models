# API Reference: Models

## LLMInfo

`llming_models.model_info.LLMInfo`

Rich model descriptor with pricing, capabilities, and UI metadata.

### Fields

| Field | Type | Description |
|---|---|---|
| `provider` | `str` | Provider name (`openai`, `anthropic`, etc.) |
| `name` | `str` | Internal model identifier |
| `label` | `str` | Human-readable display name |
| `model` | `str` | API model identifier |
| `description` | `str` | Model description |
| `input_token_price` | `float` | Price per 1M input tokens (USD) |
| `output_token_price` | `float` | Price per 1M output tokens (USD) |
| `cached_input_token_price` | `float \| None` | Cached input price per 1M tokens |
| `size` | `ModelSize` | Model size category |
| `max_input_tokens` | `int` | Maximum input context window |
| `max_output_tokens` | `int` | Maximum output tokens |
| `supports_image_input` | `bool` | Vision capability |
| `reasoning` | `bool` | Reasoning capability |
| `default_reasoning_effort` | `ReasoningEffort \| None` | Default reasoning level |
| `speed` | `int` | Speed rating (1-10) |
| `quality` | `int` | Quality rating (1-10) |
| `popularity` | `int` | Popularity score |
| `best_use` | `str` | Recommended use case |
| `highlights` | `list[str]` | Feature highlight tags |

### Example

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
```

## ModelSize

`llming_models.model_info.ModelSize`

Enum for model size categories:

| Value | Description |
|---|---|
| `ModelSize.SMALL` | Small/fast models |
| `ModelSize.MEDIUM` | Balanced models |
| `ModelSize.LARGE` | Largest/most capable |

## ReasoningEffort

`llming_models.model_info.ReasoningEffort`

Controls reasoning depth for models that support it:

| Value | Description |
|---|---|
| `ReasoningEffort.LOW` | Minimal reasoning |
| `ReasoningEffort.MEDIUM` | Balanced reasoning |
| `ReasoningEffort.HIGH` | Maximum reasoning depth |

## ModelCategories

`llming_models.model_categories.ModelCategories`

Constants for model category selection:

```python
from llming_models import ModelCategories

ModelCategories.SMALL             # "small"
ModelCategories.MEDIUM            # "medium"
ModelCategories.LARGE             # "large"
ModelCategories.REASONING_SMALL   # "reasoning_small"
ModelCategories.REASONING_MEDIUM  # "reasoning_medium"
ModelCategories.REASONING_LARGE   # "reasoning_large"
```
