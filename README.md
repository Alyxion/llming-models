# LLMing-Models

Model metadata, configuration, and budget management for LLM applications.

LLMing-Models provides the foundational types for building multi-provider LLM applications. It defines model metadata (capabilities, pricing, context windows), configuration (model selection, filtering, defaults), and budget management (cost tracking, reservation, enforcement) — all with zero external dependencies.

## Features

### Model Metadata

Describe LLM models with rich, structured metadata.

- **`LLMInfo`**: Dataclass capturing everything about a model — provider, pricing (input/output/cached per 1M tokens), context window sizes, capability flags (vision, reasoning), UI metadata (speed/quality ratings, highlights), and tool configuration.
- **`ModelSize`**: Size categories (`VERY_SMALL` through `VERY_LARGE`) for grouping models.
- **`ReasoningEffort`**: Effort levels (`NONE`, `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`) for models with configurable thinking depth.

### Configuration

Flexible model selection and filtering at global and user levels.

- **`LLMGlobalConfig`**: System-wide defaults — model category mappings (small/medium/large/reasoning), provider cascade priority, include/exclude filters via fnmatch globs.
- **`LLMUserConfig`**: Per-user overrides — inherits from global config, user can pin specific models per category. Supports prompt parameters.
- **`ModelCategories`**: Standard category constants (`SMALL`, `MEDIUM`, `LARGE`, `REASONING_SMALL`, `REASONING_MEDIUM`, `REASONING_LARGE`).

### Budget Management

Track and enforce monetary limits on LLM usage.

- **`BudgetLimit`** (abstract): Base class for budget backends. Supports named limits with configurable time periods and timezone-aware key generation.
- **`MemoryBudgetLimit`**: Thread-safe in-memory implementation with per-period tracking. Suitable for single-process applications and testing.
- **`LLMBudgetManager`**: Coordinates multiple budget limits — checks all limits before reserving, rolls back on failure, returns unused budget after operations complete. Calculates costs from per-million-token pricing.
- **`TimeInterval`**: Period types (`TOTAL`, `YEARLY`, `MONTHLY`, `DAILY`, `HOURLY`, `MINUTES`, `SECONDS`) with bucketed key generation and expiry calculation.
- **`TokenUsage`**: Token count + cost tracking for completed operations.
- **`InsufficientBudgetError`**: Raised when an operation exceeds available budget.

### Zero Dependencies

The entire package uses only Python stdlib — no `pydantic`, no `tiktoken`, no provider SDKs. This means downstream applications can query model metadata and manage budgets without pulling in heavy ML dependencies.

## Installation

```bash
pip install llming-models
```

Or from source with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/Alyxion/llming-models.git
cd llming-models
poetry install
```

## Quick Start

### Model Metadata

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

print(f"{model.label}: ${model.input_token_price}/1M in, ${model.output_token_price}/1M out")
```

### Configuration

```python
from llming_models import LLMGlobalConfig, LLMUserConfig, ModelCategories

global_config = LLMGlobalConfig(
    default_models={
        ModelCategories.SMALL: ["claude_haiku", "gpt-4o-mini"],
        ModelCategories.LARGE: ["claude_sonnet", "gpt-4o"],
    },
    provider_cascade=["anthropic", "openai"],
)

user_config = LLMUserConfig(
    global_config=global_config,
    default_models={ModelCategories.LARGE: "claude_sonnet"},
)

print(user_config.get_default_model(ModelCategories.LARGE))  # "claude_sonnet"
print(user_config.is_model_supported("anthropic:claude_sonnet"))  # True
```

### Budget Management

```python
import asyncio
from llming_models import MemoryBudgetLimit, LLMBudgetManager, LimitPeriod

async def main():
    limits = [
        MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY),
        MemoryBudgetLimit(name="monthly", amount=100.0, period=LimitPeriod.MONTHLY),
    ]
    manager = LLMBudgetManager(limits)

    available = await manager.available_budget_async()
    print(f"Available: {available:.2f}€")

    await manager.reserve_budget_async(
        input_tokens=1000,
        max_output_tokens=2000,
        input_token_price=3.0,    # per 1M tokens
        output_token_price=15.0,  # per 1M tokens
    )

    await manager.return_unused_budget_async(
        reserved_output_tokens=2000,
        actual_output_tokens=500,
        output_token_price=15.0,
    )

asyncio.run(main())
```

## Architecture

```
llming_models/
├── __init__.py              # Public API exports
├── model_info.py            # LLMInfo, ModelSize, ReasoningEffort
├── model_categories.py      # ModelCategories constants
├── config.py                # LLMBaseConfig, LLMGlobalConfig, LLMUserConfig
└── budget/
    ├── __init__.py           # Budget subpackage exports
    ├── budget_types.py       # TokenUsage, InsufficientBudgetError, LimitPeriod
    ├── time_intervals.py     # TimeInterval, TimeIntervalHandler
    ├── budget_limit.py       # BudgetLimit (abstract base)
    ├── budget_manager.py     # LLMBudgetManager (multi-limit coordinator)
    └── memory_budget_limit.py # MemoryBudgetLimit (in-memory implementation)
```

## Running Tests

```bash
poetry install
poetry run pytest
```

## License

**LLMing-Models** is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
