# API Reference: Budget

## LLMBudgetManager

`llming_models.budget.budget_manager.LLMBudgetManager`

Manages multiple budget limits with reservation/rollback semantics.

### Constructor

```python
LLMBudgetManager(
    limits: list[BudgetLimit],
    *,
    reserve_output_ratio: float = 0.25,  # fraction of max_output to reserve
)
```

### Methods

```python
# Check available budget (minimum across all limits)
await manager.available_budget_async(user_id: str | None = None) -> float

# Reserve budget before an LLM call
await manager.reserve_budget_async(
    *,
    input_tokens: int,
    max_output_tokens: int,
    input_token_price: float,     # USD per 1M tokens
    output_token_price: float,    # USD per 1M tokens
    user_id: str | None = None,
) -> int  # returns reserved output token count

# Return unused budget after completion
await manager.return_unused_budget_async(
    *,
    reserved_output_tokens: int,
    actual_output_tokens: int,
    output_token_price: float,
    user_id: str | None = None,
) -> None
```

## BudgetLimit

`llming_models.budget.budget_limit.BudgetLimit`

Abstract base class for budget limits.

### Constructor

```python
BudgetLimit(
    *,
    name: str,
    amount: float,                           # Budget amount in currency
    period: LimitPeriod,                     # Time period
    interval_value: int | None = None,       # Custom interval
    timezone_str: str = "UTC",
    scope: BudgetScope = BudgetScope.GLOBAL,
)
```

### Abstract Methods

```python
await limit.get_available_budget_async(user_id=None) -> float
await limit.reserve_budget_async(amount, user_id=None) -> bool
await limit.return_budget_async(amount, user_id=None) -> None
await limit.reset_async() -> None

# Optional: log usage after completion
await limit.log_usage_async(
    model_name=..., tokens_input=..., tokens_output=...,
    costs=..., duration_ms=..., user_id=...,
) -> None
```

## MemoryBudgetLimit

`llming_models.budget.memory_budget_limit.MemoryBudgetLimit`

In-memory budget tracking. Resets when the process restarts.

```python
from llming_models import MemoryBudgetLimit, LimitPeriod

limit = MemoryBudgetLimit(
    name="daily",
    amount=10.0,
    period=LimitPeriod.DAILY,
)
```

## MongoDBBudgetLimit

`llming_models.budget.mongodb_budget_limit.MongoDBBudgetLimit`

Persistent budget tracking backed by MongoDB. Requires `pip install llming-models[mongodb]`.

```python
from llming_models.budget import MongoDBBudgetLimit, LimitPeriod

limit = MongoDBBudgetLimit(
    name="monthly",
    amount=100.0,
    period=LimitPeriod.MONTHLY,
    mongo_uri="mongodb://localhost:27017",
    database="llming",
)
```

## Enums

### LimitPeriod

`llming_models.budget.budget_types.LimitPeriod`

Alias for `TimeInterval`:

| Value | Description |
|---|---|
| `LimitPeriod.DAILY` | Resets daily |
| `LimitPeriod.WEEKLY` | Resets weekly |
| `LimitPeriod.MONTHLY` | Resets monthly |

### BudgetScope

`llming_models.budget.budget_types.BudgetScope`

| Value | Description |
|---|---|
| `BudgetScope.GLOBAL` | Shared across all users |
| `BudgetScope.PER_USER` | Separate per user_id |

## Exceptions

### InsufficientBudgetError

`llming_models.budget.budget_types.InsufficientBudgetError`

Raised when budget is exhausted.

```python
from llming_models.budget import InsufficientBudgetError

try:
    await manager.reserve_budget_async(...)
except InsufficientBudgetError as e:
    print(f"Limit exceeded: {e.limit_name}")
```

## TokenUsage

`llming_models.budget.budget_types.TokenUsage`

Tracks token consumption for a single operation:

```python
from llming_models.budget import TokenUsage

usage = TokenUsage(
    input_tokens=1500,
    output_tokens=800,
    input_cost=0.0045,
    output_cost=0.012,
)

usage.total_tokens  # 2300
usage.total_cost    # 0.0165
```
