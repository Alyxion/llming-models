# Budget Management

llming-models provides a complete budget tracking system that enforces monetary limits on LLM operations with reservation/rollback semantics.

## Architecture

```
LLMBudgetManager
  -> BudgetLimit (abstract)
       -> MemoryBudgetLimit     (in-memory, resets on restart)
       -> MongoDBBudgetLimit    (persistent, requires pymongo)
```

## Quick Setup

```python
import asyncio
from llming_models import LLMBudgetManager, MemoryBudgetLimit, LimitPeriod

async def main():
    limits = [
        MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY),
        MemoryBudgetLimit(name="monthly", amount=100.0, period=LimitPeriod.MONTHLY),
    ]
    manager = LLMBudgetManager(limits)

    available = await manager.available_budget_async()
    print(f"Available: {available:.2f}")

asyncio.run(main())
```

## Time Periods

Budget limits support multiple time periods via `LimitPeriod` (aliased from `TimeInterval`):

| Period | Resets |
|---|---|
| `LimitPeriod.DAILY` | Every day at midnight |
| `LimitPeriod.WEEKLY` | Every Monday |
| `LimitPeriod.MONTHLY` | First of each month |

## Budget Scopes

| Scope | Description |
|---|---|
| `BudgetScope.GLOBAL` | Shared budget across all users |
| `BudgetScope.PER_USER` | Separate counters per `user_id` |

```python
from llming_models.budget import MemoryBudgetLimit, LimitPeriod, BudgetScope

per_user_limit = MemoryBudgetLimit(
    name="user_daily",
    amount=2.0,
    period=LimitPeriod.DAILY,
    scope=BudgetScope.PER_USER,
)
```

## Reserve / Return Flow

Budget tracking uses a reservation pattern:

1. **Reserve** -- Before an LLM call, reserve estimated cost (input + partial output)
2. **Execute** -- Run the LLM operation
3. **Return** -- After completion, return the unused portion

```python
# Reserve budget for an operation
reserved_output = await manager.reserve_budget_async(
    input_tokens=1000,
    max_output_tokens=4000,
    input_token_price=3.0,    # $ per 1M tokens
    output_token_price=15.0,  # $ per 1M tokens
    user_id="user-123",
)

# ... run LLM operation, get actual usage ...

await manager.return_unused_budget_async(
    reserved_output_tokens=reserved_output,
    actual_output_tokens=500,
    output_token_price=15.0,
    user_id="user-123",
)
```

### Conservative Reservation

The `reserve_output_ratio` controls how much of `max_output_tokens` is reserved upfront:

```python
# Default: 25% of max_output_tokens reserved
manager = LLMBudgetManager(limits, reserve_output_ratio=0.25)

# Full worst-case reservation (old behavior)
manager = LLMBudgetManager(limits, reserve_output_ratio=1.0)
```

!!! tip "Why 25%?"
    Reserving 100% of max output tokens often blocks valid requests. For example, a $2 remaining budget would block all Opus calls because worst-case reservation ($3.45) exceeds it. The 25% default covers ~95% of real responses.

## MongoDB Persistence

For production use, store budget data in MongoDB:

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

!!! note "Optional dependency"
    MongoDB support requires the `mongodb` extra: `pip install llming-models[mongodb]`

## Error Handling

```python
from llming_models.budget import InsufficientBudgetError

try:
    await manager.reserve_budget_async(
        input_tokens=50000,
        max_output_tokens=8000,
        input_token_price=5.0,
        output_token_price=25.0,
    )
except InsufficientBudgetError as e:
    print(f"Budget exceeded on limit: {e.limit_name}")
```
