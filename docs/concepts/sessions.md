# Chat Sessions

`ChatSession` is the primary interface for interacting with LLM providers. It manages conversation history, streaming, tool dispatch, budget tracking, and context condensation.

## Creating a Session

```python
from llming_models import ChatSession, LLMConfig

session = ChatSession(
    config=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=4096,
    ),
    system_prompt="You are a helpful coding assistant.",
)
```

## Streaming Responses

The primary method is async streaming, which yields message chunks:

```python
async for chunk in session.stream("Explain Python generators"):
    print(chunk.content, end="")
```

## LLMConfig

`LLMConfig` controls all session behavior:

```python
from llming_models import LLMConfig, ReasoningEffort

config = LLMConfig(
    # Required
    provider="openai",
    model="gpt-5.2",

    # Response control
    temperature=0.5,
    max_tokens=8192,
    max_input_tokens=64000,
    reasoning_effort=ReasoningEffort.MEDIUM,

    # History management
    max_history_images=20,
    condense_threshold_pct=0.80,
    condense_model=None,        # auto-select cheapest model
    condense_max_tokens=5000,

    # Tools
    tools=["web_search", "generate_image"],
    mcp_servers=None,
)
```

## Conversation History

`ChatSession` automatically manages conversation history via `ChatHistory`:

```python
# History is maintained internally
session.history  # ChatHistory instance

# Messages are typed
from llming_models import Role
for msg in session.history.messages:
    print(f"[{msg.role}] {msg.content[:80]}...")
```

## Budget Integration

Pass a budget manager to track and enforce spending limits:

```python
from llming_models import LLMBudgetManager, MemoryBudgetLimit, LimitPeriod

budget = LLMBudgetManager([
    MemoryBudgetLimit(name="daily", amount=5.0, period=LimitPeriod.DAILY),
])

session = ChatSession(
    config=LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
    budget_manager=budget,
    user_id="user-123",
)

# Budget is automatically reserved before each request
# and unused portions are returned after completion
```

!!! warning "InsufficientBudgetError"
    When the budget is exhausted, `session.stream()` raises `InsufficientBudgetError`. Handle it gracefully in your application.

## Context Condensation

When conversation history approaches the context limit, `ChatSession` automatically condenses it:

- Triggered when usage exceeds `condense_threshold_pct` (default: 80%)
- Uses `condense_model` (defaults to cheapest model from the same provider)
- Summary output capped at `condense_max_tokens` (default: 5000)

```python
config = LLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-6",
    condense_threshold_pct=0.75,     # condense at 75% context usage
    condense_model="claude-haiku-4-5-20251001",  # use Haiku for condensation
)
```

## Explicit Credentials

For multi-tenant applications, pass credentials directly instead of using environment variables:

```python
from llming_models import ProviderCredentials

session = ChatSession(
    config=LLMConfig(provider="openai", model="gpt-5-mini"),
    credentials=ProviderCredentials(api_key="sk-user-specific-key"),
)
```

!!! tip "Image support"
    ChatSession handles image encoding automatically. The `max_history_images` setting (default: 20) controls how many images are kept active -- older ones are flagged stale to save context space.
