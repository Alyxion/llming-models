"""Budget tracking example.

Shows how to set up budget limits and track LLM usage costs.
"""

import asyncio
from llming_models import MemoryBudgetLimit, LLMBudgetManager, LimitPeriod, InsufficientBudgetError


async def main():
    # Set up daily and monthly limits
    limits = [
        MemoryBudgetLimit(name="daily", amount=5.0, period=LimitPeriod.DAILY),
        MemoryBudgetLimit(name="monthly", amount=50.0, period=LimitPeriod.MONTHLY),
    ]
    manager = LLMBudgetManager(limits)

    print("=== Budget Tracking Demo ===\n")

    # Check available budget
    available = await manager.available_budget_async()
    print(f"Available budget: {available:.2f}EUR")

    # Simulate a Claude Sonnet request
    print("\n--- Reserving for Claude Sonnet request ---")
    await manager.reserve_budget_async(
        input_tokens=5000,
        max_output_tokens=4000,
        input_token_price=3.0,    # $3/1M input tokens
        output_token_price=15.0,  # $15/1M output tokens
    )
    available = await manager.available_budget_async()
    print(f"After reservation: {available:.4f}EUR")

    # Return unused tokens (actual output was shorter)
    await manager.return_unused_budget_async(
        reserved_output_tokens=4000,
        actual_output_tokens=800,
        output_token_price=15.0,
    )
    available = await manager.available_budget_async()
    print(f"After return: {available:.4f}EUR")

    # Try to exceed budget
    print("\n--- Testing budget limit ---")
    try:
        await manager.reserve_budget_async(
            input_tokens=1_000_000,
            max_output_tokens=500_000,
            input_token_price=3.0,
            output_token_price=15.0,
        )
    except InsufficientBudgetError as e:
        print(f"Budget exceeded: {e}")
        print(f"Limit that blocked: {e.limit_name}")


if __name__ == "__main__":
    asyncio.run(main())
