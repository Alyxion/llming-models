"""Tests for llming_models.budget — budget management."""

import pytest
from llming_models.budget import (
    LimitPeriod,
    MemoryBudgetLimit,
    LLMBudgetManager,
    InsufficientBudgetError,
    TokenUsage,
    TimeInterval,
    TimeIntervalHandler,
)
from datetime import datetime


class TestTimeInterval:
    def test_values(self):
        assert TimeInterval.TOTAL.value == "total"
        assert TimeInterval.DAILY.value == "daily"
        assert TimeInterval.MONTHLY.value == "monthly"

    def test_key_suffix_total(self):
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.TOTAL, datetime.now()) == "total"

    def test_key_suffix_daily(self):
        dt = datetime(2026, 3, 21, 14, 30, 0)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.DAILY, dt) == "2026-03-21"

    def test_key_suffix_monthly(self):
        dt = datetime(2026, 3, 21)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.MONTHLY, dt) == "2026-03"

    def test_key_suffix_hourly(self):
        dt = datetime(2026, 3, 21, 14, 30, 0)
        assert TimeIntervalHandler.get_key_suffix(TimeInterval.HOURLY, dt) == "2026-03-21-14"

    def test_expiry_total(self):
        assert TimeIntervalHandler.get_expiry(TimeInterval.TOTAL) is None

    def test_expiry_daily(self):
        expiry = TimeIntervalHandler.get_expiry(TimeInterval.DAILY)
        assert expiry.days == 2


class TestTokenUsage:
    def test_totals(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50,
                          input_cost=0.01, output_cost=0.005)
        assert usage.total_tokens == 150
        assert usage.total_cost == 0.015


class TestMemoryBudgetLimit:
    @pytest.mark.asyncio
    async def test_total_budget(self):
        limit = MemoryBudgetLimit(name="test", amount=10.0, period=LimitPeriod.TOTAL)
        available = await limit.get_available_budget_async()
        assert available == 10.0

    @pytest.mark.asyncio
    async def test_reserve_and_return(self):
        limit = MemoryBudgetLimit(name="test", amount=10.0, period=LimitPeriod.TOTAL)
        assert await limit.reserve_budget_async(3.0) is True
        assert await limit.get_available_budget_async() == 7.0
        await limit.return_budget_async(3.0)
        assert await limit.get_available_budget_async() == 10.0

    @pytest.mark.asyncio
    async def test_reserve_exceeds_budget(self):
        limit = MemoryBudgetLimit(name="test", amount=5.0, period=LimitPeriod.TOTAL)
        assert await limit.reserve_budget_async(6.0) is False
        assert await limit.get_available_budget_async() == 5.0

    @pytest.mark.asyncio
    async def test_daily_budget(self):
        limit = MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY)
        assert await limit.reserve_budget_async(3.0) is True
        assert await limit.get_available_budget_async() == 7.0

    @pytest.mark.asyncio
    async def test_reset(self):
        limit = MemoryBudgetLimit(name="test", amount=10.0, period=LimitPeriod.TOTAL)
        await limit.reserve_budget_async(5.0)
        await limit.reset_async()
        assert await limit.get_available_budget_async() == 10.0


class TestLLMBudgetManager:
    @pytest.mark.asyncio
    async def test_available_budget(self):
        limits = [
            MemoryBudgetLimit(name="a", amount=10.0, period=LimitPeriod.TOTAL),
            MemoryBudgetLimit(name="b", amount=5.0, period=LimitPeriod.TOTAL),
        ]
        manager = LLMBudgetManager(limits)
        assert await manager.available_budget_async() == 5.0  # minimum

    @pytest.mark.asyncio
    async def test_reserve_budget(self):
        limits = [
            MemoryBudgetLimit(name="a", amount=10.0, period=LimitPeriod.TOTAL),
        ]
        manager = LLMBudgetManager(limits)
        await manager.reserve_budget_async(
            input_tokens=1000, max_output_tokens=1000,
            input_token_price=1.0, output_token_price=1.0,
        )
        available = await manager.available_budget_async()
        assert available < 10.0

    @pytest.mark.asyncio
    async def test_insufficient_budget_raises(self):
        limits = [
            MemoryBudgetLimit(name="tiny", amount=0.001, period=LimitPeriod.TOTAL),
        ]
        manager = LLMBudgetManager(limits)
        with pytest.raises(InsufficientBudgetError):
            await manager.reserve_budget_async(
                input_tokens=1_000_000, max_output_tokens=1_000_000,
                input_token_price=10.0, output_token_price=10.0,
            )

    @pytest.mark.asyncio
    async def test_return_unused(self):
        limits = [
            MemoryBudgetLimit(name="a", amount=10.0, period=LimitPeriod.TOTAL),
        ]
        manager = LLMBudgetManager(limits)
        await manager.reserve_budget_async(
            input_tokens=1000, max_output_tokens=2000,
            input_token_price=1.0, output_token_price=1.0,
        )
        budget_after_reserve = await manager.available_budget_async()
        await manager.return_unused_budget_async(
            reserved_output_tokens=2000, actual_output_tokens=500,
            output_token_price=1.0,
        )
        budget_after_return = await manager.available_budget_async()
        assert budget_after_return > budget_after_reserve

    @pytest.mark.asyncio
    async def test_reset(self):
        limits = [
            MemoryBudgetLimit(name="a", amount=10.0, period=LimitPeriod.TOTAL),
        ]
        manager = LLMBudgetManager(limits)
        await manager.reserve_budget_async(
            input_tokens=1000, max_output_tokens=1000,
            input_token_price=1.0, output_token_price=1.0,
        )
        await manager.reset_async()
        assert await manager.available_budget_async() == 10.0
