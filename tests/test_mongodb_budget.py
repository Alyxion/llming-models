"""Comprehensive tests for MongoDBBudgetLimit with fully mocked pymongo.

Uses unittest.mock.AsyncMock to simulate the async MongoDB collection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.budget.budget_types import BudgetScope
from llming_models.budget.time_intervals import TimeInterval
from llming_models.budget.mongodb_budget_limit import MongoDBBudgetLimit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limit(
    *,
    name: str = "test_limit",
    amount: float = 100.0,
    period: TimeInterval = TimeInterval.MONTHLY,
    mongo_uri: str = "mongodb://localhost:27017",
    mongo_db: str = "test_db",
    mongo_collection: str = "budgets",
    interval_value: int | None = None,
    timezone_str: str = "UTC",
    enable_logging: bool = False,
    user_id: str | None = None,
    scope: BudgetScope = BudgetScope.GLOBAL,
) -> MongoDBBudgetLimit:
    """Create a MongoDBBudgetLimit with sensible defaults for testing."""
    return MongoDBBudgetLimit(
        name=name,
        amount=amount,
        period=period,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        interval_value=interval_value,
        timezone_str=timezone_str,
        enable_logging=enable_logging,
        user_id=user_id,
        scope=scope,
    )


def _mock_coll() -> AsyncMock:
    """Create a fresh AsyncMock collection."""
    coll = AsyncMock()
    return coll


def _patch_coll(limit: MongoDBBudgetLimit, coll: AsyncMock) -> None:
    """Patch the limit to use a mock collection instead of real MongoDB."""
    limit._async_coll = coll


def _fixed_time() -> datetime:
    """Return a fixed UTC datetime for deterministic tests."""
    return datetime(2024, 6, 15, 14, 30, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestMongoDBBudgetLimitInit:
    """Constructor validation and field storage."""

    def test_valid_params(self) -> None:
        limit = _make_limit(name="my_limit", amount=50.0)
        assert limit.name == "my_limit"
        assert limit.amount == 50.0
        assert limit.mongo_uri == "mongodb://localhost:27017"
        assert limit.mongo_db == "test_db"
        assert limit.mongo_collection == "budgets"
        assert limit.enable_logging is False
        assert limit.user_id is None
        assert limit._default_amount == 50.0
        assert limit._async_coll is None

    def test_empty_mongo_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="MongoDB URI"):
            _make_limit(mongo_uri="")

    def test_empty_mongo_db_raises(self) -> None:
        with pytest.raises(ValueError, match="MongoDB URI"):
            _make_limit(mongo_db="")

    def test_empty_mongo_collection_raises(self) -> None:
        with pytest.raises(ValueError, match="MongoDB URI"):
            _make_limit(mongo_collection="")

    def test_stores_scope(self) -> None:
        limit = _make_limit(scope=BudgetScope.PER_USER, user_id="u1")
        assert limit.scope == BudgetScope.PER_USER

    def test_stores_enable_logging(self) -> None:
        limit = _make_limit(enable_logging=True)
        assert limit.enable_logging is True

    def test_stores_user_id(self) -> None:
        limit = _make_limit(user_id="user-123")
        assert limit.user_id == "user-123"

    def test_stores_interval_value(self) -> None:
        limit = _make_limit(interval_value=3)
        assert limit.interval_value == 3

    def test_non_int_interval_value_set_to_none(self) -> None:
        limit = _make_limit(interval_value=None)
        assert limit.interval_value is None


# ---------------------------------------------------------------------------
# _get_mongo_key
# ---------------------------------------------------------------------------

class TestGetMongoKey:
    """MongoDB document key generation."""

    def test_global_scope(self) -> None:
        limit = _make_limit(name="global_limit", period=TimeInterval.MONTHLY)
        key = limit._get_mongo_key(time=_fixed_time())
        assert key == {"name": "global_limit", "period": "monthly"}

    def test_per_user_scope_with_user_id(self) -> None:
        limit = _make_limit(
            name="user_limit",
            period=TimeInterval.DAILY,
            scope=BudgetScope.PER_USER,
            user_id="user-abc",
        )
        key = limit._get_mongo_key(time=_fixed_time())
        assert key == {"name": "user_limit", "period": "daily", "user_id": "user-abc"}

    def test_per_user_scope_with_explicit_user_id(self) -> None:
        limit = _make_limit(
            name="user_limit",
            period=TimeInterval.DAILY,
            scope=BudgetScope.PER_USER,
            user_id="default-user",
        )
        key = limit._get_mongo_key(time=_fixed_time(), user_id="override-user")
        assert key["user_id"] == "override-user"

    def test_per_user_scope_no_user_id_raises(self) -> None:
        limit = _make_limit(
            scope=BudgetScope.PER_USER,
            user_id=None,
        )
        with pytest.raises(ValueError, match="user_id is required"):
            limit._get_mongo_key(time=_fixed_time())

    def test_global_scope_does_not_include_user_id(self) -> None:
        limit = _make_limit(user_id="some-user")  # global scope by default
        key = limit._get_mongo_key(time=_fixed_time())
        assert "user_id" not in key


# ---------------------------------------------------------------------------
# _get_usage_path
# ---------------------------------------------------------------------------

class TestGetUsagePath:
    """Usage path generation for nested MongoDB documents."""

    def test_monthly_path(self) -> None:
        limit = _make_limit(period=TimeInterval.MONTHLY)
        path = limit._get_usage_path(time=_fixed_time())
        assert path == ["usage", "2024-06"]

    def test_daily_path(self) -> None:
        limit = _make_limit(period=TimeInterval.DAILY)
        path = limit._get_usage_path(time=_fixed_time())
        assert path == ["usage", "2024-06-15"]

    def test_total_path(self) -> None:
        limit = _make_limit(period=TimeInterval.TOTAL)
        path = limit._get_usage_path(time=_fixed_time())
        assert path == ["usage", "total"]

    def test_hourly_path(self) -> None:
        limit = _make_limit(period=TimeInterval.HOURLY)
        path = limit._get_usage_path(time=_fixed_time())
        assert path == ["usage", "2024-06-15-14"]


# ---------------------------------------------------------------------------
# _get_effective_amount
# ---------------------------------------------------------------------------

class TestGetEffectiveAmount:
    """Effective amount: stored override vs constructor default."""

    @pytest.mark.asyncio
    async def test_returns_stored_amount(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = {"amount": 200.0}
        result = await limit._get_effective_amount()
        assert result == 200.0

    @pytest.mark.asyncio
    async def test_returns_default_when_no_stored(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        result = await limit._get_effective_amount()
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_returns_default_when_doc_has_no_amount(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = {"_id": "something"}
        result = await limit._get_effective_amount()
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_returns_default_on_exception(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.side_effect = Exception("DB error")
        result = await limit._get_effective_amount()
        assert result == 100.0


# ---------------------------------------------------------------------------
# get_available_budget_async
# ---------------------------------------------------------------------------

class TestGetAvailableBudgetAsync:
    """Available budget computation."""

    @pytest.mark.asyncio
    async def test_full_budget_when_no_usage(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.MONTHLY)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # No document found -> no usage
        coll.find_one.return_value = None
        result = await limit.get_available_budget_async()
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_budget_minus_usage(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # Simulate document with nested usage
        coll.find_one.return_value = {
            "usage": {"total": {"used": 30.0}},
            "amount": 100.0,
        }
        result = await limit.get_available_budget_async()
        assert result == 70.0

    @pytest.mark.asyncio
    async def test_budget_never_negative(self) -> None:
        limit = _make_limit(amount=10.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = {
            "usage": {"total": {"used": 50.0}},
            "amount": 10.0,
        }
        result = await limit.get_available_budget_async()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_uses_stored_amount_override(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # First call for get_available_budget_async main query
        # Second call for _get_effective_amount
        coll.find_one.return_value = {
            "usage": {"total": {"used": 10.0}},
            "amount": 200.0,  # override
        }
        result = await limit.get_available_budget_async()
        assert result == 190.0

    @pytest.mark.asyncio
    async def test_db_error_raises_runtime_error(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.side_effect = Exception("connection failed")
        with pytest.raises(RuntimeError, match="Failed to get available budget"):
            await limit.get_available_budget_async()


# ---------------------------------------------------------------------------
# reserve_budget_async
# ---------------------------------------------------------------------------

class TestReserveBudgetAsync:
    """Budget reservation with atomic MongoDB operations."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # _get_effective_amount
        coll.find_one.return_value = None  # no stored override
        # find_one_and_update returns the updated document
        coll.find_one_and_update.return_value = {
            "usage": {"total": {"used": 10.0}},
        }
        result = await limit.reserve_budget_async(10.0)
        assert result is True
        coll.find_one_and_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_exceeds_total_amount(self) -> None:
        limit = _make_limit(amount=5.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        # Trying to reserve more than total
        result = await limit.reserve_budget_async(10.0)
        assert result is False
        # find_one_and_update should NOT be called
        coll.find_one_and_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_exceeds_limit_after_increment_rolls_back(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        # After increment, used exceeds amount
        coll.find_one_and_update.return_value = {
            "usage": {"total": {"used": 150.0}},
        }
        result = await limit.reserve_budget_async(50.0)
        assert result is False
        # Rollback: update_one called with negative inc
        coll.update_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_error_raises_runtime_error(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        coll.find_one_and_update.side_effect = Exception("DB write failed")
        with pytest.raises(RuntimeError, match="Failed to reserve budget"):
            await limit.reserve_budget_async(10.0)

    @pytest.mark.asyncio
    async def test_per_user_reservation(self) -> None:
        limit = _make_limit(
            amount=50.0,
            period=TimeInterval.DAILY,
            scope=BudgetScope.PER_USER,
            user_id="user-1",
        )
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        coll.find_one_and_update.return_value = {
            "usage": {"2024-06-15": {"used": 5.0}},
        }
        result = await limit.reserve_budget_async(5.0)
        assert result is True


# ---------------------------------------------------------------------------
# return_budget_async
# ---------------------------------------------------------------------------

class TestReturnBudgetAsync:
    """Budget return (decrement usage)."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one_and_update.return_value = {
            "usage": {"total": {"used": 5.0}},
        }
        await limit.return_budget_async(10.0)
        coll.find_one_and_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_negative_guard_resets_to_zero(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # After decrement, used goes below 0
        coll.find_one_and_update.return_value = {
            "usage": {"total": {"used": -5.0}},
        }
        await limit.return_budget_async(10.0)
        # Should reset to 0
        coll.update_one.assert_called_once()
        call_args = coll.update_one.call_args
        assert "$set" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_reset_when_usage_positive(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one_and_update.return_value = {
            "usage": {"total": {"used": 20.0}},
        }
        await limit.return_budget_async(5.0)
        coll.update_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_error_raises_runtime_error(self) -> None:
        limit = _make_limit(amount=100.0, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one_and_update.side_effect = Exception("DB error")
        with pytest.raises(RuntimeError, match="Failed to return budget"):
            await limit.return_budget_async(5.0)


# ---------------------------------------------------------------------------
# reset_async
# ---------------------------------------------------------------------------

class TestResetAsync:
    """Budget reset removes all usage records."""

    @pytest.mark.asyncio
    async def test_deletes_documents(self) -> None:
        limit = _make_limit(name="my_budget", period=TimeInterval.MONTHLY)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.reset_async()
        coll.delete_many.assert_called_once_with({
            "name": "my_budget",
            "period": "monthly",
        })

    @pytest.mark.asyncio
    async def test_db_error_raises_runtime_error(self) -> None:
        limit = _make_limit()
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.delete_many.side_effect = Exception("DB error")
        with pytest.raises(RuntimeError, match="Failed to reset budget"):
            await limit.reset_async()


# ---------------------------------------------------------------------------
# set_amount_async
# ---------------------------------------------------------------------------

class TestSetAmountAsync:
    """Persist budget amount override."""

    @pytest.mark.asyncio
    async def test_persists_new_amount(self) -> None:
        limit = _make_limit(amount=100.0)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.set_amount_async(200.0)
        coll.update_one.assert_called_once()
        call_args = coll.update_one.call_args
        assert call_args[0][1] == {"$set": {"amount": 200.0}}
        assert limit.amount == 200.0

    @pytest.mark.asyncio
    async def test_upserts(self) -> None:
        limit = _make_limit()
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.set_amount_async(50.0)
        call_kwargs = coll.update_one.call_args
        assert call_kwargs[1]["upsert"] is True


# ---------------------------------------------------------------------------
# get_usage_async
# ---------------------------------------------------------------------------

class TestGetUsageAsync:
    """Get current usage for the period."""

    @pytest.mark.asyncio
    async def test_returns_usage_from_nested_dict(self) -> None:
        limit = _make_limit(period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = {
            "usage": {"total": {"used": 42.5}},
        }
        result = await limit.get_usage_async()
        assert result == 42.5

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_document(self) -> None:
        limit = _make_limit(period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = None
        result = await limit.get_usage_async()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_usage_key(self) -> None:
        limit = _make_limit(period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.return_value = {"_id": "something"}
        result = await limit.get_usage_async()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_exception(self) -> None:
        limit = _make_limit()
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.find_one.side_effect = Exception("DB error")
        result = await limit.get_usage_async()
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_returns_numeric_usage_directly(self) -> None:
        """When usage path resolves to a number instead of dict with 'used' key."""
        limit = _make_limit(period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        # Some edge case where the value is stored directly as a number
        coll.find_one.return_value = {
            "usage": {"total": 15.5},
        }
        result = await limit.get_usage_async()
        assert result == 15.5


# ---------------------------------------------------------------------------
# log_usage_async
# ---------------------------------------------------------------------------

class TestLogUsageAsync:
    """Usage logging with enable_logging flag."""

    @pytest.mark.asyncio
    async def test_logging_enabled_pushes_entry(self) -> None:
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="claude-sonnet",
            tokens_input=1000,
            tokens_output=500,
            costs=0.05,
        )
        coll.update_one.assert_called_once()
        call_args = coll.update_one.call_args
        update_op = call_args[0][1]
        assert "$push" in update_op

    @pytest.mark.asyncio
    async def test_logging_disabled_is_noop(self) -> None:
        limit = _make_limit(enable_logging=False)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="claude-sonnet",
            tokens_input=1000,
            tokens_output=500,
            costs=0.05,
        )
        coll.update_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_logging_includes_optional_fields(self) -> None:
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL, user_id="user-1")
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="gpt-5",
            tokens_input=2000,
            tokens_output=1000,
            costs=0.10,
            duration_ms=150.0,
            operation_type="chat",
        )
        coll.update_one.assert_called_once()
        call_args = coll.update_one.call_args
        update_op = call_args[0][1]
        # Extract the pushed log entry
        push_field = list(update_op["$push"].keys())[0]
        log_entry = update_op["$push"][push_field]
        assert log_entry["model"] == "gpt-5"
        assert log_entry["tokens_input"] == 2000
        assert log_entry["tokens_output"] == 1000
        assert log_entry["costs"] == 0.10
        assert log_entry["duration_ms"] == 150.0
        assert log_entry["operation_type"] == "chat"
        assert log_entry["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_logging_user_id_override(self) -> None:
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL, user_id="default")
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="test",
            tokens_input=100,
            tokens_output=50,
            costs=0.01,
            user_id="override-user",
        )
        call_args = coll.update_one.call_args
        update_op = call_args[0][1]
        push_field = list(update_op["$push"].keys())[0]
        log_entry = update_op["$push"][push_field]
        assert log_entry["user_id"] == "override-user"

    @pytest.mark.asyncio
    async def test_logging_no_user_id(self) -> None:
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL, user_id=None)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="test",
            tokens_input=100,
            tokens_output=50,
            costs=0.01,
        )
        call_args = coll.update_one.call_args
        update_op = call_args[0][1]
        push_field = list(update_op["$push"].keys())[0]
        log_entry = update_op["$push"][push_field]
        assert "user_id" not in log_entry

    @pytest.mark.asyncio
    async def test_logging_omits_none_optional_fields(self) -> None:
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        await limit.log_usage_async(
            model_name="test",
            tokens_input=100,
            tokens_output=50,
            costs=0.01,
            duration_ms=None,
            operation_type=None,
        )
        call_args = coll.update_one.call_args
        update_op = call_args[0][1]
        push_field = list(update_op["$push"].keys())[0]
        log_entry = update_op["$push"][push_field]
        assert "duration_ms" not in log_entry
        assert "operation_type" not in log_entry

    @pytest.mark.asyncio
    async def test_logging_db_error_does_not_raise(self) -> None:
        """DB errors in logging are caught and logged, not raised."""
        limit = _make_limit(enable_logging=True, period=TimeInterval.TOTAL)
        coll = _mock_coll()
        _patch_coll(limit, coll)
        coll.update_one.side_effect = Exception("DB write error")
        # Should not raise
        await limit.log_usage_async(
            model_name="test",
            tokens_input=100,
            tokens_output=50,
            costs=0.01,
        )
