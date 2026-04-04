from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo
from pymongo import ReturnDocument

from .time_intervals import TimeIntervalHandler
from .budget_limit import BudgetLimit
from .budget_types import BudgetScope, LimitPeriod

logger = logging.getLogger(__name__)

class MongoDBBudgetLimit(BudgetLimit):
    """Budget limit with MongoDB-based tracking using atomic operations.

    The budget amount can be stored per-document in MongoDB. If a stored
    ``amount`` field exists on the document, it takes precedence over the
    constructor default. This allows per-user budget overrides without
    code changes.

    Supports both GLOBAL (shared) and PER_USER (per user_id) scopes.
    For PER_USER limits the MongoDB document key includes the ``user_id``
    so each user has an independent counter.
    """
    mongo_uri: str
    mongo_db: str
    mongo_collection: str
    enable_logging: bool
    user_id: str | None
    _default_amount: float
    _async_coll: Any | None

    def __init__(
        self,
        *,
        name: str,
        amount: float,
        period: LimitPeriod,
        mongo_uri: str,
        mongo_db: str,
        mongo_collection: str,
        interval_value: int | None = None,
        timezone_str: str = "UTC",
        enable_logging: bool = False,
        user_id: str | None = None,
        scope: BudgetScope = BudgetScope.GLOBAL,
    ) -> None:
        super().__init__(name=name, amount=amount, period=period,
                         interval_value=interval_value, timezone_str=timezone_str,
                         scope=scope)
        if not mongo_uri or not mongo_db or not mongo_collection:
            raise ValueError("MongoDB URI, database, and collection are required")
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection
        self.enable_logging = enable_logging
        self.user_id = user_id  # default user_id (for logging fallback)
        self._default_amount = amount  # constructor default, used as fallback
        self._async_coll = None

    def _ensure_async_coll(self) -> Any:
        """Lazily get the shared async client and set up the collection handle."""
        if self._async_coll is None:
            from ._mongo import get_async_mongo_client
            client = get_async_mongo_client(self.mongo_uri)
            self._async_coll = client[self.mongo_db][self.mongo_collection]
        return self._async_coll

    def _get_current_time(self) -> datetime:
        """Get current time in the configured timezone."""
        return datetime.now(timezone.utc).astimezone(ZoneInfo(self.timezone))

    def _get_mongo_key(self, time: datetime | None = None, user_id: str | None = None) -> dict[str, str]:
        """Generate the MongoDB document key for this budget and period (top-level doc).

        For PER_USER scoped limits the key includes ``user_id`` so each
        user gets an independent document.
        """
        if time is None:
            time = self._get_current_time()
        key: dict[str, str] = {
            "name": self.name,
            "period": self.period.value,
        }
        if self.scope == BudgetScope.PER_USER:
            effective_uid = user_id or self.user_id
            if not effective_uid:
                raise ValueError(f"user_id is required for PER_USER limit '{self.name}'")
            key["user_id"] = effective_uid
        return key

    def _get_usage_path(self, time: datetime | None = None) -> list[str]:
        """Generate the linear path for usage in the document."""
        if time is None:
            time = self._get_current_time()
        key_suffix = self._get_key_suffix(time)
        return ["usage", key_suffix]

    def _get_expiry(self) -> timedelta | None:
        """Get expiry duration based on period."""
        return TimeIntervalHandler.get_expiry(self.period, self.interval_value)

    async def _get_effective_amount(self, user_id: str | None = None) -> float:
        """Get the effective budget amount -- stored override or constructor default."""
        try:
            coll = self._ensure_async_coll()
            doc = await coll.find_one(self._get_mongo_key(user_id=user_id), {"amount": 1})
            if doc and "amount" in doc:
                return float(doc["amount"])
        except Exception:
            pass
        return self._default_amount

    async def get_available_budget_async(self, user_id: str | None = None) -> float:
        """Get available budget for the current period."""
        logger.debug(f"Getting available budget (async) for limit '{self.name}' (period: {self.period.value})")
        current_time = self._get_current_time()
        key = self._get_mongo_key(current_time, user_id=user_id)
        usage_path = self._get_usage_path(current_time)
        field = ".".join(usage_path + ["used"])
        try:
            coll = self._ensure_async_coll()
            doc = await coll.find_one(key, {field: 1, "amount": 1})
            effective_amount = await self._get_effective_amount(user_id=user_id)
            used = 0.0
            d = doc
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            if d is not None and isinstance(d, dict) and "used" in d:
                used = float(d["used"])
            elif isinstance(d, (int, float)):
                used = float(d)
            available = max(effective_amount - used, 0.0)
            logger.debug(f"Available budget (async) for limit '{self.name}': {effective_amount} - {used} = {available}")
            return available
        except Exception as e:
            raise RuntimeError(f"Failed to get available budget from MongoDB (async): {e}")

    async def reserve_budget_async(self, amount: float, user_id: str | None = None) -> bool:
        """Reserve budget for an operation."""
        if amount <= 0:
            return False
        effective_amount = await self._get_effective_amount(user_id=user_id)
        logger.debug(f"Attempting to reserve {amount} (async) from limit '{self.name}' (period: {self.period.value})")
        if amount > effective_amount:
            logger.debug(f"Amount {amount} exceeds total budget {effective_amount} for limit '{self.name}'")
            return False

        current_time = self._get_current_time()
        key = self._get_mongo_key(current_time, user_id=user_id)
        usage_path = self._get_usage_path(current_time)
        field = ".".join(usage_path + ["used"])
        now = current_time
        coll = self._ensure_async_coll()

        try:
            update: dict[str, Any] = {
                "$inc": {field: amount},
                "$setOnInsert": {"created_at": now},
            }
            result = await coll.find_one_and_update(
                key,
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            d = result
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            used = float(d["used"]) if d and "used" in d else amount

            if used > effective_amount:
                await coll.update_one(key, {"$inc": {field: -amount}})
                logger.debug(f"Exceeded budget limit (async), rolling back. Available: {effective_amount - used + amount}")
                return False

            logger.debug(f"Successfully reserved {amount} (async), remaining: {effective_amount - used}")
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to reserve budget in MongoDB (async): {e}")

    async def return_budget_async(self, amount: float, user_id: str | None = None) -> None:
        """Return unused budget."""
        if amount <= 0:
            return
        key = self._get_mongo_key(user_id=user_id)
        usage_path = self._get_usage_path()
        field = ".".join(usage_path + ["used"])
        coll = self._ensure_async_coll()

        try:
            update: dict[str, Any] = {
                "$inc": {field: -amount},
            }
            result = await coll.find_one_and_update(
                key,
                update,
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            d = result
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            used = float(d["used"]) if d and "used" in d else 0.0

            if used < 0:
                await coll.update_one(key, {"$set": {field: 0}})
                logger.debug("Usage went below 0 (async), resetting to 0")
        except Exception as e:
            raise RuntimeError(f"Failed to return budget to MongoDB (async): {e}")

    async def reset_async(self) -> None:
        """Reset budget to initial amount (remove all usage records for this budget)."""
        pattern: dict[str, str] = {
            "name": self.name,
            "period": self.period.value,
        }
        try:
            coll = self._ensure_async_coll()
            await coll.delete_many(pattern)
            logger.debug(f"Reset all budget records (async) for '{self.name}' (period: {self.period.value})")
        except Exception as e:
            raise RuntimeError(f"Failed to reset budget in MongoDB (async): {e}")

    async def set_amount_async(self, new_amount: float) -> None:
        """Persist a budget amount override in the MongoDB document.

        The stored amount takes precedence over the constructor default.
        Pass the constructor default to remove the override.
        """
        key = self._get_mongo_key()
        coll = self._ensure_async_coll()
        await coll.update_one(key, {"$set": {"amount": new_amount}}, upsert=True)
        self.amount = new_amount
        logger.info(f"Budget amount for '{self.name}' set to {new_amount}€")

    async def get_usage_async(self) -> float:
        """Get the used budget for the current period."""
        current_time = self._get_current_time()
        key = self._get_mongo_key(current_time)
        usage_path = self._get_usage_path(current_time)
        field = ".".join(usage_path + ["used"])
        try:
            coll = self._ensure_async_coll()
            doc = await coll.find_one(key, {field: 1})
            d = doc
            for part in usage_path:
                if d is None or part not in d:
                    d = None
                    break
                d = d[part]
            if d is not None and isinstance(d, dict) and "used" in d:
                return float(d["used"])
            elif isinstance(d, (int, float)):
                return float(d)
            return 0.0
        except Exception:
            return 0.0

    async def log_usage_async(self, *, model_name: str, tokens_input: int, tokens_output: int, costs: float, duration_ms: float | None = None, user_id: str | None = None, operation_type: str | None = None) -> None:
        """Log usage information for a completed request."""
        if not self.enable_logging:
            return

        key = self._get_mongo_key()
        now = self._get_current_time()
        usage_path = self._get_usage_path(now)
        logs_field = ".".join(usage_path + ["logs"])

        log_entry: dict[str, Any] = {
            "timestamp": now,
            "model": model_name,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "costs": costs
        }

        if duration_ms is not None:
            log_entry["duration_ms"] = duration_ms

        if operation_type:
            log_entry["operation_type"] = operation_type

        effective_user_id = user_id if user_id is not None else self.user_id
        if effective_user_id:
            log_entry["user_id"] = effective_user_id

        try:
            coll = self._ensure_async_coll()
            await coll.update_one(
                key,
                {"$push": {logs_field: log_entry}},
                upsert=True
            )
            logger.debug(f"Logged usage (async) for model {model_name}: {tokens_input} input tokens, {tokens_output} output tokens, {costs} cost")
        except Exception as e:
            logger.error(f"Failed to log usage in MongoDB (async): {e}")
