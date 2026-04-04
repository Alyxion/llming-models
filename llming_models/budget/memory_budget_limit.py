from __future__ import annotations

from datetime import datetime
import logging
from zoneinfo import ZoneInfo

from .budget_limit import BudgetLimit
from .budget_types import BudgetScope, LimitPeriod

logger = logging.getLogger(__name__)


class MemoryBudgetLimit(BudgetLimit):
    """Budget limit with in-memory tracking.

    Supports both GLOBAL (shared) and PER_USER (per user_id) scopes.
    For PER_USER limits the storage key includes the user_id so each
    user has an independent counter.
    """
    _usage: dict[str, float] | None

    def __init__(self, *, name: str, amount: float, period: LimitPeriod,
                 timezone_str: str = "UTC", scope: BudgetScope = BudgetScope.GLOBAL) -> None:
        super().__init__(name=name, amount=amount, period=period,
                         timezone_str=timezone_str, scope=scope)
        self._initial_amount = amount
        # For TOTAL period we track the running balance directly;
        # for periodic limits we track usage per period key.
        # PER_USER+TOTAL uses a dict keyed by user_id.
        if period == LimitPeriod.TOTAL and scope == BudgetScope.GLOBAL:
            self._usage = None  # use self.amount directly
        else:
            self._usage = {}

    def _get_current_time(self) -> datetime:
        """Get current time."""
        return datetime.now(tz=ZoneInfo(self.timezone))

    def _storage_key(self, user_id: str | None = None) -> str:
        """Build the storage key incorporating scope + period + user_id."""
        if self.period == LimitPeriod.TOTAL:
            period_key = "total"
        else:
            period_key = self._get_key_suffix(self._get_current_time())

        if self.scope == BudgetScope.PER_USER:
            if user_id is None:
                raise ValueError(f"user_id is required for PER_USER limit '{self.name}'")
            return f"{user_id}:{period_key}"
        return period_key

    # -- sync interface -------------------------------------------------- #

    def get_available_budget(self, user_id: str | None = None) -> float:
        """Get available budget for the current period."""
        # GLOBAL + TOTAL: simple running balance
        if self.period == LimitPeriod.TOTAL and self.scope == BudgetScope.GLOBAL:
            return self.amount

        assert self._usage is not None  # guaranteed: GLOBAL+TOTAL returned above
        key = self._storage_key(user_id)
        with self._lock:
            current_usage = self._usage.get(key, 0.0)
            return max(self.amount - current_usage, 0.0)

    def reserve_budget(self, amount: float, user_id: str | None = None) -> bool:
        """Reserve budget for an operation."""
        if amount > self.amount:
            return False

        # GLOBAL + TOTAL: simple running balance
        if self.period == LimitPeriod.TOTAL and self.scope == BudgetScope.GLOBAL:
            with self._lock:
                if amount <= self.amount:
                    self.amount -= amount
                    return True
                return False

        assert self._usage is not None  # guaranteed: GLOBAL+TOTAL returned above
        key = self._storage_key(user_id)
        with self._lock:
            # Re-check key in case period rolled over
            fresh_key = self._storage_key(user_id)
            if fresh_key != key:
                key = fresh_key

            current_usage = self._usage.get(key, 0.0)
            if amount <= (self.amount - current_usage):
                self._usage[key] = current_usage + amount
                return True
            return False

    def return_budget(self, amount: float, user_id: str | None = None) -> None:
        """Return unused budget."""
        if self.period == LimitPeriod.TOTAL and self.scope == BudgetScope.GLOBAL:
            with self._lock:
                self.amount += amount
            return

        assert self._usage is not None  # guaranteed: GLOBAL+TOTAL returned above
        key = self._storage_key(user_id)
        with self._lock:
            current_usage = self._usage.get(key, 0.0)
            self._usage[key] = max(0.0, current_usage - amount)

    def reset(self) -> None:
        """Reset budget to initial amount."""
        with self._lock:
            if self.period == LimitPeriod.TOTAL and self.scope == BudgetScope.GLOBAL:
                self.amount = self._initial_amount
            else:
                self._usage = {}

    # -- async wrappers -------------------------------------------------- #

    async def get_available_budget_async(self, user_id: str | None = None) -> float:
        return self.get_available_budget(user_id)

    async def reserve_budget_async(self, amount: float, user_id: str | None = None) -> bool:
        if amount <= 0:
            return False
        return self.reserve_budget(amount, user_id)

    async def return_budget_async(self, amount: float, user_id: str | None = None) -> None:
        if amount <= 0:
            return
        self.return_budget(amount, user_id)

    async def reset_async(self) -> None:
        self.reset()
