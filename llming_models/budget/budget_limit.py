from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import threading

from .time_intervals import TimeIntervalHandler
from .budget_types import BudgetScope, LimitPeriod


class BudgetLimit(ABC):
    """Base class for budget limits."""
    name: str
    amount: float
    period: LimitPeriod
    interval_value: int | None
    timezone: str
    scope: BudgetScope
    _initial_amount: float
    _lock: threading.Lock

    def __init__(self, *, name: str, amount: float, period: LimitPeriod,
                 interval_value: int | None = None, timezone_str: str = "UTC",
                 scope: BudgetScope = BudgetScope.GLOBAL) -> None:
        self.name = name
        self.amount = amount
        self.period = period
        self.interval_value = interval_value if isinstance(interval_value, int) else None
        self.timezone = timezone_str
        self.scope = scope
        self._initial_amount = amount
        self._lock = threading.Lock()

    def _get_key_suffix(self, time: datetime) -> str:
        """Generate key suffix based on period."""
        return TimeIntervalHandler.get_key_suffix(self.period, time, self.interval_value)

    @abstractmethod
    async def get_available_budget_async(self, user_id: str | None = None) -> float:
        """Get available budget for the current period.

        :param user_id: Required for PER_USER scoped limits.
        """
        raise NotImplementedError("Subclasses must implement get_available_budget_async")

    @abstractmethod
    async def reserve_budget_async(self, amount: float, user_id: str | None = None) -> bool:
        """Reserve budget for an operation.

        :param amount: Amount in Euros to reserve.
        :param user_id: Required for PER_USER scoped limits.
        """
        raise NotImplementedError("Subclasses must implement reserve_budget_async")

    @abstractmethod
    async def return_budget_async(self, amount: float, user_id: str | None = None) -> None:
        """Return unused budget.

        :param amount: Amount in Euros to return.
        :param user_id: Required for PER_USER scoped limits.
        """
        raise NotImplementedError("Subclasses must implement return_budget_async")

    @abstractmethod
    async def reset_async(self) -> None:
        """Reset budget to initial amount."""
        raise NotImplementedError("Subclasses must implement reset_async")

    async def log_usage_async(self, *, model_name: str, tokens_input: int, tokens_output: int, costs: float, duration_ms: float | None = None, user_id: str | None = None, operation_type: str | None = None) -> None:
        """Log usage information for a completed request.

        Optional -- subclasses can override to log usage information.
        """
        pass
