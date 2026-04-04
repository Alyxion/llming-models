from __future__ import annotations

from enum import Enum
from typing import Callable, TypedDict

from .time_intervals import TimeInterval

#: Alias kept for backward compatibility with existing code.
LimitPeriod = TimeInterval


class BudgetScope(str, Enum):
    """Scope of a budget limit."""
    GLOBAL = "global"       # shared across all users
    PER_USER = "per_user"   # separate counter per user_id


class BudgetInfo(TypedDict, total=False):
    """Budget information returned by a BudgetHandler callback."""
    available: float
    reserved: float


BudgetHandler = Callable[[], BudgetInfo]

class InsufficientBudgetError(Exception):
    """Raised when there is not enough budget available for the requested operation."""
    limit_name: str

    def __init__(self, message: str, limit_name: str) -> None:
        self.limit_name = limit_name
        super().__init__(message)


class TokenUsage:
    """Represents the token usage for an LLM operation."""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    def __init__(self, input_tokens: int, output_tokens: int,
                 input_cost: float, output_cost: float) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_cost = input_cost
        self.output_cost = output_cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost
