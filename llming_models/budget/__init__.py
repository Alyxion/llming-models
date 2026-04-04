from __future__ import annotations

from .budget_types import BudgetScope, LimitPeriod, InsufficientBudgetError, TokenUsage, BudgetInfo, BudgetHandler
from .time_intervals import TimeInterval, TimeIntervalHandler
from .budget_limit import BudgetLimit
from .memory_budget_limit import MemoryBudgetLimit
from .budget_manager import LLMBudgetManager

def __getattr__(name: str) -> type:
    """Lazy imports for optional dependencies."""
    if name == "MongoDBBudgetLimit":
        from .mongodb_budget_limit import MongoDBBudgetLimit
        return MongoDBBudgetLimit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'BudgetScope',
    'LimitPeriod',
    'InsufficientBudgetError',
    'TokenUsage',
    'BudgetInfo',
    'BudgetHandler',
    'TimeInterval',
    'TimeIntervalHandler',
    'BudgetLimit',
    'MemoryBudgetLimit',
    'MongoDBBudgetLimit',
    'LLMBudgetManager',
]
