"""llming-models — Model metadata, configuration, and budget management.

Provides the foundational types shared across llming applications:

- **Model Info**: LLMInfo, ModelSize, ReasoningEffort — model metadata and capabilities
- **Config**: LLMBaseConfig, LLMGlobalConfig, LLMUserConfig — model selection and filtering
- **Categories**: ModelCategories — standard model category constants
- **Budget**: BudgetLimit, LLMBudgetManager, MemoryBudgetLimit — budget tracking and enforcement
"""

from llming_models.model_info import LLMInfo, ModelSize, ReasoningEffort
from llming_models.model_categories import ModelCategories
from llming_models.config import LLMBaseConfig, LLMGlobalConfig, LLMUserConfig
from llming_models.budget import (
    LimitPeriod,
    InsufficientBudgetError,
    TokenUsage,
    BudgetInfo,
    BudgetHandler,
    BudgetLimit,
    MemoryBudgetLimit,
    LLMBudgetManager,
)

__all__ = [
    # Model info
    "LLMInfo",
    "ModelSize",
    "ReasoningEffort",
    # Categories
    "ModelCategories",
    # Config
    "LLMBaseConfig",
    "LLMGlobalConfig",
    "LLMUserConfig",
    # Budget
    "LimitPeriod",
    "InsufficientBudgetError",
    "TokenUsage",
    "BudgetInfo",
    "BudgetHandler",
    "BudgetLimit",
    "MemoryBudgetLimit",
    "LLMBudgetManager",
]
