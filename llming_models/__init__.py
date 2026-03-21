"""llming-models — LLM execution engine.

Multi-provider streaming, MCP tools, and budget management:

- **Model Info**: LLMInfo, ModelSize, ReasoningEffort — model metadata and capabilities
- **Config**: LLMBaseConfig, LLMGlobalConfig, LLMUserConfig — model selection and filtering
- **Session**: ChatSession, LLMConfig — streaming LLM interactions with tool support
- **Providers**: LLMManager, BaseProvider — multi-provider orchestration
- **Messages**: ChatMessage, ChatHistory, Role, LlmSystemMessage, LlmHumanMessage, LlmAIMessage
- **Tools**: ToolDefinition, ToolRegistry, MCPServerConfig — MCP-compatible tool system
- **Budget**: BudgetLimit, LLMBudgetManager, MemoryBudgetLimit — budget tracking and enforcement
"""

from llming_models.model_info import LLMInfo, ModelSize, ReasoningEffort
from llming_models.model_categories import ModelCategories
from llming_models.config import LLMBaseConfig, LLMGlobalConfig, LLMUserConfig
from llming_models.llm_base_models import ChatHistory, ChatMessage, Role
from llming_models.session import ChatSession, LLMConfig
from llming_models.llm_provider_manager import LLMManager
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
    # Messages
    "ChatHistory",
    "ChatMessage",
    "Role",
    # Session
    "ChatSession",
    "LLMConfig",
    # Provider manager
    "LLMManager",
    # Budget
    "LimitPeriod",
    "InsufficientBudgetError",
    "TokenUsage",
    "BudgetInfo",
    "BudgetHandler",
    "BudgetLimit",
    "MemoryBudgetLimit",
    "MongoDBBudgetLimit",
    "LLMBudgetManager",
]


def __getattr__(name):
    if name == "MongoDBBudgetLimit":
        from llming_models.budget import MongoDBBudgetLimit
        return MongoDBBudgetLimit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
