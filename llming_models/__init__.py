"""llming-models — LLM execution engine.

Multi-provider streaming, MCP tools, and budget management:

- **Model Info**: LLMInfo, ModelSize, ReasoningEffort — model metadata and capabilities
- **Config**: LLMBaseConfig, LLMGlobalConfig, LLMUserConfig — model selection and filtering
- **Session**: ChatSession, LLMConfig — streaming LLM interactions with tool support
- **Providers**: LLMManager, BaseProvider — multi-provider orchestration
- **Messages**: ChatMessage, ChatHistory, Role, LlmSystemMessage, LlmHumanMessage, LlmAIMessage
- **Tools**: ToolDefinition, ToolRegistry, MCPServerConfig — MCP-compatible tool system
- **Budget**: BudgetLimit, LLMBudgetManager, MemoryBudgetLimit — budget tracking and enforcement
- **Persistence**: Conversation, ConversationMeta, FileRef — IndexedDB-compatible conversation storage
"""

from llming_models.model_info import LLMInfo, ModelSize, ReasoningEffort
from llming_models.model_categories import ModelCategories
from llming_models.config import LLMBaseConfig, LLMGlobalConfig, LLMUserConfig
from llming_models.llm_base_models import ChatHistory, ChatMessage, Role
from llming_models.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage
from llming_models.session import ChatSession, LLMConfig
from llming_models.credentials import ProviderCredentials, LLMCredentials
from llming_models.llm_provider_manager import LLMManager
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_models.conversation import (
    Conversation,
    ConversationMeta,
    AvatarOverride,
    FileRef,
    serialize_message,
)
from llming_models.content_blocks import (
    encode_content_block,
    extract_content_blocks,
    strip_content_blocks,
)
from llming_models.budget import (
    BudgetScope,
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
    "LlmSystemMessage",
    "LlmHumanMessage",
    "LlmAIMessage",
    # Session
    "ChatSession",
    "LLMConfig",
    # Credentials
    "ProviderCredentials",
    "LLMCredentials",
    # Provider manager
    "LLMManager",
    # Tools
    "ToolCallInfo",
    "ToolCallStatus",
    # Persistence
    "Conversation",
    "ConversationMeta",
    "AvatarOverride",
    "FileRef",
    "serialize_message",
    # Content blocks
    "encode_content_block",
    "extract_content_blocks",
    "strip_content_blocks",
    # Budget
    "BudgetScope",
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


def __getattr__(name: str) -> type:
    if name == "MongoDBBudgetLimit":
        from llming_models.budget import MongoDBBudgetLimit
        return MongoDBBudgetLimit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
