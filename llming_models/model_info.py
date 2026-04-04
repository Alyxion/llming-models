"""Model metadata types for LLM providers.

Defines the core types used to describe LLM models: their capabilities,
pricing, size category, reasoning support, and UI metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum


class ModelSize(IntEnum):
    """Size categories for LLM models."""
    VERY_SMALL = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    VERY_LARGE = 5


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for models that support it.

    - NONE: Disable reasoning completely (fastest, most deterministic)
    - MINIMAL: Very fast, almost no thinking traces
    - LOW: Some reasoning, faster responses
    - MEDIUM: Moderate reasoning (default for most tasks)
    - HIGH: Full reasoning capabilities
    """
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class LLMInfo:
    """Information about an LLM model."""
    provider: str  # Provider name
    name: str  # High-level name for identification
    label: str  # Human-readable label
    model: str  # Actual model name for the API
    description: str
    input_token_price: float  # Price per 1M input tokens
    cached_input_token_price: float = 0.0  # Price per 1M cached input tokens
    output_token_price: float = 0.0  # Price per 1M output tokens
    size: ModelSize = ModelSize.MEDIUM  # Model size category
    max_input_tokens: int = 64000  # Maximum number of input tokens
    max_output_tokens: int = 4096  # Maximum number of output tokens
    api_base: str | None = None  # Base URL for the API
    supports_system_prompt: bool = True  # Whether the model supports system prompts
    tokenizer_name: str | None = None  # Tokenizer for token counting
    model_icon: str | None = None  # Path to model-specific icon
    company_icon: str | None = None  # Path to company/inventor icon
    hosting_icon: str | None = None  # Path to optional hosting company icon
    popularity: int = 0  # Higher value = more popular
    reasoning: bool = False  # True if this is a reasoning model
    reasoning_effort: ReasoningEffort | None = None  # Reasoning effort level
    default_reasoning_effort: ReasoningEffort | None = None  # Default for this model size
    enforced_temperature: float | None = None  # Enforced temperature value
    supports_image_input: bool = False  # Whether the model supports image inputs

    # UI metadata for model selector
    speed: int = 5  # 1-10, higher = faster response
    quality: int = 5  # 1-10, higher = better reasoning/output quality
    best_use: str = "General"  # Short label for the model's strength
    highlights: list[str] = field(default_factory=list)  # Key capabilities

    # Tool configuration
    default_tools: list[str] = field(default_factory=list)  # Default tools to enable
    native_tools: dict[str, dict] = field(default_factory=dict)  # Provider-native tool configs
