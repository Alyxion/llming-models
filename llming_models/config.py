"""LLM configuration classes.

Provides global and user-level model configuration with include/exclude
filters and default model selection per category.
"""

import fnmatch
from dataclasses import dataclass, field


@dataclass
class LLMBaseConfig:
    """Base configuration for LLMs."""
    included_models: list[str] = field(default_factory=lambda: ["*"])
    """Filter for included models. The filter is applied to the fully qualified model name
    (e.g., "anthropic:claude_3_opus"). "*" matches all models."""
    excluded_models: list[str] = field(default_factory=list)
    """Filter for excluded models."""

    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by this config.

        :param model: The fully qualified model name
        :return: True if the model is supported, False otherwise
        """
        included: bool = False
        for included_model in self.included_models:
            if fnmatch.fnmatch(model, included_model):
                included = True
                break
        if not included:
            return False
        for excluded_model in self.excluded_models:
            if fnmatch.fnmatch(model, excluded_model):
                return False
        return True


@dataclass
class LLMGlobalConfig(LLMBaseConfig):
    """Defines the global configuration for LLMs."""
    default_models: dict[str, str | list[str]] = field(default_factory=lambda: {
        "small": ["claude_haiku", "gpt-5-nano"],
        "medium": ["claude_sonnet", "gpt-5-mini"],
        "large": ["claude_sonnet", "gpt-5.2"],
        "reasoning_small": ["claude_haiku", "gpt-5-mini"],
        "reasoning_medium": ["claude_sonnet", "gpt-5.2"],
        "reasoning_large": ["claude_sonnet", "gpt-5.2"],
    })
    """Default models per category. Values can be a single model name or a list of fallbacks
    (first available wins). Use bare model names for cascade resolution, or 'provider:model'
    to pin to a specific provider."""
    provider_cascade: list[str] = field(default_factory=lambda: [
        "azure_openai", "openai", "azure_anthropic", "anthropic",
        "mistral", "google", "together",
    ])
    """Ordered list of provider priorities."""
    budgets: list = field(default_factory=list)
    """List of budgets."""
    model_defaults_version: int = 2
    """Bump this to invalidate stored model preferences in client browsers."""

    def get_default_model_candidates(self, category: str) -> list[str]:
        """Get the ordered list of candidate models for a category.

        :param category: The category to look up.
        :return: List of model names in priority order.
        """
        val = self.default_models.get(category)
        if val is None:
            return []
        return val if isinstance(val, list) else [val]

    def get_default_model(self, category: str) -> str | None:
        """Get the first default model for a category (ignores availability).

        :param category: The category to look up.
        :return: The first default model for the specified category
        """
        candidates = self.get_default_model_candidates(category)
        return candidates[0] if candidates else None


@dataclass
class LLMUserConfig(LLMBaseConfig):
    """User specific configuration for LLMs."""

    global_config: LLMGlobalConfig = field(default_factory=LLMGlobalConfig)
    """Pointer to the global configuration."""
    default_models: dict[str, str] = field(default_factory=dict)
    """Default models for providers."""
    budgets: list = field(default_factory=list)
    """List of budgets."""
    prompt_parameters: dict[str, str] = field(default_factory=dict)
    """Prompt parameters."""

    def is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by the user config."""
        return self.global_config.is_model_supported(model) and super().is_model_supported(model)

    def get_default_model(self, category: str) -> str:
        """Get the default model for a given category. User setting overrules global.

        :param category: The category to look up.
        :return: The default model for the specified category
        """
        if category in self.default_models:
            return self.default_models[category]
        else:
            return self.global_config.get_default_model(category)
