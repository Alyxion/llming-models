"""LLM Manager for managing available LLM providers and sessions."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .session import LLMConfig, ChatSession
from .config import LLMUserConfig
from .credentials import LLMCredentials
from .providers import (
    LLMInfo,
    get_provider,
    BaseProvider,
)

if TYPE_CHECKING:
    from llming_models.budget import LLMBudgetManager

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages multiple LLM providers and sessions."""

    def __init__(
        self,
        *,
        user_config: LLMUserConfig | None = None,
        budget_manager: LLMBudgetManager | None = None,
        credentials: LLMCredentials | None = None,
    ) -> None:
        """Initialize LLM manager.

        :param user_config: Optional user configuration
        :param budget_manager: Optional budget manager
        :param credentials: Optional explicit credentials for providers.
            When omitted, providers fall back to environment variables.
        """
        # Initialize provider instances
        self.user_config = user_config or LLMUserConfig()
        self.credentials = credentials
        self.provider_cascade: list[str] = list(self.user_config.global_config.provider_cascade)
        self.budget_manager: LLMBudgetManager | None = budget_manager
        if self.budget_manager is None and (len(self.user_config.budgets) > 0 or len(self.user_config.global_config.budgets) > 0):
            from llming_models.budget import LLMBudgetManager
            combined_list = self.user_config.budgets + self.user_config.global_config.budgets
            self.budget_manager = LLMBudgetManager(combined_list)
        self.providers: dict[str, BaseProvider] = {}
        for provider_name in self.provider_cascade:
            try:
                provider_class = get_provider(provider_name)
                creds = self.credentials.for_provider(provider_name) if self.credentials else None
                provider = provider_class(credentials=creds)  # type: ignore[call-arg]
                # Only add provider if it's available (has valid API key)
                if provider.is_available:
                    # check if any model is available for the user
                    models = provider.get_models()
                    any_model = False
                    for model in models:
                        full_model_name = f"{provider_name}:{model.name}"
                        if self.user_config.is_model_supported(full_model_name):
                            any_model = True
                            break
                    if any_model:
                        self.providers[provider_name] = provider
            except (ValueError, KeyError):
                # Skip providers that can't be initialized
                continue
        active = [p for p in self.provider_cascade if p in self.providers]
        logger.info(f"LLM providers active (cascade order): {active}")

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a custom provider instance.

        Use this to add OpenAI-compatible or other custom providers at runtime.
        Runtime-registered providers are appended to the end of the cascade list.

        :param provider: A BaseProvider instance to register
        """
        from .providers import PROVIDERS
        PROVIDERS[provider.name] = type(provider)
        if provider.name not in self.provider_cascade:
            self.provider_cascade.append(provider.name)
        if provider.is_available:
            self.providers[provider.name] = provider

    def register_openai_compatible(
        self,
        name: str,
        label: str,
        api_key: str,
        base_url: str,
        models: list[LLMInfo],
    ) -> None:
        """Register a generic OpenAI-compatible provider.

        Convenience method for adding any OpenAI-compatible API endpoint.

        Example::

            manager.register_openai_compatible(
                name="deepseek",
                label="DeepSeek",
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
                models=[
                    LLMInfo(provider="deepseek", name="deepseek_chat", label="DeepSeek V3",
                            model="deepseek-chat", description="...",
                            input_token_price=0.5, output_token_price=1.5),
                ],
            )

        :param name: Provider name (used as key in registry)
        :param label: Human-readable label
        :param api_key: API key for the endpoint
        :param base_url: Base URL for the OpenAI-compatible API
        :param models: List of LLMInfo model definitions
        """
        from .providers.generic_openai_provider import GenericOpenAIProvider
        provider = GenericOpenAIProvider(
            name=name,
            label=label,
            api_key=api_key,
            base_url=base_url,
            models=models,
        )
        self.register_provider(provider)

    def get_available_llms(self) -> list[LLMInfo]:
        """Get deduplicated list of available LLMs, preferring cascade-priority providers.

        When the same model (by ``name`` field) is offered by multiple providers,
        only the version from the highest-cascade-priority provider is returned.
        """
        seen: dict[str, None] = {}
        models: list[LLMInfo] = []
        for provider in self.providers.values():
            for info in provider.get_models():
                if info.name not in seen:
                    seen[info.name] = None
                    models.append(info)
        return models

    def get_cascade_debug_info(self) -> list[dict]:
        """Return per-model cascade resolution info (debug / admin use).

        For every unique model name, shows which providers can serve it,
        which one is the active cascade winner, and whether each provider
        is available.
        """
        # Collect all models across all providers
        model_providers: dict[str, list[dict]] = {}
        for provider_name, provider in self.providers.items():
            for info in provider.get_models():
                entry = {
                    "provider": provider_name,
                    "label": info.label,
                    "hosting_icon": info.hosting_icon,
                }
                model_providers.setdefault(info.model, []).append(entry)

        result = []
        for model_name, entries in model_providers.items():
            try:
                active = self.get_provider_for_model(model_name)
            except ValueError:
                active = None
            result.append({
                "model": model_name,
                "active_provider": active,
                "providers": entries,
            })
        return result

    def get_providers_for_model(self, model: str) -> list[str]:
        """Get all providers that can serve a given model name, ordered by cascade priority.

        :param model: The model name or high-level name to look up. Alternatively, the model name can be specified as "provider:model".
        :return: List of provider names that can serve this model, ordered by cascade priority
        :raises ValueError: If model is not found
        """
        if ":" in model:  # explicit provider name
            provider_name, model_name = model.split(":")
            if provider_name not in self.providers:
                return []
            provider = self.providers[provider_name]
            for model_info in provider.get_models():
                if model_info.name == model_name:
                    return [provider_name]
            return []
        providers = []
        for provider_name, provider in self.providers.items():
            for model_info in provider.get_models():
                if model_info.name == model or model_info.model == model:
                    full_model_name = f"{provider_name}:{model_info.name}"
                    if not self.user_config.is_model_supported(full_model_name):
                        continue
                    providers.append(provider_name)
                    break

        if not providers:
            raise ValueError(f"Model {model} not found")
        return providers

    def get_provider_for_model(self, model: str) -> str:
        """Get the highest-priority provider for a given model name.

        :param model: The model name to look up, or "provider:model" to pin to a specific provider
        :return: Highest cascade-priority provider name for the specified model

        :raises ValueError: If model is not found
        """
        providers = self.get_providers_for_model(model)
        if not providers:
            raise ValueError(f"Model {model} not found")
        return providers[0]

    def get_default_model(self, category: str) -> str | None:
        """Get the first *available* default model for a given category.

        Iterates through the candidate list (from config) and returns the first
        model that is actually served by an active provider and not excluded.

        :param category: The category to look up. E.g. "small", "medium", "large",
            "reasoning_small", "reasoning_medium", "reasoning_large"
        :return: The first available default model, or the raw first candidate
            if none could be verified, or None if the category is unknown.
        """
        # User-level override takes priority (single model, no fallback list)
        if category in self.user_config.default_models:
            return self.user_config.default_models[category]
        candidates = self.user_config.global_config.get_default_model_candidates(category)
        if not candidates:
            return None
        for model in candidates:
            try:
                self.get_provider_for_model(model)
                return model  # available
            except ValueError:
                continue
        # None verified — return first candidate as best-effort fallback
        return candidates[0]

    def get_model_info(self, model: str) -> LLMInfo:
        """Get model info for a specific model.
        
        :param model: The model to get info for, or "provider/model" to look up by provider name
        :return: LLMInfo for the specified model
            
        :raises ValueError: If model is not found
        """
        provider_name = self.get_provider_for_model(model)
        provider = self.providers[provider_name]
        if ":" in model:
            model = model.split(":")[1]
        
        for info in provider.get_models():
            if info.name == model or info.model == model:
                return info
                
        raise ValueError(f"Model {model} not found")

    def get_config_for_model(self, model: str) -> LLMConfig:
        """Get configuration for a specific model.
        
        :param model: The model to get configuration for
        :return: LLMConfig for the specified model
            
        :raises ValueError: If model is not found or provider is not available
        """
        model_info = self.get_model_info(model)
        return LLMConfig(  # type: ignore[call-arg]
            provider=model_info.provider,
            model=model_info.model,
            base_url=model_info.api_base,
            temperature=0.7,
            max_input_tokens=model_info.max_input_tokens,
            max_tokens=model_info.max_output_tokens
        )

    def create_session(
        self,
        config: LLMConfig | None = None,
        model: str | None = None,
        category: str | None = None,
        system_prompt: str | None = None,
        budget_manager: LLMBudgetManager | None = None,
        user_id: str | None = None,
    ) -> ChatSession:
        """Create a new chat session.

        :param config: Optional LLMConfig to use. If not provided, model must be specified.
        :param model: Optional model name to use. Ignored if config is provided.
        :param category: Optional category to use. Ignored if config is provided.
        :param system_prompt: Optional system prompt to set context
        :param budget_manager: Optional budget manager to use. Defaults to predefined budget manager if provided
        :param user_id: Optional user ID to use
        :return: New ChatSession instance

        :raises ValueError: If neither config nor model is provided, or if model is not found
        """
        if category is not None:
            model = self.get_default_model(category)
            if model is None:
                raise ValueError(f"No default model found for category {category}")
        if budget_manager is None:
            budget_manager = self.budget_manager
        if config is None:
            if model is None:
                raise ValueError("Either config or model must be provided")
            config = self.get_config_for_model(model)

        # Forward stored credentials for the resolved provider
        creds = self.credentials.for_provider(config.provider) if self.credentials else None
        return ChatSession(config=config, system_prompt=system_prompt, budget_manager=budget_manager, user_id=user_id, credentials=creds)
