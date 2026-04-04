"""Mistral provider implementation."""
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from llming_models.providers import BaseProvider, register_provider
from llming_models.llm_base_client import LlmClient
from .mistral_models import MISTRAL_MODELS, LLMInfo
from llming_models.providers.openai_compat_client import OpenAICompatibleClient
from llming_models.tools.llm_toolbox import LlmToolbox

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials


@register_provider("mistral")
class MistralProvider(BaseProvider):
    """Mistral provider implementation."""

    DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, credentials: ProviderCredentials | None = None):
        """Initialize Mistral provider."""
        super().__init__("mistral", "Mistral", credentials)
        if self._credentials is None:
            key = os.environ.get('MISTRAL_API_KEY')
            if key:
                from llming_models.credentials import ProviderCredentials as PC
                self._credentials = PC(api_key=key)

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._credentials is not None

    def get_models(self) -> list[LLMInfo]:
        """Get list of available Mistral models."""
        return MISTRAL_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        streaming: bool = False,
        base_url: str | None = None,
        toolboxes: list[LlmToolbox] | None = None,
        **kwargs: Any,
    ) -> LlmClient:
        """Create a Mistral chat model client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API
            toolboxes: Optional list of toolboxes (not used by Mistral)
            **kwargs: Additional arguments

        Returns:
            Configured OpenAICompatibleClient instance

        Raises:
            ValueError: If MISTRAL_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        assert self._credentials is not None
        return OpenAICompatibleClient(
            api_key=self._credentials.api_key.get_secret_value(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self._credentials.base_url or self.DEFAULT_BASE_URL
        )
