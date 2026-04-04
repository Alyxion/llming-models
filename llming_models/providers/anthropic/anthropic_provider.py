"""Anthropic provider implementation."""
from __future__ import annotations

import os
import logging
from typing import Any, TYPE_CHECKING

from llming_models.providers import BaseProvider, register_provider
from llming_models.llm_base_client import LlmClient
from .anthropic_models import ANTHROPIC_MODELS, LLMInfo
from .anthropic_client import AnthropicClient
from llming_models.tools.llm_toolbox import LlmToolbox

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials

logger = logging.getLogger(__name__)


@register_provider("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    def __init__(self, credentials: ProviderCredentials | None = None):
        """Initialize Anthropic provider."""
        super().__init__("anthropic", "Anthropic", credentials)
        if self._credentials is None:
            key = os.environ.get('ANTHROPIC_API_KEY')
            if key:
                from llming_models.credentials import ProviderCredentials as PC
                self._credentials = PC(api_key=key)

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        return self._credentials is not None

    def get_models(self) -> list[LLMInfo]:
        """Get list of available Anthropic models."""
        return ANTHROPIC_MODELS

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
        """Create an Anthropic chat model client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API (not used by Anthropic)
            toolboxes: Optional list of toolboxes for tool support
            **kwargs: Additional arguments passed to client

        Returns:
            Configured Anthropic client instance

        Raises:
            ValueError: If ANTHROPIC_API_KEY environment variable is not set
        """
        if not self.is_available:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        assert self._credentials is not None
        return AnthropicClient(
            api_key=self._credentials.api_key.get_secret_value(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            toolboxes=toolboxes or []
        )
