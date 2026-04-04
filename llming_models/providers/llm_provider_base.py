"""Base provider interface for LLM providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from .llm_provider_models import LLMInfo
from ..llm_base_client import LlmClient
from ..tools.llm_toolbox import LlmToolbox

if TYPE_CHECKING:
    from ..credentials import ProviderCredentials


class BaseProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, name: str, label: str, credentials: ProviderCredentials | None = None):
        """Initialize provider.

        :param name: The provider name, "openai", "anthropic", etc.
        :param label: The provider label, "OpenAI", "Anthropic", etc.
        :param credentials: Optional explicit credentials.  Subclasses fall
            back to environment variables when *None*.
        """
        self.name = name
        self.label = label
        self._credentials = credentials

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key)."""
        pass

    @abstractmethod
    def get_models(self) -> list[LLMInfo]:
        """Get list of available models for this provider."""
        pass

    @abstractmethod
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
        """Create an LLM client.

        Args:
            model: Model name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL for the API
            toolboxes: Optional list of LlmToolbox objects to provide tool support
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LlmClient instance
        """
        pass
