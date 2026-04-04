"""Azure OpenAI provider implementation."""
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from llming_models.providers import BaseProvider, register_provider
from llming_models.llm_base_client import LlmClient
from .azure_openai_models import AZURE_OPENAI_MODELS, LLMInfo
from llming_models.providers.openai.openai_client import OpenAILlmClient
from llming_models.tools.llm_toolbox import LlmToolbox
from llming_models.providers.llm_provider_models import ReasoningEffort

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials


@register_provider("azure_openai")
class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI provider implementation.

    Uses the same OpenAILlmClient as the standard OpenAI provider but
    configured with Azure endpoints and API type.
    """

    def __init__(self, credentials: ProviderCredentials | None = None):
        """Initialize Azure OpenAI provider."""
        super().__init__("azure_openai", "Azure OpenAI", credentials)
        if self._credentials is None:
            key = os.environ.get('AZURE_OPENAI_API_KEY')
            endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
            if key and endpoint:
                from llming_models.credentials import ProviderCredentials as PC
                self._credentials = PC(
                    api_key=key,
                    base_url=endpoint,
                    api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2025-04-01-preview'),
                )

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key and endpoint)."""
        return self._credentials is not None and self._credentials.base_url is not None

    def get_models(self) -> list[LLMInfo]:
        """Get list of available Azure OpenAI models."""
        return AZURE_OPENAI_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        streaming: bool = False,
        base_url: str | None = None,
        toolboxes: list[LlmToolbox] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> LlmClient:
        """Create an Azure OpenAI client.

        Args:
            model: Deployment name to use (matches model names)
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL override (defaults to AZURE_OPENAI_ENDPOINT)
            toolboxes: Optional list of LlmToolbox objects
            reasoning_effort: Optional reasoning effort level
            **kwargs: Additional arguments

        Returns:
            Configured LlmClient instance
        """
        if not self.is_available:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "environment variables must be set"
            )

        assert self._credentials is not None
        return OpenAILlmClient(
            api_key=self._credentials.api_key.get_secret_value(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self._credentials.base_url,
            toolboxes=toolboxes,
            api_type="azure",
            api_version=self._credentials.api_version or '2025-04-01-preview',
            reasoning_effort=reasoning_effort,
        )
