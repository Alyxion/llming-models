"""Azure Anthropic provider implementation.

Uses the same AnthropicClient as the standard Anthropic provider but
configured with Azure AI Services endpoints via AnthropicFoundry.
"""
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from llming_models.providers import BaseProvider, register_provider
from llming_models.llm_base_client import LlmClient
from .azure_anthropic_models import get_azure_anthropic_models, LLMInfo
from llming_models.providers.anthropic.anthropic_client import AnthropicClient
from llming_models.tools.llm_toolbox import LlmToolbox

if TYPE_CHECKING:
    from llming_models.credentials import ProviderCredentials


@register_provider("azure_anthropic")
class AzureAnthropicProvider(BaseProvider):
    """Azure Anthropic provider implementation.

    Uses the same AnthropicClient as the standard Anthropic provider but
    configured with Azure AI Services endpoints.
    """

    def __init__(self, credentials: ProviderCredentials | None = None):
        """Initialize Azure Anthropic provider."""
        super().__init__("azure_anthropic", "Azure Anthropic", credentials)
        if self._credentials is None:
            key = os.environ.get('AZURE_AI_SERVICES_KEY')
            endpoint = os.environ.get('AZURE_AI_SERVICES_ENDPOINT')
            if key and endpoint:
                from llming_models.credentials import ProviderCredentials as PC
                self._credentials = PC(api_key=key, base_url=endpoint)

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key and endpoint)."""
        return self._credentials is not None and self._credentials.base_url is not None

    def get_models(self) -> list[LLMInfo]:
        """Get list of available Azure Anthropic models (resolved dynamically)."""
        return get_azure_anthropic_models()

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
        """Create an Azure Anthropic client.

        Args:
            model: Deployment name to use
            temperature: Temperature for responses
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream responses
            base_url: Optional base URL override
            toolboxes: Optional list of LlmToolbox objects
            **kwargs: Additional arguments

        Returns:
            Configured AnthropicClient instance
        """
        if not self.is_available:
            raise ValueError(
                "AZURE_AI_SERVICES_KEY and AZURE_AI_SERVICES_ENDPOINT "
                "environment variables must be set"
            )

        assert self._credentials is not None
        resolved_base = base_url or self._credentials.base_url
        assert resolved_base is not None
        azure_base_url = resolved_base.rstrip("/") + "/anthropic/"

        reasoning = next(
            (m.reasoning for m in get_azure_anthropic_models() if m.model == model),
            True,
        )
        return AnthropicClient(
            api_key=self._credentials.api_key.get_secret_value(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            toolboxes=toolboxes or [],
            azure_base_url=azure_base_url,
            reasoning=reasoning,
        )
