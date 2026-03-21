"""Azure Anthropic provider implementation.

Uses the same AnthropicClient as the standard Anthropic provider but
configured with Azure AI Services endpoints via AnthropicFoundry.
"""
import os
from typing import List, Optional

from llming_models.providers import BaseProvider, register_provider
from llming_models.llm_base_client import LlmClient
from .azure_anthropic_models import AZURE_ANTHROPIC_MODELS, LLMInfo
from llming_models.providers.anthropic.anthropic_client import AnthropicClient
from llming_models.tools.llm_toolbox import LlmToolbox


@register_provider("azure_anthropic")
class AzureAnthropicProvider(BaseProvider):
    """Azure Anthropic provider implementation.

    Uses the same AnthropicClient as the standard Anthropic provider but
    configured with Azure AI Services endpoints.
    """

    def __init__(self):
        """Initialize Azure Anthropic provider."""
        super().__init__("azure_anthropic", "Azure Anthropic")
        self._api_key = os.environ.get('AZURE_AI_SERVICES_KEY')
        self._endpoint = os.environ.get('AZURE_AI_SERVICES_ENDPOINT')

    @property
    def is_available(self) -> bool:
        """Check if provider is available (has valid API key and endpoint)."""
        return self._api_key is not None and self._endpoint is not None

    def get_models(self) -> List[LLMInfo]:
        """Get list of available Azure Anthropic models."""
        return AZURE_ANTHROPIC_MODELS

    def create_client(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        streaming: bool = False,
        base_url: Optional[str] = None,
        toolboxes: Optional[List[LlmToolbox]] = None,
        **kwargs
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

        azure_base_url = (base_url or self._endpoint).rstrip("/") + "/anthropic/"

        return AnthropicClient(
            api_key=self._api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            toolboxes=toolboxes or [],
            azure_base_url=azure_base_url,
        )
