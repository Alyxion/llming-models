"""Generic provider for any OpenAI-compatible API endpoint.

Register custom providers at runtime via LLMManager::

    from llming_models.llm_provider_manager import LLMManager
    from llming_models.providers.llm_provider_models import LLMInfo

    manager = LLMManager()
    manager.register_openai_compatible(
        name="deepseek",
        label="DeepSeek",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        models=[
            LLMInfo(
                provider="deepseek", name="deepseek_chat",
                label="DeepSeek V3", model="deepseek-chat",
                description="Optimized for conversational interactions.",
                input_token_price=0.5, output_token_price=1.5,
            ),
        ],
    )
"""
from __future__ import annotations

from typing import Any

from llming_models.providers.llm_provider_base import BaseProvider
from llming_models.llm_base_client import LlmClient
from llming_models.providers.llm_provider_models import LLMInfo
from llming_models.providers.openai_compat_client import OpenAICompatibleClient
from llming_models.tools.llm_toolbox import LlmToolbox


class GenericOpenAIProvider(BaseProvider):
    """Provider for any OpenAI-compatible API endpoint.

    Unlike the built-in providers which read API keys from environment
    variables, this provider accepts all configuration explicitly.
    """

    def __init__(
        self,
        name: str,
        label: str,
        api_key: str,
        base_url: str,
        models: list[LLMInfo],
    ):
        from llming_models.credentials import ProviderCredentials
        super().__init__(name, label, ProviderCredentials(api_key=api_key, base_url=base_url))
        self._models = models

    @property
    def is_available(self) -> bool:
        return self._credentials is not None and bool(self._credentials.api_key.get_secret_value())

    def get_models(self) -> list[LLMInfo]:
        return self._models

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
        assert self._credentials is not None
        return OpenAICompatibleClient(
            api_key=self._credentials.api_key.get_secret_value(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=base_url or self._credentials.base_url,
        )
