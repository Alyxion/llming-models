"""Credential models for LLM providers.

Allows injecting API keys programmatically instead of relying solely on
environment variables.  When no explicit credentials are supplied the
providers fall back to reading os.environ as before — zero breaking changes.
"""
from __future__ import annotations

from pydantic import BaseModel, SecretStr


class ProviderCredentials(BaseModel):
    """API credentials for a single provider."""
    api_key: SecretStr = SecretStr("")
    base_url: str | None = None
    api_version: str | None = None
    organization: str | None = None


class LLMCredentials(BaseModel):
    """Credentials for all providers.  Each field matches a provider name."""
    anthropic: ProviderCredentials | None = None
    openai: ProviderCredentials | None = None
    azure_openai: ProviderCredentials | None = None
    azure_anthropic: ProviderCredentials | None = None
    google: ProviderCredentials | None = None
    mistral: ProviderCredentials | None = None
    together: ProviderCredentials | None = None
    elevenlabs: ProviderCredentials | None = None

    _PROVIDERS = frozenset((
        "anthropic", "openai", "azure_openai", "azure_anthropic",
        "google", "mistral", "together", "elevenlabs",
    ))

    def for_provider(self, name: str) -> ProviderCredentials | None:
        """Look up credentials by provider name."""
        if name not in self._PROVIDERS:
            return None
        return self.__dict__.get(name)
