"""Integration tests for all LLM providers — hits real APIs.

Each test is skipped if the required env vars are not set.
Run with: poetry run python -m pytest tests/test_providers_integration.py -v

Env vars loaded from project .env via conftest.py.
"""
import os
import logging

import pytest

from llming_models.messages import LlmSystemMessage, LlmHumanMessage
from llming_models.session import ChatSession, LLMConfig
from llming_models.llm_provider_manager import LLMManager
from llming_models.credentials import ProviderCredentials, LLMCredentials

logger = logging.getLogger(__name__)

SIMPLE_PROMPT = "Reply with exactly: INTEGRATION_TEST_OK"


# ---------------------------------------------------------------------------
# Helper: stream and collect
# ---------------------------------------------------------------------------
async def _stream_and_collect(client, messages):
    """Stream a response and return (text, usage_dict)."""
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}

    def cb(inp, out, cached_input_tokens=0):
        usage["input_tokens"] += inp
        usage["output_tokens"] += out
        usage["cached_input_tokens"] += cached_input_tokens

    text_parts = []
    async for chunk in client.astream(messages, usage_callback=cb):
        if chunk.content:
            text_parts.append(chunk.content)

    return "".join(text_parts), usage


# =========================================================================
# Anthropic (direct)
# =========================================================================

_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_anthropic, reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_invoke():
    """Anthropic non-streaming invoke returns a response."""
    from llming_models.providers.anthropic.anthropic_client import AnthropicClient
    client = AnthropicClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    response = client.invoke(msgs)
    assert "INTEGRATION_TEST_OK" in response.content
    assert response.response_metadata.get("input_tokens", 0) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_anthropic, reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_stream():
    """Anthropic streaming returns text and usage."""
    from llming_models.providers.anthropic.anthropic_client import AnthropicClient
    client = AnthropicClient(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_anthropic, reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_provider():
    """AnthropicProvider creates a working client via credentials."""
    from llming_models.providers.anthropic.anthropic_provider import AnthropicProvider
    provider = AnthropicProvider(
        credentials=ProviderCredentials(api_key=os.environ["ANTHROPIC_API_KEY"])
    )
    assert provider.is_available
    client = provider.create_client(model="claude-haiku-4-5-20251001", max_tokens=64)
    response = client.invoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


# =========================================================================
# Azure Anthropic
# =========================================================================

_has_azure_anthropic = bool(
    os.environ.get("AZURE_AI_SERVICES_KEY") and os.environ.get("AZURE_AI_SERVICES_ENDPOINT")
)
_azure_anthropic_deployment = os.environ.get("AZURE_ANTHROPIC_DEPLOYMENTS", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_anthropic, reason="AZURE_AI_SERVICES_KEY/ENDPOINT not set")
async def test_azure_anthropic_invoke():
    """Azure Anthropic returns a response via AnthropicFoundry."""
    from llming_models.providers.anthropic.anthropic_client import AnthropicClient
    endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/") + "/anthropic/"
    client = AnthropicClient(
        api_key=os.environ["AZURE_AI_SERVICES_KEY"],
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        azure_base_url=endpoint,
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    response = client.invoke(msgs)
    assert "INTEGRATION_TEST_OK" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_anthropic, reason="AZURE_AI_SERVICES_KEY/ENDPOINT not set")
async def test_azure_anthropic_stream():
    """Azure Anthropic streaming returns text and usage."""
    from llming_models.providers.anthropic.anthropic_client import AnthropicClient
    endpoint = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/") + "/anthropic/"
    client = AnthropicClient(
        api_key=os.environ["AZURE_AI_SERVICES_KEY"],
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        azure_base_url=endpoint,
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_anthropic, reason="AZURE_AI_SERVICES_KEY/ENDPOINT not set")
async def test_azure_anthropic_provider():
    """AzureAnthropicProvider creates a working client via credentials."""
    from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider
    provider = AzureAnthropicProvider(
        credentials=ProviderCredentials(
            api_key=os.environ["AZURE_AI_SERVICES_KEY"],
            base_url=os.environ["AZURE_AI_SERVICES_ENDPOINT"],
        )
    )
    assert provider.is_available
    client = provider.create_client(model="claude-haiku-4-5-20251001", max_tokens=64)
    response = client.invoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


# =========================================================================
# OpenAI (direct)
# =========================================================================

_has_openai = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai, reason="OPENAI_API_KEY not set")
async def test_openai_invoke():
    """OpenAI non-streaming invoke returns a response."""
    from llming_models.providers.openai.openai_client import OpenAILlmClient
    client = OpenAILlmClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=256,
        api_type="openai",
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    response = await client.ainvoke(msgs)
    assert "INTEGRATION_TEST_OK" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai, reason="OPENAI_API_KEY not set")
async def test_openai_stream():
    """OpenAI streaming returns text and usage."""
    from llming_models.providers.openai.openai_client import OpenAILlmClient
    client = OpenAILlmClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=256,
        api_type="openai",
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai, reason="OPENAI_API_KEY not set")
async def test_openai_provider():
    """OpenAIProvider creates a working client via credentials."""
    from llming_models.providers.openai.openai_provider import OpenAIProvider
    provider = OpenAIProvider(
        credentials=ProviderCredentials(api_key=os.environ["OPENAI_API_KEY"])
    )
    assert provider.is_available
    client = provider.create_client(model="gpt-5-nano", max_tokens=256)
    response = await client.ainvoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


# =========================================================================
# Azure OpenAI
# =========================================================================

_has_azure_openai = bool(
    os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT")
)


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_openai, reason="AZURE_OPENAI_API_KEY/ENDPOINT not set")
async def test_azure_openai_invoke():
    """Azure OpenAI non-streaming invoke returns a response."""
    from llming_models.providers.openai.openai_client import OpenAILlmClient
    client = OpenAILlmClient(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=256,
        api_type="azure",
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    response = await client.ainvoke(msgs)
    assert "INTEGRATION_TEST_OK" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_openai, reason="AZURE_OPENAI_API_KEY/ENDPOINT not set")
async def test_azure_openai_stream():
    """Azure OpenAI streaming returns text and usage."""
    from llming_models.providers.openai.openai_client import OpenAILlmClient
    client = OpenAILlmClient(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        model="gpt-5-nano",
        max_tokens=256,
        api_type="azure",
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_azure_openai, reason="AZURE_OPENAI_API_KEY/ENDPOINT not set")
async def test_azure_openai_provider():
    """AzureOpenAIProvider creates a working client via credentials."""
    from llming_models.providers.azure_openai.azure_openai_provider import AzureOpenAIProvider
    provider = AzureOpenAIProvider(
        credentials=ProviderCredentials(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    )
    assert provider.is_available
    client = provider.create_client(model="gpt-5-nano", max_tokens=256)
    response = await client.ainvoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


# =========================================================================
# Google (Gemini)
# =========================================================================

_has_google = bool(os.environ.get("GEMINI_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_google, reason="GEMINI_KEY not set")
async def test_google_invoke():
    """Google Gemini non-streaming invoke returns a response."""
    from llming_models.providers.google.google_client import GoogleClient
    client = GoogleClient(
        api_key=os.environ["GEMINI_KEY"],
        model="gemini-2.0-flash",
        max_tokens=64,
    )
    response = await client.ainvoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_google, reason="GEMINI_KEY not set")
async def test_google_stream():
    """Google Gemini streaming returns text and usage (the fix!)."""
    from llming_models.providers.google.google_client import GoogleClient
    client = GoogleClient(
        api_key=os.environ["GEMINI_KEY"],
        model="gemini-2.0-flash",
        max_tokens=64,
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0, "Google streaming should now report input tokens"
    assert usage["output_tokens"] > 0, "Google streaming should now report output tokens"


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_google, reason="GEMINI_KEY not set")
async def test_google_provider():
    """GoogleProvider creates a working client via credentials."""
    from llming_models.providers.google.google_provider import GoogleProvider
    provider = GoogleProvider(
        credentials=ProviderCredentials(api_key=os.environ["GEMINI_KEY"])
    )
    assert provider.is_available
    client = provider.create_client(model="gemini-2.0-flash", max_tokens=64)
    response = await client.ainvoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


# =========================================================================
# Mistral
# =========================================================================

_has_mistral = bool(os.environ.get("MISTRAL_API_KEY"))


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_mistral, reason="MISTRAL_API_KEY not set")
async def test_mistral_invoke():
    """Mistral non-streaming invoke returns a response."""
    from llming_models.providers.openai_compat_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(
        api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-small-latest",
        max_tokens=64,
        base_url="https://api.mistral.ai/v1",
    )
    response = await client.ainvoke([LlmHumanMessage(content=SIMPLE_PROMPT)])
    assert "INTEGRATION_TEST_OK" in response.content


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_mistral, reason="MISTRAL_API_KEY not set")
async def test_mistral_stream():
    """Mistral streaming returns text and usage."""
    from llming_models.providers.openai_compat_client import OpenAICompatibleClient
    client = OpenAICompatibleClient(
        api_key=os.environ["MISTRAL_API_KEY"],
        model="mistral-small-latest",
        max_tokens=64,
        base_url="https://api.mistral.ai/v1",
    )
    msgs = [LlmHumanMessage(content=SIMPLE_PROMPT)]
    text, usage = await _stream_and_collect(client, msgs)
    assert "INTEGRATION_TEST_OK" in text
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0


# =========================================================================
# LLMManager — credential injection
# =========================================================================


def test_manager_env_fallback():
    """LLMManager without credentials uses env vars."""
    mgr = LLMManager()
    # We don't assert specific providers since env varies,
    # just check it doesn't crash
    assert isinstance(mgr.providers, dict)


@pytest.mark.skipif(not _has_anthropic, reason="ANTHROPIC_API_KEY not set")
def test_manager_explicit_credentials():
    """LLMManager with explicit credentials makes provider available."""
    creds = LLMCredentials(
        anthropic=ProviderCredentials(api_key=os.environ["ANTHROPIC_API_KEY"]),
    )
    mgr = LLMManager(credentials=creds)
    assert "anthropic" in mgr.providers


def test_manager_bad_key_makes_provider_available_but_calls_fail():
    """A provider is 'available' with any key — the key is validated on API call."""
    creds = LLMCredentials(
        anthropic=ProviderCredentials(api_key="sk-not-real"),
    )
    mgr = LLMManager(credentials=creds)
    assert "anthropic" in mgr.providers


# =========================================================================
# Provider cascade and config
# =========================================================================


def test_provider_cascade_order():
    """azure_anthropic appears before anthropic in the default cascade."""
    from llming_models.config import LLMGlobalConfig
    config = LLMGlobalConfig()
    cascade = config.provider_cascade
    assert "azure_anthropic" in cascade
    assert "anthropic" in cascade
    assert cascade.index("azure_anthropic") < cascade.index("anthropic")


def test_provider_unavailable_without_env(monkeypatch):
    """Providers report unavailable when env vars are missing."""
    from unittest.mock import patch
    from llming_models.providers.anthropic.anthropic_provider import AnthropicProvider
    from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider

    with patch.dict(os.environ, {}, clear=True):
        assert not AnthropicProvider().is_available
        assert not AzureAnthropicProvider().is_available


def test_azure_anthropic_deployments_env_parsing():
    """AZURE_ANTHROPIC_DEPLOYMENTS env var configures available models."""
    from unittest.mock import patch
    from llming_models.providers.azure_anthropic.azure_anthropic_models import get_azure_anthropic_models

    with patch.dict(os.environ, {
        "AZURE_ANTHROPIC_DEPLOYMENTS": "claude_opus=my-opus,claude_haiku=my-opus"
    }):
        models = get_azure_anthropic_models()

    names = [m.name for m in models]
    assert "claude_opus" in names
    assert "claude_haiku" in names
    assert "claude_sonnet" not in names
    assert all(m.model == "my-opus" for m in models)


def test_azure_anthropic_deployments_env_empty():
    """No models when AZURE_ANTHROPIC_DEPLOYMENTS is unset."""
    from unittest.mock import patch
    from llming_models.providers.azure_anthropic.azure_anthropic_models import get_azure_anthropic_models

    with patch.dict(os.environ, {}, clear=True):
        models = get_azure_anthropic_models()
    assert models == []
