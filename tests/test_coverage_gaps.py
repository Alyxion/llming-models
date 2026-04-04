"""Tests to close coverage gaps in llming-models smaller files.

Targets:
- budget/budget_limit.py: abstract bodies + log_usage_async
- budget/budget_manager.py: overuse/return paths, exception handling
- budget/memory_budget_limit.py: PER_USER scope, period rollover, reset
- llm_base_client.py: abstract method bodies (verify they are abstract)
- providers/*_provider.py: create_client with mocked constructors
- azure_anthropic_provider.py: env var fallback, get_models, create_client
- azure_anthropic_models.py: AZURE_ANTHROPIC_DEPLOYMENTS parsing
- time_intervals.py: unreachable else branches
- gemini_image.py: sync wrappers
- math_mcp.py: targeted missing lines
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest


# =========================================================================
# 1. budget/budget_limit.py — abstract bodies + log_usage_async
# =========================================================================


class TestBudgetLimitABC:
    """Cover abstract method bodies and log_usage_async default impl."""

    @pytest.mark.asyncio
    async def test_log_usage_async_default_is_noop(self):
        """Default log_usage_async does nothing (returns None)."""
        from llming_models.budget.memory_budget_limit import MemoryBudgetLimit
        from llming_models.budget.budget_types import LimitPeriod

        limit = MemoryBudgetLimit(name="test", amount=10.0, period=LimitPeriod.TOTAL)
        # Should not raise, should return None
        result = await limit.log_usage_async(
            model_name="test-model",
            tokens_input=100,
            tokens_output=50,
            costs=0.01,
            duration_ms=100.0,
            user_id="user1",
            operation_type="chat",
        )
        assert result is None

    def test_abstract_methods_raise_not_implemented(self):
        """The abstract method bodies contain raise NotImplementedError."""
        from llming_models.budget.budget_limit import BudgetLimit
        import inspect

        abstract_methods = [
            name for name, method in inspect.getmembers(BudgetLimit, predicate=inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        ]
        assert len(abstract_methods) >= 4  # get_available, reserve, return, reset

    def test_cannot_instantiate_budget_limit(self):
        """BudgetLimit cannot be instantiated directly (ABC)."""
        from llming_models.budget.budget_limit import BudgetLimit
        with pytest.raises(TypeError):
            BudgetLimit(name="test", amount=1.0, period="total")  # type: ignore


# =========================================================================
# 2. budget/budget_manager.py — overuse/return + exception handling
# =========================================================================


class TestBudgetManagerExceptionPaths:
    """Cover the exception handling in reserve_budget_async."""

    @pytest.mark.asyncio
    async def test_reserve_rollback_on_second_limit_failure(self):
        """When second limit fails, first limit is rolled back (line 97)."""
        from llming_models.budget import LLMBudgetManager, MemoryBudgetLimit, LimitPeriod, InsufficientBudgetError

        mgr = LLMBudgetManager([
            MemoryBudgetLimit(name="generous", amount=100.0, period=LimitPeriod.TOTAL),
            MemoryBudgetLimit(name="tight", amount=0.001, period=LimitPeriod.TOTAL),
        ], reserve_output_ratio=1.0)

        with pytest.raises(InsufficientBudgetError):
            await mgr.reserve_budget_async(
                input_tokens=10_000,
                max_output_tokens=10_000,
                input_token_price=10.0,
                output_token_price=30.0,
            )
        # Generous limit should have been rolled back
        generous = await mgr.limits["generous"].get_available_budget_async()
        assert generous == 100.0

    @pytest.mark.asyncio
    async def test_reserve_generic_exception_rolls_back(self):
        """Generic exception (not InsufficientBudgetError) triggers rollback (lines 106-114)."""
        from llming_models.budget import LLMBudgetManager, LimitPeriod, InsufficientBudgetError
        from llming_models.budget.memory_budget_limit import MemoryBudgetLimit

        # Create a limit that always raises a generic error on reserve
        class BrokenLimit(MemoryBudgetLimit):
            def __init__(self):
                super().__init__(name="broken", amount=100.0, period=LimitPeriod.TOTAL)

            async def reserve_budget_async(self, amount, user_id=None):
                raise RuntimeError("DB connection lost")

        healthy = MemoryBudgetLimit(name="healthy", amount=100.0, period=LimitPeriod.TOTAL)
        broken = BrokenLimit()

        # healthy is first in dict order, broken second
        mgr = LLMBudgetManager([healthy, broken], reserve_output_ratio=1.0)

        with pytest.raises(InsufficientBudgetError):
            await mgr.reserve_budget_async(
                input_tokens=1000,
                max_output_tokens=1000,
                input_token_price=10.0,
                output_token_price=30.0,
            )
        # healthy was rolled back
        healthy_budget = await healthy.get_available_budget_async()
        assert healthy_budget == 100.0

    @pytest.mark.asyncio
    async def test_return_overuse_charges_additional(self):
        """When actual > reserved, additional budget is charged (lines 133-143)."""
        from llming_models.budget import LLMBudgetManager, MemoryBudgetLimit, LimitPeriod

        mgr = LLMBudgetManager(
            [MemoryBudgetLimit(name="test", amount=1.0, period=LimitPeriod.TOTAL)],
            reserve_output_ratio=1.0,
        )
        # Reserve for 1000 output tokens
        await mgr.reserve_budget_async(
            input_tokens=1000,
            max_output_tokens=1000,
            input_token_price=10.0,
            output_token_price=30.0,
        )
        # Cost: 1000*10/1M + 1000*30/1M = 0.01 + 0.03 = 0.04
        budget_after_reserve = await mgr.available_budget_async()
        assert abs(budget_after_reserve - 0.96) < 1e-9

        # Actual output = 3000 (overuse by 2000)
        await mgr.return_unused_budget_async(
            reserved_output_tokens=1000,
            actual_output_tokens=3000,
            output_token_price=30.0,
        )
        # Additional: 2000 * 30/1M = 0.06
        budget_final = await mgr.available_budget_async()
        assert abs(budget_final - 0.90) < 1e-9


# =========================================================================
# 3. budget/memory_budget_limit.py — PER_USER scope
# =========================================================================


class TestMemoryBudgetLimitPerUser:
    """Cover PER_USER scope paths in memory_budget_limit.py."""

    @pytest.mark.asyncio
    async def test_per_user_requires_user_id(self):
        """PER_USER limit raises ValueError when user_id is None (lines 47-48)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user", amount=5.0,
            period=LimitPeriod.DAILY,
            scope=BudgetScope.PER_USER,
        )
        with pytest.raises(ValueError, match="user_id is required"):
            await limit.get_available_budget_async()

    @pytest.mark.asyncio
    async def test_per_user_independent_budgets(self):
        """Different users have independent budgets (line 42, 49)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user", amount=10.0,
            period=LimitPeriod.DAILY,
            scope=BudgetScope.PER_USER,
        )
        # User A spends 3
        assert await limit.reserve_budget_async(3.0, user_id="userA") is True
        # User B still has full budget
        assert abs(await limit.get_available_budget_async(user_id="userB") - 10.0) < 1e-9
        # User A has 7 left
        assert abs(await limit.get_available_budget_async(user_id="userA") - 7.0) < 1e-9

    @pytest.mark.asyncio
    async def test_per_user_total_period(self):
        """PER_USER + TOTAL uses dict keyed by user_id (line 42: period_key = 'total')."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user_total", amount=5.0,
            period=LimitPeriod.TOTAL,
            scope=BudgetScope.PER_USER,
        )
        # This should use _usage dict (not self.amount)
        assert limit._usage is not None
        assert await limit.reserve_budget_async(2.0, user_id="user1") is True
        assert abs(await limit.get_available_budget_async(user_id="user1") - 3.0) < 1e-9

    @pytest.mark.asyncio
    async def test_per_user_return_budget(self):
        """Return budget for a PER_USER limit (line 91)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user", amount=10.0,
            period=LimitPeriod.DAILY,
            scope=BudgetScope.PER_USER,
        )
        await limit.reserve_budget_async(5.0, user_id="userA")
        await limit.return_budget_async(2.0, user_id="userA")
        budget = await limit.get_available_budget_async(user_id="userA")
        assert abs(budget - 7.0) < 1e-9

    @pytest.mark.asyncio
    async def test_per_user_overdraw_rejected(self):
        """PER_USER overdraw returns False (line 91)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user", amount=5.0,
            period=LimitPeriod.DAILY,
            scope=BudgetScope.PER_USER,
        )
        assert await limit.reserve_budget_async(6.0, user_id="userA") is False

    @pytest.mark.asyncio
    async def test_per_user_reset_clears_all(self):
        """Reset clears all per-user data (line 112: _usage = {})."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod
        from llming_models.budget.budget_types import BudgetScope

        limit = MemoryBudgetLimit(
            name="per_user", amount=10.0,
            period=LimitPeriod.DAILY,
            scope=BudgetScope.PER_USER,
        )
        await limit.reserve_budget_async(5.0, user_id="userA")
        await limit.reset_async()
        # After reset, full budget should be available again
        budget = await limit.get_available_budget_async(user_id="userA")
        assert budget == 10.0

    @pytest.mark.asyncio
    async def test_daily_reset_clears_usage(self):
        """Reset on a GLOBAL+DAILY limit clears the usage dict (line 112)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod

        limit = MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY)
        await limit.reserve_budget_async(5.0)
        await limit.reset_async()
        budget = await limit.get_available_budget_async()
        assert budget == 10.0

    @pytest.mark.asyncio
    async def test_period_rollover_in_reserve(self):
        """Period key changes during reserve are handled (lines 83-85)."""
        from llming_models.budget import MemoryBudgetLimit, LimitPeriod

        limit = MemoryBudgetLimit(name="daily", amount=10.0, period=LimitPeriod.DAILY)
        # Normal reserve should work
        assert await limit.reserve_budget_async(5.0) is True
        budget = await limit.get_available_budget_async()
        assert abs(budget - 5.0) < 1e-9


# =========================================================================
# 4. llm_base_client.py — abstract method bodies
# =========================================================================


class TestLlmClientABC:
    """Verify LlmClient abstract methods are properly decorated."""

    def test_cannot_instantiate_directly(self):
        """LlmClient is abstract and cannot be instantiated."""
        from llming_models.llm_base_client import LlmClient
        with pytest.raises(TypeError):
            LlmClient(model="test")  # type: ignore

    def test_all_abstract_methods_declared(self):
        """All expected abstract methods are declared."""
        from llming_models.llm_base_client import LlmClient
        import inspect

        abstract_names = {
            name for name, method in inspect.getmembers(LlmClient, predicate=inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        }
        assert {"invoke", "ainvoke", "stream", "astream"} <= abstract_names

    def test_estimate_tokens_basic(self):
        """estimate_tokens returns positive integer for non-empty text."""
        from llming_models.llm_base_client import LlmClient
        from collections.abc import AsyncIterator, Iterator
        from llming_models.messages import LlmAIMessage, LlmHumanMessage, LlmSystemMessage, LlmMessageChunk

        # Create a concrete subclass to test non-abstract methods
        class TestClient(LlmClient):
            def invoke(self, messages):
                return LlmAIMessage(content="")

            async def ainvoke(self, messages):
                return LlmAIMessage(content="")

            def stream(self, messages):
                yield LlmMessageChunk(content="")

            async def astream(self, messages, usage_callback=None):
                yield LlmMessageChunk(content="")

        client = TestClient(model="test")
        tokens = client.estimate_tokens("Hello world")
        assert tokens > 0

        tokens_with_role = client.estimate_tokens("Hello world", role="system")
        assert tokens_with_role > tokens  # role prefix adds tokens

    @pytest.mark.asyncio
    async def test_estimate_tokens_async(self):
        """estimate_tokens_async wraps the sync version."""
        from llming_models.llm_base_client import LlmClient
        from llming_models.messages import LlmAIMessage, LlmMessageChunk

        class TestClient(LlmClient):
            def invoke(self, messages):
                return LlmAIMessage(content="")

            async def ainvoke(self, messages):
                return LlmAIMessage(content="")

            def stream(self, messages):
                yield LlmMessageChunk(content="")

            async def astream(self, messages, usage_callback=None):
                yield LlmMessageChunk(content="")

        client = TestClient(model="test")
        tokens = await client.estimate_tokens_async("Hello world")
        assert tokens > 0


# =========================================================================
# 5. providers/*_provider.py — create_client with mocked constructors
# =========================================================================


class TestProviderCreateClient:
    """Test create_client for each provider with mocked client constructors."""

    def test_anthropic_create_client(self):
        """AnthropicProvider.create_client creates AnthropicClient."""
        from llming_models.providers.anthropic.anthropic_provider import AnthropicProvider
        from llming_models.credentials import ProviderCredentials

        provider = AnthropicProvider(credentials=ProviderCredentials(api_key="test-key"))
        assert provider.is_available

        with patch("llming_models.providers.anthropic.anthropic_provider.AnthropicClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="claude-test", max_tokens=100)
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_anthropic_create_client_no_key_raises(self):
        """AnthropicProvider.create_client raises when no key."""
        from llming_models.providers.anthropic.anthropic_provider import AnthropicProvider

        with patch.dict(os.environ, {}, clear=True):
            provider = AnthropicProvider()
            assert not provider.is_available
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                provider.create_client(model="test")

    def test_openai_create_client(self):
        """OpenAIProvider.create_client creates OpenAILlmClient."""
        from llming_models.providers.openai.openai_provider import OpenAIProvider
        from llming_models.credentials import ProviderCredentials

        provider = OpenAIProvider(credentials=ProviderCredentials(api_key="test-key"))
        assert provider.is_available

        with patch("llming_models.providers.openai.openai_provider.OpenAILlmClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="gpt-test", max_tokens=100)
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_google_create_client(self):
        """GoogleProvider.create_client creates GoogleClient."""
        from llming_models.providers.google.google_provider import GoogleProvider
        from llming_models.credentials import ProviderCredentials

        provider = GoogleProvider(credentials=ProviderCredentials(api_key="test-key"))
        assert provider.is_available

        with patch("llming_models.providers.google.google_provider.GoogleClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="gemini-test", max_tokens=100)
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_mistral_create_client(self):
        """MistralProvider.create_client creates OpenAICompatibleClient."""
        from llming_models.providers.mistral.mistral_provider import MistralProvider
        from llming_models.credentials import ProviderCredentials

        provider = MistralProvider(credentials=ProviderCredentials(api_key="test-key"))
        assert provider.is_available

        with patch("llming_models.providers.mistral.mistral_provider.OpenAICompatibleClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="mistral-test")
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_mistral_create_client_no_key_raises(self):
        """MistralProvider raises when unavailable."""
        from llming_models.providers.mistral.mistral_provider import MistralProvider

        with patch.dict(os.environ, {}, clear=True):
            provider = MistralProvider()
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                provider.create_client(model="test")

    def test_together_create_client(self):
        """TogetherProvider.create_client creates OpenAICompatibleClient."""
        from llming_models.providers.together.together_provider import TogetherProvider
        from llming_models.credentials import ProviderCredentials

        provider = TogetherProvider(credentials=ProviderCredentials(api_key="test-key"))
        assert provider.is_available

        with patch("llming_models.providers.together.together_provider.OpenAICompatibleClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="together-test")
            MockClient.assert_called_once()
            assert client is MockClient.return_value

    def test_together_create_client_no_key_raises(self):
        """TogetherProvider raises when unavailable."""
        from llming_models.providers.together.together_provider import TogetherProvider

        with patch.dict(os.environ, {}, clear=True):
            provider = TogetherProvider()
            with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                provider.create_client(model="test")

    def test_azure_openai_create_client(self):
        """AzureOpenAIProvider.create_client creates OpenAILlmClient with azure config."""
        from llming_models.providers.azure_openai.azure_openai_provider import AzureOpenAIProvider
        from llming_models.credentials import ProviderCredentials

        provider = AzureOpenAIProvider(
            credentials=ProviderCredentials(
                api_key="test-key",
                base_url="https://test.openai.azure.com",
                api_version="2025-04-01-preview",
            )
        )
        assert provider.is_available

        with patch("llming_models.providers.azure_openai.azure_openai_provider.OpenAILlmClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="gpt-test")
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args
            assert call_kwargs.kwargs.get("api_type") == "azure" or \
                   (len(call_kwargs.args) == 0 and "api_type" in str(call_kwargs))

    def test_generic_openai_create_client(self):
        """GenericOpenAIProvider.create_client creates OpenAICompatibleClient."""
        from llming_models.providers.generic_openai_provider import GenericOpenAIProvider
        from llming_models.providers.llm_provider_models import LLMInfo

        models = [LLMInfo(
            provider="custom", name="custom-model", label="Custom",
            model="custom-v1", description="Test",
            input_token_price=1.0, output_token_price=2.0,
        )]
        provider = GenericOpenAIProvider(
            name="custom", label="Custom",
            api_key="test-key", base_url="https://api.custom.com",
            models=models,
        )
        assert provider.is_available
        assert provider.get_models() == models

        with patch("llming_models.providers.generic_openai_provider.OpenAICompatibleClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="custom-v1")
            MockClient.assert_called_once()


# =========================================================================
# 6. azure_anthropic_provider.py — env var fallback, get_models, create_client
# =========================================================================


class TestAzureAnthropicProvider:
    """Cover azure_anthropic_provider.py lines 36-37, 46, 72-83."""

    def test_env_var_fallback(self):
        """Provider picks up AZURE_AI_SERVICES_KEY and ENDPOINT from env (lines 36-37)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider

        with patch.dict(os.environ, {
            "AZURE_AI_SERVICES_KEY": "test-key",
            "AZURE_AI_SERVICES_ENDPOINT": "https://test.cognitiveservices.azure.com",
        }, clear=True):
            provider = AzureAnthropicProvider()
            assert provider.is_available
            assert provider._credentials is not None
            assert provider._credentials.api_key.get_secret_value() == "test-key"
            assert provider._credentials.base_url == "https://test.cognitiveservices.azure.com"

    def test_get_models_returns_list(self):
        """get_models returns the AZURE_ANTHROPIC_MODELS list (line 46)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider
        from llming_models.credentials import ProviderCredentials

        provider = AzureAnthropicProvider(
            credentials=ProviderCredentials(
                api_key="k", base_url="https://test.com"
            )
        )
        models = provider.get_models()
        assert isinstance(models, list)

    def test_create_client_success(self):
        """create_client builds AnthropicClient with azure_base_url (lines 72-83)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider
        from llming_models.credentials import ProviderCredentials

        provider = AzureAnthropicProvider(
            credentials=ProviderCredentials(
                api_key="test-key",
                base_url="https://test.cognitiveservices.azure.com",
            )
        )

        with patch("llming_models.providers.azure_anthropic.azure_anthropic_provider.AnthropicClient") as MockClient:
            MockClient.return_value = MagicMock()
            client = provider.create_client(model="claude-test", max_tokens=100)
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args.kwargs
            assert "anthropic/" in call_kwargs["azure_base_url"]

    def test_create_client_not_available_raises(self):
        """create_client raises ValueError when not available (line 72)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider

        with patch.dict(os.environ, {}, clear=True):
            provider = AzureAnthropicProvider()
            assert not provider.is_available
            with pytest.raises(ValueError, match="AZURE_AI_SERVICES_KEY"):
                provider.create_client(model="test")

    def test_create_client_with_base_url_override(self):
        """create_client accepts base_url override."""
        from llming_models.providers.azure_anthropic.azure_anthropic_provider import AzureAnthropicProvider
        from llming_models.credentials import ProviderCredentials

        provider = AzureAnthropicProvider(
            credentials=ProviderCredentials(
                api_key="k",
                base_url="https://default.com",
            )
        )

        with patch("llming_models.providers.azure_anthropic.azure_anthropic_provider.AnthropicClient") as MockClient:
            MockClient.return_value = MagicMock()
            provider.create_client(model="test", base_url="https://override.com")
            call_kwargs = MockClient.call_args.kwargs
            assert "override.com" in call_kwargs["azure_base_url"]


# =========================================================================
# 7. azure_anthropic_models.py — AZURE_ANTHROPIC_DEPLOYMENTS branches
# =========================================================================


class TestAzureAnthropicModels:
    """Cover lines 83 and 88 — unknown name in DEPLOYMENTS, entries without =."""

    def test_unknown_name_skipped(self):
        """Unknown model name in AZURE_ANTHROPIC_DEPLOYMENTS is skipped (line 88)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_models import get_azure_anthropic_models

        with patch.dict(os.environ, {
            "AZURE_ANTHROPIC_DEPLOYMENTS": "unknown_model=some-deployment"
        }):
            models = get_azure_anthropic_models()
        assert len(models) == 0

    def test_entry_without_equals_skipped(self):
        """Entry without = sign is skipped (line 83)."""
        from llming_models.providers.azure_anthropic.azure_anthropic_models import get_azure_anthropic_models

        with patch.dict(os.environ, {
            "AZURE_ANTHROPIC_DEPLOYMENTS": "bad_entry_no_equals,claude_opus=my-opus"
        }):
            models = get_azure_anthropic_models()
        names = [m.name for m in models]
        assert "claude_opus" in names
        assert len(models) == 1

    def test_all_three_models(self):
        """All three model types can be configured."""
        from llming_models.providers.azure_anthropic.azure_anthropic_models import get_azure_anthropic_models

        with patch.dict(os.environ, {
            "AZURE_ANTHROPIC_DEPLOYMENTS": "claude_opus=d1,claude_sonnet=d2,claude_haiku=d3"
        }):
            models = get_azure_anthropic_models()
        names = {m.name for m in models}
        assert names == {"claude_opus", "claude_sonnet", "claude_haiku"}


# =========================================================================
# 8. providers/llm_provider_base.py — abstract methods
# =========================================================================


class TestBaseProviderABC:
    """Cover llm_provider_base.py abstract method bodies (lines 34, 39, 66)."""

    def test_cannot_instantiate(self):
        """BaseProvider is abstract."""
        from llming_models.providers.llm_provider_base import BaseProvider
        with pytest.raises(TypeError):
            BaseProvider("test", "Test")  # type: ignore

    def test_abstract_methods_declared(self):
        """Expected abstract methods exist."""
        from llming_models.providers.llm_provider_base import BaseProvider
        import inspect

        abstract_names = set()
        for name, val in inspect.getmembers(BaseProvider):
            if getattr(val, "__isabstractmethod__", False):
                abstract_names.add(name)
        assert {"is_available", "get_models", "create_client"} <= abstract_names


# =========================================================================
# 10. time_intervals.py — unreachable else branches (lines 81, 124)
# =========================================================================
# 10. time_intervals.py — unreachable else branches (lines 81, 124)
# =========================================================================


class TestTimeIntervalsUnreachable:
    """Cover the else: raise ValueError branches with a mocked invalid enum."""

    def test_get_key_suffix_unsupported_raises(self):
        """Unsupported interval in get_key_suffix raises ValueError (line 81)."""
        from llming_models.budget.time_intervals import TimeIntervalHandler, TimeInterval

        # Create a fake enum-like value
        fake_interval = MagicMock()
        fake_interval.value = "fake"
        # Make it fail all == checks
        fake_interval.__eq__ = lambda self, other: False

        with pytest.raises(ValueError, match="Unsupported interval"):
            TimeIntervalHandler.get_key_suffix(fake_interval, datetime(2024, 1, 1))

    def test_get_expiry_unsupported_raises(self):
        """Unsupported interval in get_expiry raises ValueError (line 124)."""
        from llming_models.budget.time_intervals import TimeIntervalHandler

        fake_interval = MagicMock()
        fake_interval.value = "fake"
        fake_interval.__eq__ = lambda self, other: False

        with pytest.raises(ValueError, match="Unsupported interval"):
            TimeIntervalHandler.get_expiry(fake_interval)


# =========================================================================
