"""Tests for LLMManager with mocked providers (no real API calls)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llming_models.config import LLMGlobalConfig, LLMUserConfig
from llming_models.credentials import LLMCredentials, ProviderCredentials
from llming_models.llm_provider_manager import LLMManager
from llming_models.model_info import LLMInfo, ModelSize
from llming_models.providers import PROVIDERS
from llming_models.providers.llm_provider_base import BaseProvider
from llming_models.session import ChatSession, LLMConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_info(
    provider: str = "fake",
    name: str = "fake_model",
    model: str = "fake-model-v1",
    label: str = "Fake Model",
    *,
    api_base: str | None = None,
    max_input_tokens: int = 16000,
    max_output_tokens: int = 4096,
    hosting_icon: str | None = None,
) -> LLMInfo:
    return LLMInfo(
        provider=provider,
        name=name,
        model=model,
        label=label,
        description="A fake model for testing",
        input_token_price=1.0,
        output_token_price=2.0,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        api_base=api_base,
        hosting_icon=hosting_icon,
    )


class FakeProvider(BaseProvider):
    """Minimal provider for testing."""

    def __init__(
        self,
        name: str = "fake",
        label: str = "Fake",
        available: bool = True,
        models: list[LLMInfo] | None = None,
        credentials=None,
    ):
        super().__init__(name, label, credentials=credentials)
        self._available = available
        self._models = models or [_make_model_info(provider=name)]

    @property
    def is_available(self) -> bool:
        return self._available

    def get_models(self) -> list[LLMInfo]:
        return self._models

    def create_client(self, model, temperature=0.7, max_tokens=None,
                      streaming=False, base_url=None, toolboxes=None, **kwargs):
        return MagicMock()


def _empty_manager(**kwargs) -> LLMManager:
    """Create an LLMManager with no providers in cascade."""
    global_cfg = kwargs.pop("global_config", LLMGlobalConfig(provider_cascade=[]))
    user_cfg = kwargs.pop("user_config", LLMUserConfig(global_config=global_cfg))
    return LLMManager(user_config=user_cfg, **kwargs)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestLLMManagerInit:
    """Tests for LLMManager.__init__."""

    def test_init_no_providers_available(self):
        """When the cascade references a non-existent provider, no providers should be active."""
        global_cfg = LLMGlobalConfig(provider_cascade=["fake_no_exist"])
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        assert len(mgr.providers) == 0

    def test_init_with_mocked_provider_in_cascade(self):
        """When a fake provider is in the global PROVIDERS dict and the cascade,
        __init__ picks it up automatically."""

        class _InitTestProvider(FakeProvider):
            def __init__(self, credentials=None):
                super().__init__(name="test_prov", label="TestProv", credentials=credentials)

        global_cfg = LLMGlobalConfig(provider_cascade=["test_prov"])
        user_cfg = LLMUserConfig(global_config=global_cfg)

        with patch.dict(PROVIDERS, {"test_prov": _InitTestProvider}, clear=False):
            mgr = LLMManager(user_config=user_cfg)

        assert "test_prov" in mgr.providers
        assert isinstance(mgr.providers["test_prov"], _InitTestProvider)

    def test_init_defaults(self):
        """Default init without arguments creates empty config."""
        with patch.dict(os.environ, {}, clear=True):
            mgr = _empty_manager()
        assert mgr.user_config is not None
        assert mgr.credentials is None
        assert mgr.budget_manager is None
        assert isinstance(mgr.providers, dict)
        assert isinstance(mgr.provider_cascade, list)

    def test_init_with_credentials(self):
        """Passing LLMCredentials stores them on the manager."""
        creds = LLMCredentials(
            anthropic=ProviderCredentials(api_key="test-key-anthropic"),
        )
        mgr = _empty_manager(credentials=creds)
        assert mgr.credentials is creds

    def test_init_skips_unavailable_provider(self):
        """Provider whose is_available returns False is not added to providers dict."""
        from llming_models.providers import PROVIDERS

        class UnavailableProvider(FakeProvider):
            def __init__(self, credentials=None):
                super().__init__(name="unavail", label="Unavail", available=False, credentials=credentials)

        global_cfg = LLMGlobalConfig(provider_cascade=["unavail"])
        user_cfg = LLMUserConfig(global_config=global_cfg)
        with patch.dict(PROVIDERS, {"unavail": UnavailableProvider}, clear=False):
            mgr = LLMManager(user_config=user_cfg)
        assert "unavail" not in mgr.providers

    def test_init_skips_provider_with_no_supported_models(self):
        """Provider with models that are all excluded is not added."""
        from llming_models.providers import PROVIDERS

        class NarrowProvider(FakeProvider):
            def __init__(self, credentials=None):
                super().__init__(
                    name="narrow",
                    label="Narrow",
                    available=True,
                    models=[_make_model_info(provider="narrow", name="blocked_model")],
                    credentials=credentials,
                )

        global_cfg = LLMGlobalConfig(provider_cascade=["narrow"])
        # Exclude the model at user level
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            excluded_models=["narrow:blocked_model"],
        )
        with patch.dict(PROVIDERS, {"narrow": NarrowProvider}, clear=False):
            mgr = LLMManager(user_config=user_cfg)
        assert "narrow" not in mgr.providers

    def test_init_provider_cascade_order_preserved(self):
        """The provider_cascade list preserves the order from global config."""
        global_cfg = LLMGlobalConfig(provider_cascade=["alpha", "beta", "gamma"])
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        assert mgr.provider_cascade == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# register_provider
# ---------------------------------------------------------------------------


class TestRegisterProvider:
    """Tests for LLMManager.register_provider."""

    def test_register_adds_to_cascade_and_providers(self):
        mgr = _empty_manager()
        provider = FakeProvider(name="custom", label="Custom")
        mgr.register_provider(provider)
        assert "custom" in mgr.provider_cascade
        assert "custom" in mgr.providers

    def test_register_unavailable_provider_adds_cascade_not_providers(self):
        mgr = _empty_manager()
        provider = FakeProvider(name="offline", label="Offline", available=False)
        mgr.register_provider(provider)
        assert "offline" in mgr.provider_cascade
        assert "offline" not in mgr.providers

    def test_register_twice_does_not_duplicate_cascade(self):
        mgr = _empty_manager()
        provider = FakeProvider(name="dup", label="Dup")
        mgr.register_provider(provider)
        mgr.register_provider(provider)
        assert mgr.provider_cascade.count("dup") == 1

    def test_register_updates_global_registry(self):
        """Registering a provider also updates the PROVIDERS dict in the providers module."""
        from llming_models.providers import PROVIDERS

        mgr = _empty_manager()
        provider = FakeProvider(name="reg_test", label="RegTest")
        mgr.register_provider(provider)
        assert "reg_test" in PROVIDERS
        # Cleanup
        del PROVIDERS["reg_test"]

    def test_register_replaces_existing_provider(self):
        """Re-registering with a new instance replaces the old one."""
        mgr = _empty_manager()
        p1 = FakeProvider(name="rep", label="Rep v1", models=[
            _make_model_info(provider="rep", name="m_v1"),
        ])
        p2 = FakeProvider(name="rep", label="Rep v2", models=[
            _make_model_info(provider="rep", name="m_v2"),
        ])
        mgr.register_provider(p1)
        mgr.register_provider(p2)
        assert mgr.providers["rep"] is p2
        models = mgr.providers["rep"].get_models()
        assert models[0].name == "m_v2"


# ---------------------------------------------------------------------------
# register_openai_compatible
# ---------------------------------------------------------------------------


class TestRegisterOpenAICompatible:
    """Tests for LLMManager.register_openai_compatible."""

    def test_creates_and_registers_generic_provider(self):
        mgr = _empty_manager()
        model = _make_model_info(provider="deepseek", name="ds_chat", model="deepseek-chat")
        mgr.register_openai_compatible(
            name="deepseek",
            label="DeepSeek",
            api_key="sk-fake",
            base_url="https://api.deepseek.com",
            models=[model],
        )
        assert "deepseek" in mgr.providers
        assert "deepseek" in mgr.provider_cascade

    def test_models_available_after_registration(self):
        mgr = _empty_manager()
        model = _make_model_info(provider="deepseek", name="ds_chat", model="deepseek-chat")
        mgr.register_openai_compatible(
            name="deepseek",
            label="DeepSeek",
            api_key="sk-fake",
            base_url="https://api.deepseek.com",
            models=[model],
        )
        llms = mgr.get_available_llms()
        names = [m.name for m in llms]
        assert "ds_chat" in names

    def test_registered_provider_is_generic_type(self):
        """The registered provider is a GenericOpenAIProvider."""
        from llming_models.providers.generic_openai_provider import GenericOpenAIProvider

        mgr = _empty_manager()
        model = _make_model_info(provider="ollama", name="llama3", model="llama3")
        mgr.register_openai_compatible(
            name="ollama",
            label="Ollama",
            api_key="unused",
            base_url="http://localhost:11434/v1",
            models=[model],
        )
        assert isinstance(mgr.providers["ollama"], GenericOpenAIProvider)

    def test_multiple_models(self):
        """Registering multiple models at once works."""
        mgr = _empty_manager()
        models = [
            _make_model_info(provider="local", name="m_a", model="model-a"),
            _make_model_info(provider="local", name="m_b", model="model-b"),
            _make_model_info(provider="local", name="m_c", model="model-c"),
        ]
        mgr.register_openai_compatible(
            name="local",
            label="Local",
            api_key="key",
            base_url="http://localhost:8000",
            models=models,
        )
        llms = mgr.get_available_llms()
        assert len(llms) == 3


# ---------------------------------------------------------------------------
# get_available_llms
# ---------------------------------------------------------------------------


class TestGetAvailableLlms:
    """Tests for LLMManager.get_available_llms."""

    def test_empty_manager_returns_empty(self):
        mgr = _empty_manager()
        assert mgr.get_available_llms() == []

    def test_single_provider_returns_models(self):
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="a", label="A", models=[
            _make_model_info(provider="a", name="m1"),
            _make_model_info(provider="a", name="m2"),
        ]))
        assert len(mgr.get_available_llms()) == 2

    def test_deduplication_by_name(self):
        """Same model name from two providers: first wins."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="high", label="High", models=[
            _make_model_info(provider="high", name="shared_model"),
        ]))
        mgr.register_provider(FakeProvider(name="low", label="Low", models=[
            _make_model_info(provider="low", name="shared_model"),
        ]))
        llms = mgr.get_available_llms()
        assert len(llms) == 1
        assert llms[0].provider == "high"

    def test_different_names_not_deduplicated(self):
        """Models with distinct names from different providers all appear."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p1", label="P1", models=[
            _make_model_info(provider="p1", name="alpha"),
        ]))
        mgr.register_provider(FakeProvider(name="p2", label="P2", models=[
            _make_model_info(provider="p2", name="beta"),
        ]))
        llms = mgr.get_available_llms()
        assert len(llms) == 2
        names = {m.name for m in llms}
        assert names == {"alpha", "beta"}

    def test_order_follows_registration(self):
        """Models from the first-registered provider come first."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="first", label="First", models=[
            _make_model_info(provider="first", name="m1"),
        ]))
        mgr.register_provider(FakeProvider(name="second", label="Second", models=[
            _make_model_info(provider="second", name="m2"),
        ]))
        llms = mgr.get_available_llms()
        assert llms[0].name == "m1"
        assert llms[1].name == "m2"


# ---------------------------------------------------------------------------
# get_providers_for_model
# ---------------------------------------------------------------------------


class TestGetProvidersForModel:
    """Tests for LLMManager.get_providers_for_model."""

    def _make_manager(self) -> LLMManager:
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="prov_a", label="A", models=[
            _make_model_info(provider="prov_a", name="m1", model="model-v1"),
        ]))
        mgr.register_provider(FakeProvider(name="prov_b", label="B", models=[
            _make_model_info(provider="prov_b", name="m1", model="model-v1"),
            _make_model_info(provider="prov_b", name="m2", model="model-v2"),
        ]))
        return mgr

    def test_plain_name_returns_all_providers(self):
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("m1")
        assert "prov_a" in providers
        assert "prov_b" in providers

    def test_plain_name_preserves_cascade_order(self):
        """Providers returned in the order they were registered (cascade order)."""
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("m1")
        assert providers == ["prov_a", "prov_b"]

    def test_provider_colon_model_format(self):
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("prov_b:m2")
        assert providers == ["prov_b"]

    def test_provider_colon_unknown_model_returns_empty(self):
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("prov_a:nonexistent")
        assert providers == []

    def test_model_not_found_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="not found"):
            mgr.get_providers_for_model("does_not_exist")

    def test_unknown_provider_prefix_returns_empty(self):
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("unknown_prov:m1")
        assert providers == []

    def test_lookup_by_api_model_name(self):
        """Can look up by the 'model' field (API model name), not just 'name'."""
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("model-v2")
        assert "prov_b" in providers

    def test_excluded_model_filtered_out(self):
        """A model excluded by user config is not returned."""
        global_cfg = LLMGlobalConfig(provider_cascade=[])
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            excluded_models=["prov_a:m1"],
        )
        mgr = LLMManager(user_config=user_cfg)
        mgr.register_provider(FakeProvider(name="prov_a", label="A", models=[
            _make_model_info(provider="prov_a", name="m1", model="model-v1"),
        ]))
        mgr.register_provider(FakeProvider(name="prov_b", label="B", models=[
            _make_model_info(provider="prov_b", name="m1", model="model-v1"),
        ]))
        providers = mgr.get_providers_for_model("m1")
        # prov_a:m1 is excluded, only prov_b should be returned
        assert providers == ["prov_b"]

    def test_colon_format_with_first_provider(self):
        """Explicit provider:model format returns only that provider."""
        mgr = self._make_manager()
        providers = mgr.get_providers_for_model("prov_a:m1")
        assert providers == ["prov_a"]


# ---------------------------------------------------------------------------
# get_provider_for_model
# ---------------------------------------------------------------------------


class TestGetProviderForModel:
    """Tests for LLMManager.get_provider_for_model."""

    def test_returns_highest_priority(self):
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="first", label="First", models=[
            _make_model_info(provider="first", name="m1"),
        ]))
        mgr.register_provider(FakeProvider(name="second", label="Second", models=[
            _make_model_info(provider="second", name="m1"),
        ]))
        assert mgr.get_provider_for_model("m1") == "first"

    def test_not_found_raises(self):
        mgr = _empty_manager()
        with pytest.raises(ValueError):
            mgr.get_provider_for_model("nonexistent")

    def test_pinned_provider(self):
        """provider:model format returns the pinned provider even if not first."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="prov_a", label="A", models=[
            _make_model_info(provider="prov_a", name="m1"),
        ]))
        mgr.register_provider(FakeProvider(name="prov_b", label="B", models=[
            _make_model_info(provider="prov_b", name="m1"),
        ]))
        assert mgr.get_provider_for_model("prov_b:m1") == "prov_b"

    def test_pinned_unknown_provider_raises(self):
        """provider:model with unknown provider raises ValueError."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="m1"),
        ]))
        with pytest.raises(ValueError, match="not found"):
            mgr.get_provider_for_model("nonexistent:m1")

    def test_single_provider_returns_it(self):
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="only", label="Only", models=[
            _make_model_info(provider="only", name="solo_model"),
        ]))
        assert mgr.get_provider_for_model("solo_model") == "only"

    def test_lookup_by_api_model_name(self):
        """Can resolve by the actual API model string (model field)."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="my_name", model="gpt-4o-mini"),
        ]))
        assert mgr.get_provider_for_model("gpt-4o-mini") == "p"


# ---------------------------------------------------------------------------
# get_default_model
# ---------------------------------------------------------------------------


class TestGetDefaultModel:
    """Tests for LLMManager.get_default_model."""

    def _make_manager(self) -> LLMManager:
        global_cfg = LLMGlobalConfig(
            provider_cascade=[],
            default_models={"small": ["m1", "m2"], "large": "m3"},
        )
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="m1"),
            _make_model_info(provider="p", name="m3", model="model-v3"),
        ]))
        return mgr

    def test_user_override(self):
        mgr = self._make_manager()
        mgr.user_config.default_models["small"] = "user_model"
        assert mgr.get_default_model("small") == "user_model"

    def test_user_override_takes_priority_over_global(self):
        """Even when the global config has candidates, user override wins."""
        mgr = self._make_manager()
        mgr.user_config.default_models["large"] = "user_large"
        assert mgr.get_default_model("large") == "user_large"

    def test_global_config_first_available(self):
        mgr = self._make_manager()
        # m1 is available, so it should be returned
        assert mgr.get_default_model("small") == "m1"

    def test_unknown_category_returns_none(self):
        mgr = self._make_manager()
        assert mgr.get_default_model("nonexistent_category") is None

    def test_single_string_model_in_global_config(self):
        mgr = self._make_manager()
        result = mgr.get_default_model("large")
        assert result == "m3"

    def test_fallback_when_first_unavailable(self):
        """When first candidate is not available, fall back to second."""
        global_cfg = LLMGlobalConfig(
            provider_cascade=[],
            default_models={"test": ["unavailable_model", "m1"]},
        )
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="m1"),
        ]))
        assert mgr.get_default_model("test") == "m1"

    def test_all_candidates_unavailable_returns_first(self):
        """When no candidate is available, return first as best-effort fallback."""
        global_cfg = LLMGlobalConfig(
            provider_cascade=[],
            default_models={"test": ["gone_a", "gone_b"]},
        )
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        assert mgr.get_default_model("test") == "gone_a"

    def test_empty_candidates_returns_none(self):
        """Category mapped to empty list returns None."""
        global_cfg = LLMGlobalConfig(
            provider_cascade=[],
            default_models={"empty_cat": []},
        )
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        assert mgr.get_default_model("empty_cat") is None


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    """Tests for LLMManager.get_model_info."""

    def _make_manager(self) -> LLMManager:
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(
                provider="p", name="my_model", model="my-model-v1",
                label="My Model", max_input_tokens=32000, max_output_tokens=8192,
            ),
        ]))
        return mgr

    def test_success_by_name(self):
        mgr = self._make_manager()
        info = mgr.get_model_info("my_model")
        assert info.name == "my_model"
        assert info.label == "My Model"

    def test_success_by_api_model(self):
        """Can look up by the actual API model string."""
        mgr = self._make_manager()
        info = mgr.get_model_info("my-model-v1")
        assert info.name == "my_model"

    def test_failure_raises_value_error(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError):
            mgr.get_model_info("nonexistent")

    def test_with_provider_prefix(self):
        mgr = self._make_manager()
        info = mgr.get_model_info("p:my_model")
        assert info.name == "my_model"

    def test_returns_correct_token_limits(self):
        mgr = self._make_manager()
        info = mgr.get_model_info("my_model")
        assert info.max_input_tokens == 32000
        assert info.max_output_tokens == 8192

    def test_returns_correct_provider(self):
        mgr = self._make_manager()
        info = mgr.get_model_info("my_model")
        assert info.provider == "p"

    def test_with_provider_prefix_wrong_provider_raises(self):
        """Asking for a model on a non-existent provider raises."""
        mgr = self._make_manager()
        with pytest.raises(ValueError):
            mgr.get_model_info("nonexistent_prov:my_model")


# ---------------------------------------------------------------------------
# get_config_for_model
# ---------------------------------------------------------------------------


class TestGetConfigForModel:
    """Tests for LLMManager.get_config_for_model."""

    def _make_manager(self) -> LLMManager:
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(
                provider="p", name="m1", model="m1-api",
                api_base="https://custom.api.example.com",
                max_input_tokens=128000, max_output_tokens=16384,
            ),
        ]))
        return mgr

    def test_returns_llm_config(self):
        mgr = self._make_manager()
        config = mgr.get_config_for_model("m1")
        assert isinstance(config, LLMConfig)
        assert config.provider == "p"
        assert config.model == "m1-api"
        assert config.temperature == 0.7

    def test_config_has_correct_token_limits(self):
        mgr = self._make_manager()
        config = mgr.get_config_for_model("m1")
        assert config.max_input_tokens == 128000
        assert config.max_tokens == 16384

    def test_config_has_base_url(self):
        mgr = self._make_manager()
        config = mgr.get_config_for_model("m1")
        assert config.base_url == "https://custom.api.example.com"

    def test_config_for_nonexistent_model_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError):
            mgr.get_config_for_model("ghost_model")

    def test_config_with_provider_prefix(self):
        mgr = self._make_manager()
        config = mgr.get_config_for_model("p:m1")
        assert config.provider == "p"
        assert config.model == "m1-api"

    def test_config_base_url_none_when_not_set(self):
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="q", label="Q", models=[
            _make_model_info(provider="q", name="vanilla", model="vanilla-api", api_base=None),
        ]))
        config = mgr.get_config_for_model("vanilla")
        assert config.base_url is None


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class _FakeProviderForSession(FakeProvider):
    """A FakeProvider subclass that can be registered in the global PROVIDERS dict.

    ChatSession.__init__ calls ``get_provider(config.provider)`` to look up
    the provider class in the global registry.  We register this class under
    the name "p" so that ChatSession can instantiate it.
    """

    # Class-level list of models shared across all instances.  Overwritten
    # by _make_manager before each test group.
    _class_models: list[LLMInfo] = []

    def __init__(self, credentials=None):
        super().__init__(
            name="p",
            label="P",
            available=True,
            models=self.__class__._class_models,
            credentials=credentials,
        )


class TestCreateSession:
    """Tests for LLMManager.create_session."""

    _SESSION_MODELS = [
        _make_model_info(provider="p", name="m1", model="m1-api"),
        _make_model_info(provider="p", name="m_large", model="m-large-api"),
    ]

    def _make_manager(self, **kwargs) -> LLMManager:
        global_cfg = LLMGlobalConfig(
            provider_cascade=[],
            default_models={"small": "m1", "large": ["m_large"]},
        )
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg, **kwargs)
        _FakeProviderForSession._class_models = list(self._SESSION_MODELS)
        mgr.register_provider(_FakeProviderForSession())
        return mgr

    @pytest.fixture(autouse=True)
    def _register_fake_provider(self):
        """Register and unregister the fake 'p' provider in the global registry."""
        _FakeProviderForSession._class_models = list(self._SESSION_MODELS)
        PROVIDERS["p"] = _FakeProviderForSession
        yield
        PROVIDERS.pop("p", None)

    def test_create_with_config(self):
        mgr = self._make_manager()
        config = mgr.get_config_for_model("m1")
        session = mgr.create_session(config=config, system_prompt="Hello")
        assert isinstance(session, ChatSession)
        assert session.system_prompt == "Hello"

    def test_create_with_model_name(self):
        mgr = self._make_manager()
        session = mgr.create_session(model="m1")
        assert session.config.model == "m1-api"

    def test_create_with_category(self):
        mgr = self._make_manager()
        session = mgr.create_session(category="small")
        assert session.config.model == "m1-api"

    def test_create_with_category_large(self):
        mgr = self._make_manager()
        session = mgr.create_session(category="large")
        assert session.config.model == "m-large-api"

    def test_missing_config_and_model_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="Either config or model"):
            mgr.create_session()

    def test_invalid_category_raises(self):
        mgr = self._make_manager()
        with pytest.raises(ValueError, match="No default model"):
            mgr.create_session(category="unknown_cat")

    def test_config_takes_precedence_over_model(self):
        """When both config and model are provided, config wins (model arg ignored)."""
        mgr = self._make_manager()
        config = mgr.get_config_for_model("m1")
        session = mgr.create_session(config=config, model="m_large")
        # config was built for m1 so that is what the session uses
        assert session.config.model == "m1-api"

    def test_category_takes_precedence_over_model(self):
        """When category is specified, it resolves the model from defaults."""
        mgr = self._make_manager()
        session = mgr.create_session(category="small", model="m_large")
        # Category "small" -> "m1" -> "m1-api"
        assert session.config.model == "m1-api"

    def test_system_prompt_set(self):
        mgr = self._make_manager()
        session = mgr.create_session(model="m1", system_prompt="Be helpful.")
        assert session.system_prompt == "Be helpful."

    def test_system_prompt_none_by_default(self):
        mgr = self._make_manager()
        session = mgr.create_session(model="m1")
        assert session.system_prompt is None

    def test_user_id_passed_through(self):
        mgr = self._make_manager()
        session = mgr.create_session(model="m1", user_id="user-123")
        assert session.user_id == "user-123"

    def test_credentials_forwarded_to_session(self):
        """When manager has credentials, create_session forwards them."""
        creds = LLMCredentials(
            openai=ProviderCredentials(api_key="sk-test"),
        )
        mgr = self._make_manager(credentials=creds)
        session = mgr.create_session(model="m1")
        assert isinstance(session, ChatSession)

    def test_budget_manager_default(self):
        """When manager has no budget_manager, session gets None."""
        mgr = self._make_manager()
        session = mgr.create_session(model="m1")
        assert session.budget_manager is None

    def test_budget_manager_passed_through(self):
        """Explicit budget_manager in create_session overrides the default."""
        mgr = self._make_manager()
        mock_budget = MagicMock()
        session = mgr.create_session(model="m1", budget_manager=mock_budget)
        assert session.budget_manager is mock_budget

    def test_create_session_returns_chat_session(self):
        mgr = self._make_manager()
        session = mgr.create_session(model="m1")
        assert isinstance(session, ChatSession)


# ---------------------------------------------------------------------------
# get_cascade_debug_info
# ---------------------------------------------------------------------------


class TestGetCascadeDebugInfo:
    """Tests for LLMManager.get_cascade_debug_info."""

    def test_returns_list_of_dicts(self):
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="m1", model="m1-api"),
        ]))
        info = mgr.get_cascade_debug_info()
        assert isinstance(info, list)
        assert len(info) == 1
        entry = info[0]
        assert "model" in entry
        assert "active_provider" in entry
        assert "providers" in entry
        assert entry["model"] == "m1-api"
        assert entry["active_provider"] == "p"

    def test_empty_manager(self):
        mgr = _empty_manager()
        assert mgr.get_cascade_debug_info() == []

    def test_multiple_providers_single_model(self):
        """Two providers serving the same API model should appear in providers list."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="pri", label="Primary", models=[
            _make_model_info(provider="pri", name="shared", model="shared-api", hosting_icon="pri.svg"),
        ]))
        mgr.register_provider(FakeProvider(name="sec", label="Secondary", models=[
            _make_model_info(provider="sec", name="shared", model="shared-api", hosting_icon="sec.svg"),
        ]))
        info = mgr.get_cascade_debug_info()
        # Both providers serve the same "shared-api" model
        shared_entry = [e for e in info if e["model"] == "shared-api"]
        assert len(shared_entry) == 1
        providers = shared_entry[0]["providers"]
        provider_names = [p["provider"] for p in providers]
        assert "pri" in provider_names
        assert "sec" in provider_names

    def test_active_provider_is_highest_cascade(self):
        """The active_provider should be the cascade winner."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="winner", label="Winner", models=[
            _make_model_info(provider="winner", name="m1", model="m1-api"),
        ]))
        mgr.register_provider(FakeProvider(name="loser", label="Loser", models=[
            _make_model_info(provider="loser", name="m1", model="m1-api"),
        ]))
        info = mgr.get_cascade_debug_info()
        m1_entry = [e for e in info if e["model"] == "m1-api"][0]
        assert m1_entry["active_provider"] == "winner"

    def test_providers_entry_contains_expected_fields(self):
        """Each provider entry has 'provider', 'label', and 'hosting_icon'."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="x", label="X Provider", models=[
            _make_model_info(provider="x", name="mx", model="mx-api", label="MX Label", hosting_icon="icon.svg"),
        ]))
        info = mgr.get_cascade_debug_info()
        prov_entry = info[0]["providers"][0]
        assert prov_entry["provider"] == "x"
        assert prov_entry["label"] == "MX Label"
        assert prov_entry["hosting_icon"] == "icon.svg"

    def test_multiple_distinct_models(self):
        """Each unique model gets its own entry."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="p", label="P", models=[
            _make_model_info(provider="p", name="a", model="api-a"),
            _make_model_info(provider="p", name="b", model="api-b"),
            _make_model_info(provider="p", name="c", model="api-c"),
        ]))
        info = mgr.get_cascade_debug_info()
        assert len(info) == 3
        model_names = {e["model"] for e in info}
        assert model_names == {"api-a", "api-b", "api-c"}


# ---------------------------------------------------------------------------
# Integration-style tests (multiple features together)
# ---------------------------------------------------------------------------


class TestLLMManagerIntegration:
    """Tests combining multiple manager features."""

    def test_register_then_get_config(self):
        """Full workflow: register provider -> get config for session creation."""
        mgr = _empty_manager()
        model = _make_model_info(provider="int_p", name="int_model", model="int-api")
        mgr.register_openai_compatible(
            name="int_p",
            label="Int",
            api_key="test-key",
            base_url="http://localhost:9999",
            models=[model],
        )
        config = mgr.get_config_for_model("int_model")
        assert config.model == "int-api"
        assert config.provider == "int_p"

    def test_openai_compatible_info_and_config(self):
        """Register OpenAI-compatible provider, verify info and config."""
        mgr = _empty_manager()
        model = _make_model_info(provider="local", name="local_m", model="local-api")
        mgr.register_openai_compatible(
            name="local",
            label="Local",
            api_key="k",
            base_url="http://localhost:8000",
            models=[model],
        )
        info = mgr.get_model_info("local_m")
        assert info.model == "local-api"
        config = mgr.get_config_for_model("local_m")
        assert config.model == "local-api"

    def test_cascade_priority_end_to_end(self):
        """Register two providers with same model; first one wins in get_model_info."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="primary", label="Primary", models=[
            _make_model_info(provider="primary", name="shared", model="shared-api"),
        ]))
        mgr.register_provider(FakeProvider(name="fallback", label="Fallback", models=[
            _make_model_info(provider="fallback", name="shared", model="shared-api"),
        ]))
        info = mgr.get_model_info("shared")
        assert info.provider == "primary"
        # get_config_for_model should also resolve to primary
        config = mgr.get_config_for_model("shared")
        assert config.provider == "primary"

    def test_pinned_provider_bypasses_cascade(self):
        """Using provider:model pins to a specific provider."""
        mgr = _empty_manager()
        mgr.register_provider(FakeProvider(name="first", label="First", models=[
            _make_model_info(provider="first", name="m", model="m-api"),
        ]))
        mgr.register_provider(FakeProvider(name="second", label="Second", models=[
            _make_model_info(provider="second", name="m", model="m-api"),
        ]))
        info = mgr.get_model_info("second:m")
        assert info.provider == "second"

    def test_excluded_model_not_in_providers_for_model(self):
        """Excluded model is filtered from get_providers_for_model."""
        global_cfg = LLMGlobalConfig(provider_cascade=[])
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            excluded_models=["ex_prov:blocked"],
        )
        mgr = LLMManager(user_config=user_cfg)
        mgr.register_provider(FakeProvider(name="ex_prov", label="Ex", models=[
            _make_model_info(provider="ex_prov", name="blocked"),
            _make_model_info(provider="ex_prov", name="allowed"),
        ]))
        # get_providers_for_model should not return ex_prov for the blocked model
        with pytest.raises(ValueError, match="not found"):
            mgr.get_providers_for_model("blocked")
        # But allowed model works fine
        assert mgr.get_providers_for_model("allowed") == ["ex_prov"]

    def test_model_defaults_version_accessible(self):
        """model_defaults_version from global config is accessible."""
        global_cfg = LLMGlobalConfig(provider_cascade=[], model_defaults_version=5)
        user_cfg = LLMUserConfig(global_config=global_cfg)
        mgr = LLMManager(user_config=user_cfg)
        assert mgr.user_config.global_config.model_defaults_version == 5

    def test_full_workflow_openai_compatible(self):
        """Complete workflow: register, list, info, config, cascade debug."""
        mgr = _empty_manager()
        models = [
            _make_model_info(provider="custom", name="fast", model="fast-v1", label="Fast"),
            _make_model_info(provider="custom", name="smart", model="smart-v1", label="Smart"),
        ]
        mgr.register_openai_compatible(
            name="custom",
            label="Custom AI",
            api_key="key-123",
            base_url="https://custom.ai/v1",
            models=models,
        )
        # List
        llms = mgr.get_available_llms()
        assert len(llms) == 2
        # Info
        info = mgr.get_model_info("fast")
        assert info.label == "Fast"
        # Config
        config = mgr.get_config_for_model("smart")
        assert config.model == "smart-v1"
        assert config.provider == "custom"
        # Cascade debug
        debug = mgr.get_cascade_debug_info()
        api_models = {e["model"] for e in debug}
        assert "fast-v1" in api_models
        assert "smart-v1" in api_models
