"""Tests for llming_models.config — LLMBaseConfig, LLMGlobalConfig, LLMUserConfig."""

from llming_models.config import LLMBaseConfig, LLMGlobalConfig, LLMUserConfig


class TestLLMBaseConfig:
    def test_default_includes_all(self):
        config = LLMBaseConfig()
        assert config.is_model_supported("anything") is True

    def test_include_filter(self):
        config = LLMBaseConfig(included_models=["anthropic:*"])
        assert config.is_model_supported("anthropic:claude") is True
        assert config.is_model_supported("openai:gpt-4o") is False

    def test_exclude_filter(self):
        config = LLMBaseConfig(excluded_models=["*:old_model"])
        assert config.is_model_supported("anthropic:claude") is True
        assert config.is_model_supported("anthropic:old_model") is False

    def test_include_and_exclude(self):
        config = LLMBaseConfig(
            included_models=["anthropic:*"],
            excluded_models=["anthropic:old"],
        )
        assert config.is_model_supported("anthropic:claude") is True
        assert config.is_model_supported("anthropic:old") is False
        assert config.is_model_supported("openai:gpt") is False


class TestLLMGlobalConfig:
    def test_default_models(self):
        config = LLMGlobalConfig()
        assert config.get_default_model("small") is not None
        assert config.get_default_model("nonexistent") is None

    def test_candidates_list(self):
        config = LLMGlobalConfig(default_models={"small": ["a", "b", "c"]})
        assert config.get_default_model_candidates("small") == ["a", "b", "c"]
        assert config.get_default_model("small") == "a"

    def test_candidates_single(self):
        config = LLMGlobalConfig(default_models={"small": "single"})
        assert config.get_default_model_candidates("small") == ["single"]
        assert config.get_default_model("small") == "single"

    def test_empty_category(self):
        config = LLMGlobalConfig(default_models={})
        assert config.get_default_model_candidates("small") == []
        assert config.get_default_model("small") is None

    def test_provider_cascade(self):
        config = LLMGlobalConfig()
        assert "openai" in config.provider_cascade
        assert "anthropic" in config.provider_cascade


class TestLLMUserConfig:
    def test_user_overrides_global(self):
        global_config = LLMGlobalConfig(default_models={"small": "global_model"})
        user_config = LLMUserConfig(
            global_config=global_config,
            default_models={"small": "user_model"},
        )
        assert user_config.get_default_model("small") == "user_model"

    def test_falls_back_to_global(self):
        global_config = LLMGlobalConfig(default_models={"small": "global_model"})
        user_config = LLMUserConfig(global_config=global_config)
        assert user_config.get_default_model("small") == "global_model"

    def test_user_filter_combined_with_global(self):
        global_config = LLMGlobalConfig(included_models=["anthropic:*", "openai:*"])
        user_config = LLMUserConfig(
            global_config=global_config,
            excluded_models=["openai:*"],
        )
        assert user_config.is_model_supported("anthropic:claude") is True
        assert user_config.is_model_supported("openai:gpt") is False

    def test_global_exclude_overrules_user_include(self):
        global_config = LLMGlobalConfig(excluded_models=["*:blocked"])
        user_config = LLMUserConfig(global_config=global_config)
        assert user_config.is_model_supported("anthropic:blocked") is False
