"""Comprehensive tests for llming-models core types.

Covers: model_info, model_categories, config, credentials,
llm_base_models, messages, budget_types, and image_utils.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# model_info.py
# ---------------------------------------------------------------------------

from llming_models.model_info import ModelSize, ReasoningEffort, LLMInfo


class TestModelSize:
    """ModelSize IntEnum values and ordering."""

    def test_has_expected_members(self) -> None:
        assert ModelSize.VERY_SMALL.value == 1
        assert ModelSize.SMALL.value == 2
        assert ModelSize.MEDIUM.value == 3
        assert ModelSize.LARGE.value == 4
        assert ModelSize.VERY_LARGE.value == 5

    def test_ordering(self) -> None:
        assert ModelSize.SMALL < ModelSize.MEDIUM < ModelSize.LARGE

    def test_int_comparison(self) -> None:
        # IntEnum supports direct int comparison
        assert ModelSize.MEDIUM == 3


class TestReasoningEffort:
    """ReasoningEffort str enum values."""

    def test_has_expected_members(self) -> None:
        assert ReasoningEffort.NONE == "none"
        assert ReasoningEffort.MINIMAL == "minimal"
        assert ReasoningEffort.LOW == "low"
        assert ReasoningEffort.MEDIUM == "medium"
        assert ReasoningEffort.HIGH == "high"

    def test_is_str_subclass(self) -> None:
        # str enum values can be used as strings
        assert isinstance(ReasoningEffort.HIGH, str)
        assert ReasoningEffort.HIGH.upper() == "HIGH"


class TestLLMInfo:
    """LLMInfo dataclass construction and defaults."""

    def test_minimal_construction(self) -> None:
        info = LLMInfo(
            provider="anthropic",
            name="claude_sonnet",
            label="Claude Sonnet",
            model="claude-sonnet-4-20250514",
            description="A balanced model",
            input_token_price=3.0,
        )
        assert info.provider == "anthropic"
        assert info.name == "claude_sonnet"
        assert info.label == "Claude Sonnet"
        assert info.model == "claude-sonnet-4-20250514"
        assert info.description == "A balanced model"
        assert info.input_token_price == 3.0

    def test_defaults(self) -> None:
        info = LLMInfo(
            provider="test",
            name="test_model",
            label="Test",
            model="test-v1",
            description="Test model",
            input_token_price=1.0,
        )
        assert info.cached_input_token_price == 0.0
        assert info.output_token_price == 0.0
        assert info.size == ModelSize.MEDIUM
        assert info.max_input_tokens == 64000
        assert info.max_output_tokens == 4096
        assert info.api_base is None
        assert info.supports_system_prompt is True
        assert info.tokenizer_name is None
        assert info.model_icon is None
        assert info.company_icon is None
        assert info.hosting_icon is None
        assert info.popularity == 0
        assert info.reasoning is False
        assert info.reasoning_effort is None
        assert info.default_reasoning_effort is None
        assert info.enforced_temperature is None
        assert info.supports_image_input is False
        assert info.speed == 5
        assert info.quality == 5
        assert info.best_use == "General"
        assert info.highlights == []
        assert info.default_tools == []
        assert info.native_tools == {}

    def test_full_construction(self) -> None:
        info = LLMInfo(
            provider="openai",
            name="gpt5",
            label="GPT-5",
            model="gpt-5",
            description="Latest OpenAI model",
            input_token_price=5.0,
            cached_input_token_price=2.5,
            output_token_price=15.0,
            size=ModelSize.LARGE,
            max_input_tokens=128000,
            max_output_tokens=8192,
            api_base="https://api.openai.com",
            supports_system_prompt=True,
            tokenizer_name="cl100k_base",
            model_icon="/icons/gpt5.png",
            company_icon="/icons/openai.png",
            hosting_icon="/icons/azure.png",
            popularity=10,
            reasoning=True,
            reasoning_effort=ReasoningEffort.HIGH,
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            enforced_temperature=0.7,
            supports_image_input=True,
            speed=8,
            quality=9,
            best_use="Complex reasoning",
            highlights=["fast", "accurate"],
            default_tools=["web_search"],
            native_tools={"code_interpreter": {"enabled": True}},
        )
        assert info.size == ModelSize.LARGE
        assert info.reasoning is True
        assert info.reasoning_effort == ReasoningEffort.HIGH
        assert info.highlights == ["fast", "accurate"]
        assert info.native_tools == {"code_interpreter": {"enabled": True}}


# ---------------------------------------------------------------------------
# model_categories.py
# ---------------------------------------------------------------------------

from llming_models.model_categories import ModelCategories


class TestModelCategories:
    """ModelCategories class constants."""

    def test_category_values(self) -> None:
        assert ModelCategories.SMALL == "small"
        assert ModelCategories.MEDIUM == "medium"
        assert ModelCategories.LARGE == "large"
        assert ModelCategories.REASONING_SMALL == "reasoning_small"
        assert ModelCategories.REASONING_MEDIUM == "reasoning_medium"
        assert ModelCategories.REASONING_LARGE == "reasoning_large"

    def test_model_categories_is_union_of_all(self) -> None:
        expected = {"small", "medium", "large", "reasoning_small", "reasoning_medium", "reasoning_large"}
        assert ModelCategories.MODEL_CATEGORIES == expected

    def test_all_individual_categories_in_set(self) -> None:
        for cat in [
            ModelCategories.SMALL,
            ModelCategories.MEDIUM,
            ModelCategories.LARGE,
            ModelCategories.REASONING_SMALL,
            ModelCategories.REASONING_MEDIUM,
            ModelCategories.REASONING_LARGE,
        ]:
            assert cat in ModelCategories.MODEL_CATEGORIES


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------

from llming_models.config import LLMBaseConfig, LLMGlobalConfig, LLMUserConfig


class TestLLMBaseConfig:
    """LLMBaseConfig include/exclude model filtering."""

    def test_default_wildcard_includes_any_model(self) -> None:
        cfg = LLMBaseConfig()
        assert cfg.is_model_supported("anthropic:claude_3_opus") is True
        assert cfg.is_model_supported("openai:gpt-5") is True

    def test_exclusion_overrides_wildcard(self) -> None:
        cfg = LLMBaseConfig(excluded_models=["anthropic:*"])
        assert cfg.is_model_supported("anthropic:claude_3_opus") is False
        assert cfg.is_model_supported("openai:gpt-5") is True

    def test_specific_includes_only_match(self) -> None:
        cfg = LLMBaseConfig(included_models=["anthropic:claude_3_opus"])
        assert cfg.is_model_supported("anthropic:claude_3_opus") is True
        assert cfg.is_model_supported("openai:gpt-5") is False

    def test_specific_includes_with_exclusion(self) -> None:
        cfg = LLMBaseConfig(
            included_models=["anthropic:*"],
            excluded_models=["anthropic:claude_3_opus"],
        )
        assert cfg.is_model_supported("anthropic:claude_3_opus") is False
        assert cfg.is_model_supported("anthropic:claude_3_haiku") is True

    def test_empty_includes_blocks_all(self) -> None:
        cfg = LLMBaseConfig(included_models=[])
        assert cfg.is_model_supported("anything") is False

    def test_no_exclusions_includes_all_matched(self) -> None:
        cfg = LLMBaseConfig(included_models=["*"], excluded_models=[])
        assert cfg.is_model_supported("any_model") is True

    def test_multiple_include_patterns(self) -> None:
        cfg = LLMBaseConfig(included_models=["anthropic:*", "openai:*"])
        assert cfg.is_model_supported("anthropic:claude") is True
        assert cfg.is_model_supported("openai:gpt") is True
        assert cfg.is_model_supported("google:gemini") is False

    def test_multiple_exclude_patterns(self) -> None:
        cfg = LLMBaseConfig(excluded_models=["*opus*", "*large*"])
        assert cfg.is_model_supported("anthropic:claude_3_opus") is False
        assert cfg.is_model_supported("openai:gpt-5-large") is False
        assert cfg.is_model_supported("openai:gpt-5-mini") is True


class TestLLMGlobalConfig:
    """LLMGlobalConfig default model selection."""

    def test_get_default_model_candidates_known_category(self) -> None:
        cfg = LLMGlobalConfig()
        candidates = cfg.get_default_model_candidates("small")
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_get_default_model_candidates_nonexistent(self) -> None:
        cfg = LLMGlobalConfig()
        assert cfg.get_default_model_candidates("nonexistent") == []

    def test_get_default_model_known_category(self) -> None:
        cfg = LLMGlobalConfig()
        model = cfg.get_default_model("small")
        assert model is not None
        assert isinstance(model, str)

    def test_get_default_model_nonexistent(self) -> None:
        cfg = LLMGlobalConfig()
        assert cfg.get_default_model("nonexistent") is None

    def test_get_default_model_returns_first_candidate(self) -> None:
        cfg = LLMGlobalConfig()
        candidates = cfg.get_default_model_candidates("small")
        first = cfg.get_default_model("small")
        assert first == candidates[0]

    def test_single_string_value_becomes_list(self) -> None:
        cfg = LLMGlobalConfig(default_models={"custom": "single_model"})
        candidates = cfg.get_default_model_candidates("custom")
        assert candidates == ["single_model"]

    def test_inherits_base_config_filtering(self) -> None:
        cfg = LLMGlobalConfig(excluded_models=["anthropic:*"])
        assert cfg.is_model_supported("anthropic:claude") is False
        assert cfg.is_model_supported("openai:gpt") is True

    def test_provider_cascade_default(self) -> None:
        cfg = LLMGlobalConfig()
        assert isinstance(cfg.provider_cascade, list)
        assert len(cfg.provider_cascade) > 0
        assert "anthropic" in cfg.provider_cascade

    def test_all_standard_categories_have_defaults(self) -> None:
        cfg = LLMGlobalConfig()
        for cat in ["small", "medium", "large", "reasoning_small", "reasoning_medium", "reasoning_large"]:
            candidates = cfg.get_default_model_candidates(cat)
            assert len(candidates) > 0, f"Category '{cat}' should have default candidates"


class TestLLMUserConfig:
    """LLMUserConfig with global delegation and user overrides."""

    def test_delegates_to_global_config(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(global_config=global_cfg)
        assert user_cfg.is_model_supported("anthropic:claude_3_opus") is True

    def test_global_exclusion_respected(self) -> None:
        global_cfg = LLMGlobalConfig(excluded_models=["anthropic:*"])
        user_cfg = LLMUserConfig(global_config=global_cfg)
        assert user_cfg.is_model_supported("anthropic:claude") is False

    def test_user_exclusion_additional_filter(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            excluded_models=["openai:*"],
        )
        assert user_cfg.is_model_supported("openai:gpt-5") is False
        assert user_cfg.is_model_supported("anthropic:claude") is True

    def test_both_global_and_user_exclusions_combined(self) -> None:
        global_cfg = LLMGlobalConfig(excluded_models=["anthropic:*"])
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            excluded_models=["openai:*"],
        )
        assert user_cfg.is_model_supported("anthropic:claude") is False
        assert user_cfg.is_model_supported("openai:gpt") is False
        assert user_cfg.is_model_supported("google:gemini") is True

    def test_user_includes_narrower_than_global(self) -> None:
        global_cfg = LLMGlobalConfig()  # includes ["*"]
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            included_models=["anthropic:*"],
        )
        assert user_cfg.is_model_supported("anthropic:claude") is True
        assert user_cfg.is_model_supported("openai:gpt") is False

    def test_get_default_model_user_override(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            default_models={"small": "my_custom_model"},
        )
        assert user_cfg.get_default_model("small") == "my_custom_model"

    def test_get_default_model_falls_through_to_global(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(global_config=global_cfg)
        expected = global_cfg.get_default_model("small")
        assert user_cfg.get_default_model("small") == expected

    def test_get_default_model_nonexistent_category(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(global_config=global_cfg)
        assert user_cfg.get_default_model("nonexistent") is None

    def test_user_override_only_for_specific_category(self) -> None:
        global_cfg = LLMGlobalConfig()
        user_cfg = LLMUserConfig(
            global_config=global_cfg,
            default_models={"large": "user_large_model"},
        )
        # "large" is overridden
        assert user_cfg.get_default_model("large") == "user_large_model"
        # "small" falls through to global
        assert user_cfg.get_default_model("small") == global_cfg.get_default_model("small")


# ---------------------------------------------------------------------------
# credentials.py
# ---------------------------------------------------------------------------

from llming_models.credentials import ProviderCredentials, LLMCredentials


class TestProviderCredentials:
    """ProviderCredentials construction."""

    def test_minimal(self) -> None:
        creds = ProviderCredentials(api_key="test-key")
        assert creds.api_key.get_secret_value() == "test-key"
        assert creds.base_url is None
        assert creds.api_version is None
        assert creds.organization is None

    def test_full(self) -> None:
        creds = ProviderCredentials(
            api_key="sk-123",
            base_url="https://api.example.com",
            api_version="2024-01-01",
            organization="org-abc",
        )
        assert creds.api_key.get_secret_value() == "sk-123"
        assert creds.base_url == "https://api.example.com"
        assert creds.api_version == "2024-01-01"
        assert creds.organization == "org-abc"


class TestLLMCredentials:
    """LLMCredentials for_provider lookup."""

    def test_all_fields_none_by_default(self) -> None:
        creds = LLMCredentials()
        assert creds.anthropic is None
        assert creds.openai is None
        assert creds.azure_openai is None
        assert creds.azure_anthropic is None
        assert creds.google is None
        assert creds.mistral is None
        assert creds.together is None

    def test_for_provider_anthropic(self) -> None:
        provider_creds = ProviderCredentials(api_key="x")
        creds = LLMCredentials(anthropic=provider_creds)
        result = creds.for_provider("anthropic")
        assert result is not None
        assert result.api_key.get_secret_value() == "x"

    def test_for_provider_openai_none(self) -> None:
        creds = LLMCredentials()
        assert creds.for_provider("openai") is None

    def test_for_provider_nonexistent(self) -> None:
        creds = LLMCredentials()
        assert creds.for_provider("nonexistent") is None

    def test_for_provider_all_known_providers(self) -> None:
        """Each known provider field can be set and retrieved."""
        for name in ["anthropic", "openai", "azure_openai", "azure_anthropic", "google", "mistral", "together"]:
            creds = LLMCredentials(**{name: ProviderCredentials(api_key=f"key-{name}")})
            result = creds.for_provider(name)
            assert result is not None
            assert result.api_key.get_secret_value() == f"key-{name}"

    def test_multiple_providers(self) -> None:
        creds = LLMCredentials(
            anthropic=ProviderCredentials(api_key="anthropic-key"),
            openai=ProviderCredentials(api_key="openai-key"),
        )
        assert creds.for_provider("anthropic").api_key.get_secret_value() == "anthropic-key"
        assert creds.for_provider("openai").api_key.get_secret_value() == "openai-key"
        assert creds.for_provider("google") is None


# ---------------------------------------------------------------------------
# llm_base_models.py
# ---------------------------------------------------------------------------

from llming_models.llm_base_models import Role, ChatMessage, ChatHistory


class TestRole:
    """Role enum values."""

    def test_standard_roles(self) -> None:
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"

    def test_function_roles(self) -> None:
        assert Role.FUNCTION == "function"
        assert Role.FUNCTION_PENDING == "function_pending"

    def test_is_str_subclass(self) -> None:
        assert isinstance(Role.USER, str)


class TestChatMessage:
    """ChatMessage construction and defaults."""

    def test_construction_with_defaults(self) -> None:
        msg = ChatMessage(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        # Auto-generated UUID
        assert isinstance(msg.id, uuid.UUID)
        # Auto-generated timestamp
        assert isinstance(msg.timestamp, datetime)
        assert msg.timestamp.tzinfo is not None

    def test_auto_uuid_is_unique(self) -> None:
        msg1 = ChatMessage(role=Role.USER, content="a")
        msg2 = ChatMessage(role=Role.USER, content="b")
        assert msg1.id != msg2.id

    def test_auto_timestamp(self) -> None:
        before = datetime.now(timezone.utc)
        msg = ChatMessage(role=Role.USER, content="test")
        after = datetime.now(timezone.utc)
        assert before <= msg.timestamp <= after

    def test_optional_fields_default_none(self) -> None:
        msg = ChatMessage(role=Role.USER, content="test")
        assert msg.name is None
        assert msg.function_call is None
        assert msg.images is None

    def test_images_field(self) -> None:
        msg = ChatMessage(role=Role.USER, content="look", images=["base64data1", "base64data2"])
        assert msg.images == ["base64data1", "base64data2"]

    def test_images_stale_flag(self) -> None:
        msg = ChatMessage(role=Role.USER, content="old", images_stale=True)
        assert msg.images_stale is True

    def test_images_stale_default_false(self) -> None:
        msg = ChatMessage(role=Role.USER, content="new")
        assert msg.images_stale is False

    def test_content_stale_flag(self) -> None:
        msg = ChatMessage(role=Role.USER, content="summarized", content_stale=True)
        assert msg.content_stale is True

    def test_content_stale_default_false(self) -> None:
        msg = ChatMessage(role=Role.USER, content="fresh")
        assert msg.content_stale is False

    def test_function_call_field(self) -> None:
        msg = ChatMessage(
            role=Role.FUNCTION,
            content="",
            name="my_tool",
            function_call={"name": "my_tool", "arguments": "{}"},
        )
        assert msg.function_call == {"name": "my_tool", "arguments": "{}"}
        assert msg.name == "my_tool"

    def test_explicit_id_and_timestamp(self) -> None:
        fixed_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        fixed_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        msg = ChatMessage(role=Role.USER, content="test", id=fixed_id, timestamp=fixed_time)
        assert msg.id == fixed_id
        assert msg.timestamp == fixed_time


class TestChatHistory:
    """ChatHistory message management."""

    def test_empty_by_default(self) -> None:
        history = ChatHistory()
        assert history.get_messages() == []

    def test_add_message(self) -> None:
        history = ChatHistory()
        msg = ChatMessage(role=Role.USER, content="Hello")
        history.add_message(msg)
        assert len(history.get_messages()) == 1
        assert history.get_messages()[0] is msg

    def test_add_multiple_messages(self) -> None:
        history = ChatHistory()
        msg1 = ChatMessage(role=Role.USER, content="Hello")
        msg2 = ChatMessage(role=Role.ASSISTANT, content="Hi there")
        history.add_message(msg1)
        history.add_message(msg2)
        messages = history.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"

    def test_get_last_message(self) -> None:
        history = ChatHistory()
        msg1 = ChatMessage(role=Role.USER, content="first")
        msg2 = ChatMessage(role=Role.ASSISTANT, content="second")
        history.add_message(msg1)
        history.add_message(msg2)
        assert history.get_last_message() is msg2

    def test_get_last_message_empty(self) -> None:
        history = ChatHistory()
        assert history.get_last_message() is None

    def test_clear(self) -> None:
        history = ChatHistory()
        history.add_message(ChatMessage(role=Role.USER, content="msg"))
        history.add_message(ChatMessage(role=Role.ASSISTANT, content="reply"))
        assert len(history.get_messages()) == 2
        history.clear()
        assert history.get_messages() == []
        assert history.get_last_message() is None

    def test_get_messages_returns_same_list(self) -> None:
        history = ChatHistory()
        msg = ChatMessage(role=Role.USER, content="test")
        history.add_message(msg)
        # Returns the internal list (same reference)
        msgs = history.get_messages()
        assert msgs is history.messages


# ---------------------------------------------------------------------------
# messages.py
# ---------------------------------------------------------------------------

from llming_models.messages import LlmSystemMessage, LlmHumanMessage, LlmAIMessage, LlmMessageChunk
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus


class TestLlmSystemMessage:
    """LlmSystemMessage role defaults."""

    def test_role_is_system(self) -> None:
        msg = LlmSystemMessage(content="You are helpful.")
        assert msg.role == Role.SYSTEM

    def test_content(self) -> None:
        msg = LlmSystemMessage(content="System prompt here")
        assert msg.content == "System prompt here"

    def test_response_metadata_default(self) -> None:
        msg = LlmSystemMessage(content="test")
        assert msg.response_metadata == {}


class TestLlmHumanMessage:
    """LlmHumanMessage role and images."""

    def test_role_is_user(self) -> None:
        msg = LlmHumanMessage(content="Hello")
        assert msg.role == Role.USER

    def test_content(self) -> None:
        msg = LlmHumanMessage(content="What is this?")
        assert msg.content == "What is this?"

    def test_images_none_by_default(self) -> None:
        msg = LlmHumanMessage(content="test")
        assert msg.images is None

    def test_images_with_data(self) -> None:
        msg = LlmHumanMessage(content="describe this", images=["base64data"])
        assert msg.images == ["base64data"]

    def test_multiple_images(self) -> None:
        msg = LlmHumanMessage(content="compare", images=["img1", "img2", "img3"])
        assert len(msg.images) == 3


class TestLlmAIMessage:
    """LlmAIMessage role and images."""

    def test_role_is_assistant(self) -> None:
        msg = LlmAIMessage(content="Sure, I can help.")
        assert msg.role == Role.ASSISTANT

    def test_content(self) -> None:
        msg = LlmAIMessage(content="Here is your answer")
        assert msg.content == "Here is your answer"

    def test_images_none_by_default(self) -> None:
        msg = LlmAIMessage(content="test")
        assert msg.images is None

    def test_images_field(self) -> None:
        msg = LlmAIMessage(content="generated image", images=["data:image/png;base64,abc"])
        assert msg.images == ["data:image/png;base64,abc"]


class TestLlmMessageChunk:
    """LlmMessageChunk for streaming responses."""

    def test_basic_text_chunk(self) -> None:
        chunk = LlmMessageChunk(
            role=Role.ASSISTANT,
            content="Hello",
            index=0,
            is_final=False,
        )
        assert chunk.content == "Hello"
        assert chunk.index == 0
        assert chunk.is_final is False
        assert chunk.tool_call is None
        assert chunk.images is None

    def test_final_chunk(self) -> None:
        chunk = LlmMessageChunk(
            role=Role.ASSISTANT,
            content="Done.",
            index=5,
            is_final=True,
        )
        assert chunk.is_final is True
        assert chunk.index == 5

    def test_chunk_with_tool_call(self) -> None:
        tool_info = ToolCallInfo(
            name="web_search",
            call_id="call_123",
            status=ToolCallStatus.COMPLETED,
            arguments={"query": "test"},
            result="found it",
        )
        chunk = LlmMessageChunk(
            role=Role.ASSISTANT,
            content="",
            index=1,
            is_final=False,
            tool_call=tool_info,
        )
        assert chunk.tool_call is not None
        assert chunk.tool_call.name == "web_search"
        assert chunk.tool_call.call_id == "call_123"
        assert chunk.tool_call.status == ToolCallStatus.COMPLETED

    def test_chunk_with_images(self) -> None:
        chunk = LlmMessageChunk(
            role=Role.ASSISTANT,
            content="",
            index=0,
            is_final=True,
            images=["data:image/png;base64,abc123"],
        )
        assert chunk.images == ["data:image/png;base64,abc123"]

    def test_response_metadata_default(self) -> None:
        chunk = LlmMessageChunk(
            role=Role.ASSISTANT,
            content="x",
            index=0,
            is_final=True,
        )
        assert chunk.response_metadata == {}


# ---------------------------------------------------------------------------
# budget/budget_types.py
# ---------------------------------------------------------------------------

from llming_models.budget.budget_types import (
    BudgetScope,
    LimitPeriod,
    InsufficientBudgetError,
    TokenUsage,
    BudgetInfo,
)
from llming_models.budget.time_intervals import TimeInterval


class TestBudgetScope:
    """BudgetScope enum values."""

    def test_global_value(self) -> None:
        assert BudgetScope.GLOBAL == "global"

    def test_per_user_value(self) -> None:
        assert BudgetScope.PER_USER == "per_user"

    def test_is_str_enum(self) -> None:
        assert isinstance(BudgetScope.GLOBAL, str)


class TestLimitPeriod:
    """LimitPeriod is an alias for TimeInterval."""

    def test_is_time_interval(self) -> None:
        assert LimitPeriod is TimeInterval

    def test_has_total(self) -> None:
        assert LimitPeriod.TOTAL.value == "total"

    def test_has_daily(self) -> None:
        assert LimitPeriod.DAILY.value == "daily"

    def test_has_monthly(self) -> None:
        assert LimitPeriod.MONTHLY.value == "monthly"

    def test_has_yearly(self) -> None:
        assert LimitPeriod.YEARLY.value == "yearly"

    def test_has_hourly(self) -> None:
        assert LimitPeriod.HOURLY.value == "hourly"

    def test_has_minutes(self) -> None:
        assert LimitPeriod.MINUTES.value == "minutes"

    def test_has_seconds(self) -> None:
        assert LimitPeriod.SECONDS.value == "seconds"


class TestInsufficientBudgetError:
    """InsufficientBudgetError stores limit_name and message."""

    def test_stores_limit_name(self) -> None:
        err = InsufficientBudgetError("Not enough budget", limit_name="daily_limit")
        assert err.limit_name == "daily_limit"

    def test_stores_message(self) -> None:
        err = InsufficientBudgetError("Budget exceeded", limit_name="test")
        assert str(err) == "Budget exceeded"

    def test_is_exception(self) -> None:
        err = InsufficientBudgetError("fail", limit_name="x")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(InsufficientBudgetError) as exc_info:
            raise InsufficientBudgetError("over limit", limit_name="monthly")
        assert exc_info.value.limit_name == "monthly"
        assert "over limit" in str(exc_info.value)


class TestTokenUsage:
    """TokenUsage field storage and computed properties."""

    def test_stores_fields(self) -> None:
        usage = TokenUsage(10, 20, 0.5, 1.0)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert usage.input_cost == 0.5
        assert usage.output_cost == 1.0

    def test_total_tokens(self) -> None:
        usage = TokenUsage(10, 20, 0.5, 1.0)
        assert usage.total_tokens == 30

    def test_total_cost(self) -> None:
        usage = TokenUsage(10, 20, 0.5, 1.0)
        assert usage.total_cost == 1.5

    def test_zero_values(self) -> None:
        usage = TokenUsage(0, 0, 0.0, 0.0)
        assert usage.total_tokens == 0
        assert usage.total_cost == 0.0

    def test_large_values(self) -> None:
        usage = TokenUsage(1_000_000, 500_000, 10.0, 15.0)
        assert usage.total_tokens == 1_500_000
        assert usage.total_cost == 25.0


class TestBudgetInfo:
    """BudgetInfo TypedDict accepts expected keys."""

    def test_accepts_available_and_reserved(self) -> None:
        info: BudgetInfo = {"available": 10.5, "reserved": 2.3}
        assert info["available"] == 10.5
        assert info["reserved"] == 2.3

    def test_partial_keys_allowed(self) -> None:
        # total=False means all keys are optional
        info: BudgetInfo = {"available": 5.0}
        assert info["available"] == 5.0

    def test_empty_allowed(self) -> None:
        info: BudgetInfo = {}
        assert len(info) == 0


# ---------------------------------------------------------------------------
# utils/image_utils.py
# ---------------------------------------------------------------------------

from llming_models.utils.image_utils import is_likely_image_data, sniff_image_mime


class TestIsLikelyImageData:
    """is_likely_image_data detection logic."""

    def test_short_string_returns_false(self) -> None:
        assert is_likely_image_data("short") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_likely_image_data("") is False

    def test_non_string_returns_false(self) -> None:
        # The function checks isinstance(value, str)
        assert is_likely_image_data(12345) is False  # type: ignore[arg-type]

    def test_data_uri_png_long_enough(self) -> None:
        data = "data:image/png;base64," + "A" * 1000
        assert is_likely_image_data(data) is True

    def test_data_uri_jpeg_long_enough(self) -> None:
        data = "data:image/jpeg;base64," + "A" * 1000
        assert is_likely_image_data(data) is True

    def test_data_uri_too_short(self) -> None:
        data = "data:image/png;base64,abc"
        assert is_likely_image_data(data) is False

    def test_raw_png_base64_long_enough(self) -> None:
        data = "iVBOR" + "A" * 1000
        assert is_likely_image_data(data) is True

    def test_raw_png_base64_too_short(self) -> None:
        data = "iVBOR" + "A" * 10
        assert is_likely_image_data(data) is False

    def test_raw_jpeg_base64_long_enough(self) -> None:
        data = "/9j/" + "A" * 1000
        assert is_likely_image_data(data) is True

    def test_raw_jpeg_base64_too_short(self) -> None:
        data = "/9j/" + "A" * 10
        assert is_likely_image_data(data) is False

    def test_random_long_string_returns_false(self) -> None:
        data = "AAAA" * 500
        assert is_likely_image_data(data) is False

    def test_threshold_exactly_1000(self) -> None:
        # Exactly 1000 chars but no matching prefix
        data = "x" * 1000
        assert is_likely_image_data(data) is False

    def test_threshold_999_returns_false(self) -> None:
        # Just under the threshold
        data = "iVBOR" + "A" * 994  # total = 999
        assert is_likely_image_data(data) is False

    def test_threshold_1000_with_png_returns_true(self) -> None:
        # Exactly 1000 chars with PNG prefix
        data = "iVBOR" + "A" * 995  # total = 1000
        assert is_likely_image_data(data) is True


class TestSniffImageMime:
    """sniff_image_mime MIME type detection and data URI wrapping."""

    def test_already_data_uri_returned_unchanged(self) -> None:
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"
        assert sniff_image_mime(uri) == uri

    def test_data_uri_other_type_returned_unchanged(self) -> None:
        uri = "data:application/pdf;base64,abc"
        assert sniff_image_mime(uri) == uri

    def test_raw_png_base64(self) -> None:
        raw = "iVBORw0KGgoAAAANSUhEUg"
        result = sniff_image_mime(raw)
        assert result == f"data:image/png;base64,{raw}"

    def test_raw_jpeg_base64(self) -> None:
        raw = "/9j/4AAQSkZJRgABAQ"
        result = sniff_image_mime(raw)
        assert result == f"data:image/jpeg;base64,{raw}"

    def test_raw_gif_base64(self) -> None:
        raw = "R0lGODlhAQABAIAAAP"
        result = sniff_image_mime(raw)
        assert result == f"data:image/gif;base64,{raw}"

    def test_raw_webp_base64(self) -> None:
        raw = "UklGRiIAAABXRUJQVlA4"
        result = sniff_image_mime(raw)
        assert result == f"data:image/webp;base64,{raw}"

    def test_unknown_fallback_to_octet_stream(self) -> None:
        raw = "AAABBBCCCDDD"
        result = sniff_image_mime(raw)
        assert result == f"data:application/octet-stream;base64,{raw}"

    def test_empty_string_fallback(self) -> None:
        result = sniff_image_mime("")
        assert result == "data:application/octet-stream;base64,"
