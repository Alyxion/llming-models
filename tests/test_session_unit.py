"""Unit tests for ChatSession with mocked provider and client (no API calls)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.budget import InsufficientBudgetError, LLMBudgetManager
from llming_models.llm_base_models import ChatHistory, ChatMessage, Role
from llming_models.messages import LlmAIMessage, LlmHumanMessage, LlmMessageChunk, LlmSystemMessage
from llming_models.model_info import LLMInfo
from llming_models.session import ChatSession, LLMConfig
from llming_models.tools.mcp.config import MCPServerConfig


# ---------------------------------------------------------------------------
# LLMConfig Pydantic model
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig Pydantic model construction and defaults."""

    def test_minimal_construction(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"

    def test_defaults(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o")
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.max_input_tokens == 64000
        assert cfg.reasoning_effort is None
        assert cfg.max_history_images == 20
        assert cfg.condense_threshold_pct == 0.80
        assert cfg.condense_model is None
        assert cfg.condense_max_tokens == 5000
        assert cfg.tools is None
        assert cfg.tool_config is None
        assert cfg.mcp_servers is None
        assert cfg.base_url is None

    def test_full_construction(self):
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-sonnet",
            base_url="https://custom.api.com",
            temperature=0.3,
            max_tokens=2048,
            max_input_tokens=32000,
            max_history_images=10,
            condense_threshold_pct=0.5,
            condense_model="cheap-model",
            condense_max_tokens=3000,
            tools=["web_search", "generate_image"],
            tool_config={"generate_image": {"quality": "high"}},
            mcp_servers=[MCPServerConfig(command="python", args=["-m", "server"])],
        )
        assert cfg.provider == "anthropic"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        assert cfg.tools == ["web_search", "generate_image"]
        assert cfg.mcp_servers is not None
        assert len(cfg.mcp_servers) == 1
        assert cfg.mcp_servers[0].command == "python"

    def test_model_dump_roundtrip(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", temperature=0.5)
        data = cfg.model_dump()
        assert data["provider"] == "openai"
        assert data["temperature"] == 0.5
        cfg2 = LLMConfig(**data)
        assert cfg2.provider == cfg.provider
        assert cfg2.model == cfg.model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_MODEL_INFO = LLMInfo(
    provider="fake",
    name="fake_model",
    model="fake-model-v1",
    label="Fake Model",
    description="A fake model",
    input_token_price=1.0,
    output_token_price=2.0,
    max_input_tokens=16000,
    max_output_tokens=4096,
    supports_system_prompt=True,
)

FAKE_CONFIG = LLMConfig(
    provider="fake",
    model="fake-model-v1",
    temperature=0.5,
    max_tokens=1024,
    max_input_tokens=16000,
)


def _make_fake_provider():
    """Create a mock provider that returns FAKE_MODEL_INFO."""
    provider_cls = MagicMock()
    instance = MagicMock()
    instance.get_models.return_value = [FAKE_MODEL_INFO]
    instance.create_client.return_value = MagicMock()
    provider_cls.return_value = instance
    return provider_cls, instance


def _make_session(system_prompt=None, budget_manager=None, user_id=None):
    """Build a ChatSession with a fully mocked provider."""
    # Create a fresh copy of the config to avoid cross-test mutations
    config = FAKE_CONFIG.model_copy()
    provider_cls, provider_inst = _make_fake_provider()
    with patch("llming_models.session.get_provider", return_value=provider_cls):
        session = ChatSession(
            config=config,
            system_prompt=system_prompt,
            budget_manager=budget_manager,
            user_id=user_id,
        )
    # Replace the internal provider with our mock so create_client works
    session._provider = provider_inst
    return session, provider_inst


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestSessionInit:
    """Tests for ChatSession.__init__."""

    def test_basic_init(self):
        session, _ = _make_session()
        assert session.config == FAKE_CONFIG
        assert session.history.messages == []
        assert session.model_info.name == "fake_model"

    def test_init_with_system_prompt(self):
        session, _ = _make_session(system_prompt="You are a helper.")
        assert session.system_prompt == "You are a helper."

    def test_init_model_not_found_raises(self):
        provider_cls = MagicMock()
        instance = MagicMock()
        instance.get_models.return_value = []  # No models
        provider_cls.return_value = instance

        with patch("llming_models.session.get_provider", return_value=provider_cls):
            with pytest.raises(ValueError, match="not found in provider"):
                ChatSession(config=FAKE_CONFIG)


# ---------------------------------------------------------------------------
# add_message / clear_history
# ---------------------------------------------------------------------------


class TestMessageManagement:
    """Tests for add_message and clear_history."""

    def test_add_user_message(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Hello")
        assert len(session.history.messages) == 1
        assert session.history.messages[0].role == Role.USER
        assert session.history.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        session, _ = _make_session()
        session.add_message(Role.ASSISTANT, "Hi there")
        assert len(session.history.messages) == 1
        assert session.history.messages[0].role == Role.ASSISTANT

    def test_add_message_with_string_role(self):
        session, _ = _make_session()
        session.add_message("user", "Hello")
        assert session.history.messages[0].role == Role.USER

    def test_system_messages_not_stored(self):
        session, _ = _make_session()
        session.add_message(Role.SYSTEM, "System msg")
        assert len(session.history.messages) == 0

    def test_add_message_with_images(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Look at this", images=["base64data"])
        assert session.history.messages[0].images == ["base64data"]

    def test_clear_history(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Hello")
        session.add_message(Role.ASSISTANT, "Hi")
        session.clear_history()
        assert len(session.history.messages) == 0


# ---------------------------------------------------------------------------
# system_prompt getter / setter
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Tests for system_prompt property."""

    def test_getter(self):
        session, _ = _make_session(system_prompt="Test prompt")
        assert session.system_prompt == "Test prompt"

    def test_setter(self):
        session, _ = _make_session()
        session.system_prompt = "New prompt"
        assert session.system_prompt == "New prompt"

    def test_setter_none(self):
        session, _ = _make_session(system_prompt="Initial")
        session.system_prompt = None
        assert session.system_prompt is None


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Tests for _estimate_tokens."""

    def test_returns_positive_int(self):
        session, _ = _make_session()
        msgs = [LlmHumanMessage(content="Hello, how are you?")]
        tokens = session._estimate_tokens(msgs)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_empty_list_returns_zero(self):
        session, _ = _make_session()
        assert session._estimate_tokens([]) == 0

    def test_system_message_counted(self):
        session, _ = _make_session()
        msgs = [LlmSystemMessage(content="You are a helpful assistant.")]
        tokens = session._estimate_tokens(msgs)
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        session, _ = _make_session()
        short = session._estimate_tokens([LlmHumanMessage(content="Hi")])
        long = session._estimate_tokens([LlmHumanMessage(content="Hello, this is a much longer message with many words.")])
        assert long > short


# ---------------------------------------------------------------------------
# _prepare_messages
# ---------------------------------------------------------------------------


class TestPrepareMessages:
    """Tests for _prepare_messages."""

    def test_includes_system_prompt(self):
        session, _ = _make_session(system_prompt="Be helpful")
        session.add_message(Role.USER, "Hello")
        msgs = session._prepare_messages()
        assert isinstance(msgs[0], LlmSystemMessage)
        assert "Be helpful" in msgs[0].content

    def test_respects_token_limit(self):
        session, _ = _make_session()
        # Override to a very small input limit
        session.config.max_input_tokens = 50
        session.add_message(Role.USER, "A" * 500)
        session.add_message(Role.ASSISTANT, "B" * 500)
        session.add_message(Role.USER, "short msg")
        msgs = session._prepare_messages()
        # Should have dropped some old messages
        assert len(msgs) <= 3  # At most system + last couple messages

    def test_no_system_prompt(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Hello")
        msgs = session._prepare_messages()
        # No system message when no system prompt
        assert all(not isinstance(m, LlmSystemMessage) for m in msgs)

    def test_override_system_prompt(self):
        session, _ = _make_session(system_prompt="Default prompt")
        session.add_message(Role.USER, "Hello")
        msgs = session._prepare_messages(system_prompt="Override prompt")
        assert isinstance(msgs[0], LlmSystemMessage)
        assert "Override prompt" in msgs[0].content

    def test_condensed_summary_injected(self):
        session, _ = _make_session(system_prompt="Base")
        session._condensed_summary = "Previous conversation summary"
        session.add_message(Role.USER, "Hello")
        msgs = session._prepare_messages()
        system_content = msgs[0].content
        assert "Previous conversation summary" in system_content

    def test_context_preamble_prepended(self):
        session, _ = _make_session(system_prompt="Main prompt")
        session._context_preamble = "User: John Doe"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        assert "John Doe" in msgs[0].content
        assert "Main prompt" in msgs[0].content

    def test_skip_preamble(self):
        session, _ = _make_session(system_prompt="Main prompt")
        session._context_preamble = "User: John Doe"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages(skip_preamble=True)
        assert "John Doe" not in msgs[0].content

    def test_system_prompt_suffix_appended(self):
        session, _ = _make_session(system_prompt="Main")
        session._system_prompt_suffix = "Knowledge base catalog: ..."
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        assert "Knowledge base catalog" in msgs[0].content


# ---------------------------------------------------------------------------
# invalidate_client
# ---------------------------------------------------------------------------


class TestInvalidateClient:
    """Tests for invalidate_client."""

    def test_clears_client(self):
        session, _ = _make_session()
        session._client = MagicMock()
        session._last_client_config = ("some", "config")
        session.invalidate_client()
        assert session._client is None
        assert session._last_client_config is None


# ---------------------------------------------------------------------------
# chat_async non-streaming
# ---------------------------------------------------------------------------


class TestChatAsyncNonStreaming:
    """Tests for chat_async in non-streaming mode."""

    @pytest.mark.asyncio
    async def test_non_streaming_returns_ai_message(self):
        session, provider = _make_session()
        mock_client = MagicMock()
        mock_response = LlmAIMessage(content="Hello back!", response_metadata={})
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        provider.create_client.return_value = mock_client

        result = await session.chat_async("Hello", streaming=False)
        assert isinstance(result, LlmAIMessage)
        assert result.content == "Hello back!"

    @pytest.mark.asyncio
    async def test_non_streaming_adds_to_history(self):
        session, provider = _make_session()
        mock_client = MagicMock()
        mock_response = LlmAIMessage(content="Response", response_metadata={})
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        provider.create_client.return_value = mock_client

        await session.chat_async("Question", streaming=False)
        assert len(session.history.messages) == 2  # user + assistant
        assert session.history.messages[0].role == Role.USER
        assert session.history.messages[1].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_error_raises_runtime_error(self):
        session, provider = _make_session()
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(side_effect=Exception("API down"))
        provider.create_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="Chat completion failed"):
            await session.chat_async("Hello", streaming=False)

    @pytest.mark.asyncio
    async def test_insufficient_budget_reraises(self):
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.reserve_budget_async = AsyncMock(
            side_effect=InsufficientBudgetError("No budget", limit_name="test")
        )
        mock_budget.limits = {}

        session, provider = _make_session(budget_manager=mock_budget)
        with pytest.raises(InsufficientBudgetError):
            await session.chat_async("Hello", streaming=False)


# ---------------------------------------------------------------------------
# chat_async streaming
# ---------------------------------------------------------------------------


class TestChatAsyncStreaming:
    """Tests for chat_async in streaming mode."""

    @pytest.mark.asyncio
    async def test_streaming_returns_async_iterator(self):
        session, provider = _make_session()
        mock_client = MagicMock()

        chunks = [
            LlmMessageChunk(content="Hello", role=Role.ASSISTANT, index=0, is_final=False, response_metadata={}),
            LlmMessageChunk(content=" world", role=Role.ASSISTANT, index=1, is_final=True, response_metadata={}),
        ]

        async def fake_astream(messages, usage_callback=None):
            for chunk in chunks:
                yield chunk

        mock_client.astream = fake_astream
        provider.create_client.return_value = mock_client

        result = await session.chat_async("Hello", streaming=True)
        collected = []
        async for chunk in result:
            collected.append(chunk)
        assert len(collected) == 2
        assert collected[0].content == "Hello"
        assert collected[1].content == " world"


# ---------------------------------------------------------------------------
# Budget reserve/return flow
# ---------------------------------------------------------------------------


class TestBudgetFlow:
    """Tests for budget management in chat_async."""

    @pytest.mark.asyncio
    async def test_budget_reserve_and_return(self):
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.reserve_budget_async = AsyncMock(return_value=1024)
        mock_budget.return_unused_budget_async = AsyncMock()
        mock_budget.limits = {"test": MagicMock()}
        mock_budget.limits["test"].log_usage_async = AsyncMock()

        session, provider = _make_session(budget_manager=mock_budget, user_id="u1")
        mock_client = MagicMock()
        mock_response = LlmAIMessage(
            content="Response",
            response_metadata={"input_tokens": 100, "output_tokens": 50},
        )
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        provider.create_client.return_value = mock_client

        await session.chat_async("Hello", streaming=False)

        mock_budget.reserve_budget_async.assert_called_once()
        mock_budget.return_unused_budget_async.assert_called_once()


# ---------------------------------------------------------------------------
# copy_history_from / create_with_history
# ---------------------------------------------------------------------------


class TestHistoryCopy:
    """Tests for copy_history_from and create_with_history."""

    def test_copy_history_from(self):
        session_a, _ = _make_session()
        session_a.add_message(Role.USER, "Hello")
        session_a.add_message(Role.ASSISTANT, "Hi")

        session_b, _ = _make_session()
        session_b.add_message(Role.USER, "Old message")
        session_b.copy_history_from(session_a)

        assert len(session_b.history.messages) == 2
        assert session_b.history.messages[0].content == "Hello"
        assert session_b.history.messages[1].content == "Hi"

    def test_create_with_history(self):
        history = ChatHistory()
        history.add_message(ChatMessage(role=Role.USER, content="Q1"))
        history.add_message(ChatMessage(role=Role.ASSISTANT, content="A1"))
        history.add_message(ChatMessage(role=Role.SYSTEM, content="Should be skipped"))

        provider_cls, _ = _make_fake_provider()
        with patch("llming_models.session.get_provider", return_value=provider_cls):
            session = ChatSession.create_with_history(
                config=FAKE_CONFIG,
                history=history,
                system_prompt="Test",
            )

        # System messages should be excluded
        assert len(session.history.messages) == 2
        assert session.history.messages[0].content == "Q1"
        assert session.system_prompt == "Test"


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    """Tests for get_history."""

    def test_returns_history_object(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "test")
        h = session.get_history()
        assert isinstance(h, ChatHistory)
        assert len(h.messages) == 1


# ---------------------------------------------------------------------------
# _prepare_messages extended coverage
# ---------------------------------------------------------------------------


class TestPrepareMessagesExtended:
    """Extended coverage for _prepare_messages with combined prompt parts."""

    def test_preamble_and_suffix_together(self):
        session, _ = _make_session(system_prompt="Main body")
        session._context_preamble = "PREAMBLE_TEXT"
        session._system_prompt_suffix = "SUFFIX_TEXT"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        system_content = msgs[0].content
        # Preamble should be before main body
        preamble_pos = system_content.index("PREAMBLE_TEXT")
        body_pos = system_content.index("Main body")
        suffix_pos = system_content.index("SUFFIX_TEXT")
        assert preamble_pos < body_pos < suffix_pos

    def test_preamble_without_system_prompt(self):
        session, _ = _make_session()  # no system prompt
        session._context_preamble = "User: John"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        # Should still produce a system message with preamble
        assert isinstance(msgs[0], LlmSystemMessage)
        assert "User: John" in msgs[0].content

    def test_suffix_without_system_prompt(self):
        session, _ = _make_session()
        session._system_prompt_suffix = "Knowledge catalog"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        assert isinstance(msgs[0], LlmSystemMessage)
        assert "Knowledge catalog" in msgs[0].content

    def test_condensed_summary_with_preamble_and_suffix(self):
        session, _ = _make_session(system_prompt="Base")
        session._context_preamble = "PREAMBLE"
        session._condensed_summary = "SUMMARY"
        session._system_prompt_suffix = "SUFFIX"
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        content = msgs[0].content
        assert "PREAMBLE" in content
        assert "Base" in content
        assert "SUMMARY" in content
        assert "SUFFIX" in content

    def test_stale_messages_skipped(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "old message")
        session.add_message(Role.ASSISTANT, "old reply")
        # Mark first messages as stale
        session.history.messages[0].content_stale = True
        session.history.messages[1].content_stale = True
        session.add_message(Role.USER, "new message")
        msgs = session._prepare_messages()
        texts = [m.content for m in msgs]
        assert "old message" not in texts
        assert "old reply" not in texts
        assert "new message" in texts

    def test_model_without_system_prompt_support(self):
        session, _ = _make_session(system_prompt="System text")
        session.model_info.supports_system_prompt = False
        session.add_message(Role.USER, "Hi")
        msgs = session._prepare_messages()
        # System prompt should be injected as HumanMessage instead
        assert isinstance(msgs[0], LlmHumanMessage)
        assert "System text" in msgs[0].content

    def test_stale_images_skipped(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Look", images=["img_data"])
        session.history.messages[0].images_stale = True
        session.add_message(Role.USER, "More")
        msgs = session._prepare_messages()
        # The message with stale images should have no images in prepared output
        user_msgs = [m for m in msgs if isinstance(m, LlmHumanMessage)]
        first_user = user_msgs[0]
        assert first_user.images is None


# ---------------------------------------------------------------------------
# _estimate_tokens extended coverage
# ---------------------------------------------------------------------------


class TestEstimateTokensExtended:
    """Extended tests for _estimate_tokens with images and AI messages."""

    def test_ai_message_counted(self):
        session, _ = _make_session()
        msgs = [LlmAIMessage(content="This is a response from the assistant.")]
        tokens = session._estimate_tokens(msgs)
        assert tokens > 0

    def test_mixed_message_types(self):
        session, _ = _make_session()
        msgs = [
            LlmSystemMessage(content="You are helpful"),
            LlmHumanMessage(content="Hello"),
            LlmAIMessage(content="Hi there"),
        ]
        tokens = session._estimate_tokens(msgs)
        single = session._estimate_tokens([LlmHumanMessage(content="Hello")])
        assert tokens > single

    def test_image_tokens_added_for_human_message(self):
        session, _ = _make_session()
        # Create a fake base64 string long enough to count
        fake_b64 = "A" * 5000
        msgs_no_img = [LlmHumanMessage(content="Look at this")]
        msgs_with_img = [LlmHumanMessage(content="Look at this", images=[fake_b64])]
        no_img_tokens = session._estimate_tokens(msgs_no_img)
        with_img_tokens = session._estimate_tokens(msgs_with_img)
        assert with_img_tokens > no_img_tokens

    def test_ai_message_images_not_counted_as_input(self):
        """Generated images in assistant messages are output, not input."""
        session, _ = _make_session()
        fake_b64 = "A" * 5000
        msgs_no_img = [LlmAIMessage(content="Here is the image")]
        msgs_with_img = [LlmAIMessage(content="Here is the image", images=[fake_b64])]
        no_img_tokens = session._estimate_tokens(msgs_no_img)
        with_img_tokens = session._estimate_tokens(msgs_with_img)
        # AI message images should NOT add to input tokens
        assert no_img_tokens == with_img_tokens


# ---------------------------------------------------------------------------
# _estimate_image_tokens
# ---------------------------------------------------------------------------


class TestEstimateImageTokens:
    """Tests for _estimate_image_tokens with different providers."""

    def test_anthropic_provider(self):
        session, _ = _make_session()
        session.config.provider = "anthropic"
        fake_b64 = "A" * 10000
        tokens = session._estimate_image_tokens(fake_b64)
        # Anthropic: max(85, pixels // 750) where pixels = len * 1.5
        expected_pixels = int(10000 * 1.5)
        expected_tokens = max(85, expected_pixels // 750)
        assert tokens == expected_tokens

    def test_openai_provider(self):
        session, _ = _make_session()
        session.config.provider = "openai"
        fake_b64 = "A" * 10000
        tokens = session._estimate_image_tokens(fake_b64)
        # OpenAI: tile-based calculation
        assert tokens > 0
        assert tokens >= 85 + 170  # At least base + 1 tile

    def test_strips_data_uri_prefix(self):
        session, _ = _make_session()
        session.config.provider = "anthropic"
        raw_b64 = "A" * 10000
        data_uri = "data:image/png;base64," + raw_b64
        tokens_raw = session._estimate_image_tokens(raw_b64)
        tokens_uri = session._estimate_image_tokens(data_uri)
        assert tokens_raw == tokens_uri

    def test_small_image_anthropic_minimum(self):
        session, _ = _make_session()
        session.config.provider = "anthropic"
        # Very small base64 data
        small_b64 = "A" * 100
        tokens = session._estimate_image_tokens(small_b64)
        assert tokens == 85  # Minimum for Anthropic

    def test_large_image_more_tokens(self):
        session, _ = _make_session()
        session.config.provider = "anthropic"
        # Anthropic uses a simpler linear formula: max(85, pixels//750)
        small_tokens = session._estimate_image_tokens("A" * 1000)
        large_tokens = session._estimate_image_tokens("A" * 100000)
        assert large_tokens > small_tokens


# ---------------------------------------------------------------------------
# _enforce_image_limit
# ---------------------------------------------------------------------------


class TestEnforceImageLimit:
    """Tests for _enforce_image_limit."""

    def test_marks_old_images_stale(self):
        session, _ = _make_session()
        session.config.max_history_images = 2
        # Add 3 messages with 1 image each
        session.add_message(Role.USER, "first", images=["img1"])
        session.add_message(Role.USER, "second", images=["img2"])
        session.add_message(Role.USER, "third", images=["img3"])
        # Oldest should be stale, newest 2 should not
        assert session.history.messages[0].images_stale is True
        assert session.history.messages[1].images_stale is False
        assert session.history.messages[2].images_stale is False

    def test_no_stale_when_under_limit(self):
        session, _ = _make_session()
        session.config.max_history_images = 10
        session.add_message(Role.USER, "first", images=["img1"])
        session.add_message(Role.USER, "second", images=["img2"])
        assert session.history.messages[0].images_stale is False
        assert session.history.messages[1].images_stale is False

    def test_messages_without_images_unaffected(self):
        session, _ = _make_session()
        session.config.max_history_images = 1
        session.add_message(Role.USER, "no images")
        session.add_message(Role.USER, "has image", images=["img1"])
        session.add_message(Role.USER, "also no images")
        session.add_message(Role.USER, "has image 2", images=["img2"])
        assert session.history.messages[0].images_stale is False
        assert session.history.messages[1].images_stale is True
        assert session.history.messages[2].images_stale is False
        assert session.history.messages[3].images_stale is False

    def test_multiple_images_per_message(self):
        session, _ = _make_session()
        session.config.max_history_images = 3
        # First message with 2 images, second with 2 images => total 4, limit 3
        session.add_message(Role.USER, "two imgs", images=["a", "b"])
        session.add_message(Role.USER, "two more", images=["c", "d"])
        # Oldest message (2 images) should be stale since 2+2=4 > 3
        assert session.history.messages[0].images_stale is True
        assert session.history.messages[1].images_stale is False


# ---------------------------------------------------------------------------
# _extract_generated_images
# ---------------------------------------------------------------------------


class TestExtractGeneratedImages:
    """Tests for _extract_generated_images."""

    def test_no_images_in_normal_text(self):
        session, _ = _make_session()
        result = session._extract_generated_images("Hello, how are you?")
        assert result == []

    def test_extracts_image_from_function_result(self):
        session, _ = _make_session()
        fake_b64 = "A" * 200  # Must be > 100 chars
        response = '{"function": "generate_image", "function_call_result": "' + fake_b64 + '"}'
        result = session._extract_generated_images(response)
        assert len(result) == 1
        assert result[0] == fake_b64

    def test_ignores_non_image_functions(self):
        session, _ = _make_session()
        response = '{"function": "web_search", "function_call_result": "some result"}'
        result = session._extract_generated_images(response)
        assert result == []

    def test_ignores_short_results(self):
        session, _ = _make_session()
        # Result < 100 chars should be ignored (not likely an image)
        response = '{"function": "generate_image", "function_call_result": "short"}'
        result = session._extract_generated_images(response)
        assert result == []

    def test_multiple_images(self):
        session, _ = _make_session()
        fake_b64_1 = "A" * 200
        fake_b64_2 = "B" * 200
        response = (
            '{"function": "generate_image", "function_call_result": "' + fake_b64_1 + '"}'
            '{"function": "generate_image", "function_call_result": "' + fake_b64_2 + '"}'
        )
        result = session._extract_generated_images(response)
        assert len(result) == 2

    def test_missing_function_call_result_key(self):
        session, _ = _make_session()
        # Has generate_image but missing function_call_result keyword entirely
        response = 'Just text mentioning "generate_image" and "function_call_result"'
        result = session._extract_generated_images(response)
        assert result == []


# ---------------------------------------------------------------------------
# _clean_response_for_history
# ---------------------------------------------------------------------------


class TestCleanResponseForHistory:
    """Tests for _clean_response_for_history."""

    def test_strips_large_base64(self):
        session, _ = _make_session()
        fake_b64 = "A" * 2000
        response = '"function_call_result": "' + fake_b64 + '"'
        cleaned = session._clean_response_for_history(response)
        assert fake_b64 not in cleaned
        assert "[IMAGE_GENERATED]" in cleaned

    def test_preserves_short_content(self):
        session, _ = _make_session()
        response = '"function_call_result": "short text"'
        cleaned = session._clean_response_for_history(response)
        assert cleaned == response  # Not replaced because < 1000 chars

    def test_preserves_normal_text(self):
        session, _ = _make_session()
        response = "This is a normal assistant response with no images."
        cleaned = session._clean_response_for_history(response)
        assert cleaned == response


# ---------------------------------------------------------------------------
# check_and_condense
# ---------------------------------------------------------------------------


class TestCheckAndCondense:
    """Tests for check_and_condense."""

    @pytest.mark.asyncio
    async def test_no_condensation_below_threshold(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Short message")
        session.add_message(Role.ASSISTANT, "Short reply")
        result = await session.check_and_condense()
        assert result is False

    @pytest.mark.asyncio
    async def test_skip_if_already_condensing(self):
        session, _ = _make_session()
        session._is_condensing = True
        result = await session.check_and_condense()
        assert result is False

    @pytest.mark.asyncio
    async def test_force_condensation(self):
        session, provider = _make_session(system_prompt="Be helpful")
        session.add_message(Role.USER, "Tell me about quantum physics")
        session.add_message(Role.ASSISTANT, "Quantum physics is " + "fascinating " * 50)

        # Mock the condensation client
        mock_condense_client = MagicMock()

        async def fake_condense_stream(messages, **kwargs):
            yield LlmMessageChunk(
                content="Summary of the conversation about quantum physics.",
                role=Role.ASSISTANT,
                index=0,
                is_final=True,
                response_metadata={},
            )

        mock_condense_client.astream = fake_condense_stream
        provider.create_client.return_value = mock_condense_client

        # Provide a list of models for condense model selection
        cheap_model = LLMInfo(
            provider="fake",
            name="cheap_model",
            model="cheap-model-v1",
            label="Cheap",
            description="Cheap model",
            input_token_price=0.1,
            output_token_price=0.2,
            max_output_tokens=8192,
        )
        provider.get_models.return_value = [FAKE_MODEL_INFO, cheap_model]

        result = await session.check_and_condense(force=True)
        assert result is True
        assert session._condensed_summary is not None
        assert "quantum physics" in session._condensed_summary
        # All messages should be marked stale
        for msg in session.history.messages:
            assert msg.content_stale is True

    @pytest.mark.asyncio
    async def test_condense_callbacks_called(self):
        session, provider = _make_session(system_prompt="Be helpful")
        session.add_message(Role.USER, "Hello")
        session.add_message(Role.ASSISTANT, "Hi " * 100)

        mock_condense_client = MagicMock()

        async def fake_stream(messages, **kwargs):
            yield LlmMessageChunk(
                content="A valid summary of the conversation.",
                role=Role.ASSISTANT,
                index=0,
                is_final=True,
                response_metadata={},
            )

        mock_condense_client.astream = fake_stream
        provider.create_client.return_value = mock_condense_client
        provider.get_models.return_value = [FAKE_MODEL_INFO]

        start_cb = MagicMock()
        end_cb = MagicMock()
        session.on_condense_start = start_cb
        session.on_condense_end = end_cb

        await session.check_and_condense(force=True)
        start_cb.assert_called_once()
        end_cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_condense_empty_summary_skipped(self):
        session, provider = _make_session(system_prompt="Test")
        session.add_message(Role.USER, "Hello")
        session.add_message(Role.ASSISTANT, "Hi")

        mock_condense_client = MagicMock()

        async def empty_stream(messages, **kwargs):
            yield LlmMessageChunk(
                content="",
                role=Role.ASSISTANT,
                index=0,
                is_final=True,
                response_metadata={},
            )

        mock_condense_client.astream = empty_stream
        provider.create_client.return_value = mock_condense_client
        provider.get_models.return_value = [FAKE_MODEL_INFO]

        result = await session.check_and_condense(force=True)
        assert result is False
        assert session._condensed_summary is None


# ---------------------------------------------------------------------------
# chat_async streaming with base64 filtering and budget logging
# ---------------------------------------------------------------------------


class TestChatAsyncStreamingExtended:
    """Extended tests for streaming with base64 filtering and budget."""

    @pytest.mark.asyncio
    async def test_streaming_filters_base64_images(self):
        session, provider = _make_session()
        mock_client = MagicMock()

        fake_b64 = "A" * 500
        # Simulate chunks that contain a markdown base64 image
        chunks = [
            LlmMessageChunk(content="Here is the image: ![img](data:image/png;base64,", role=Role.ASSISTANT, index=0, is_final=False, response_metadata={}),
            LlmMessageChunk(content=fake_b64, role=Role.ASSISTANT, index=1, is_final=False, response_metadata={}),
            LlmMessageChunk(content=") done", role=Role.ASSISTANT, index=2, is_final=True, response_metadata={}),
        ]

        async def fake_astream(messages, usage_callback=None):
            for chunk in chunks:
                yield chunk

        mock_client.astream = fake_astream
        provider.create_client.return_value = mock_client

        result = await session.chat_async("Generate an image", streaming=True)
        collected = []
        async for chunk in result:
            collected.append(chunk)

        # The raw base64 data should be filtered from what is yielded
        full_yielded = "".join(c.content for c in collected if c.content)
        assert fake_b64 not in full_yielded

    @pytest.mark.asyncio
    async def test_streaming_budget_logging(self):
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.reserve_budget_async = AsyncMock(return_value=1024)
        mock_budget.return_unused_budget_async = AsyncMock()
        mock_limit = MagicMock()
        mock_limit.log_usage_async = AsyncMock()
        mock_budget.limits = {"test_limit": mock_limit}

        session, provider = _make_session(budget_manager=mock_budget, user_id="u1")
        mock_client = MagicMock()

        chunks = [
            LlmMessageChunk(
                content="Response text",
                role=Role.ASSISTANT,
                index=0,
                is_final=True,
                response_metadata={"total_input_tokens": 50, "total_output_tokens": 20},
            ),
        ]

        async def fake_astream(messages, usage_callback=None):
            if usage_callback:
                usage_callback(50, 20, 0)
            for chunk in chunks:
                yield chunk

        mock_client.astream = fake_astream
        provider.create_client.return_value = mock_client

        result = await session.chat_async("Hello", streaming=True)
        async for _ in result:
            pass

        mock_budget.reserve_budget_async.assert_called_once()
        mock_budget.return_unused_budget_async.assert_called_once()
        mock_limit.log_usage_async.assert_called_once()
        call_kwargs = mock_limit.log_usage_async.call_args[1]
        assert call_kwargs["operation_type"] == "normal_chat"
        assert call_kwargs["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_non_streaming_budget_logging(self):
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.reserve_budget_async = AsyncMock(return_value=1024)
        mock_budget.return_unused_budget_async = AsyncMock()
        mock_limit = MagicMock()
        mock_limit.log_usage_async = AsyncMock()
        mock_budget.limits = {"test_limit": mock_limit}

        session, provider = _make_session(budget_manager=mock_budget, user_id="u2")
        mock_client = MagicMock()
        mock_response = LlmAIMessage(
            content="Response",
            response_metadata={"input_tokens": 100, "output_tokens": 50},
        )
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        provider.create_client.return_value = mock_client

        await session.chat_async("Hello", streaming=False)

        mock_limit.log_usage_async.assert_called_once()
        call_kwargs = mock_limit.log_usage_async.call_args[1]
        assert call_kwargs["model_name"] == "fake.fake-model-v1"
        assert call_kwargs["tokens_input"] == 100
        assert call_kwargs["tokens_output"] == 50
        assert call_kwargs["operation_type"] == "normal_chat"


# ---------------------------------------------------------------------------
# _get_client caching and invalidation
# ---------------------------------------------------------------------------


class TestGetClientCaching:
    """Tests for _get_client caching behavior."""

    def test_client_cached_on_same_params(self):
        session, provider = _make_session()
        client1 = session._get_client(temperature=0.5, max_tokens=1024)
        client2 = session._get_client(temperature=0.5, max_tokens=1024)
        # Should be the same object (cached)
        assert client1 is client2
        # create_client called only once
        assert provider.create_client.call_count == 1

    def test_client_recreated_on_different_params(self):
        session, provider = _make_session()
        provider.create_client.side_effect = [MagicMock(), MagicMock()]
        client1 = session._get_client(temperature=0.5, max_tokens=1024)
        client2 = session._get_client(temperature=0.9, max_tokens=1024)
        assert client1 is not client2
        assert provider.create_client.call_count == 2

    def test_invalidate_forces_recreation(self):
        session, provider = _make_session()
        provider.create_client.side_effect = [MagicMock(), MagicMock()]
        client1 = session._get_client()
        session.invalidate_client()
        client2 = session._get_client()
        assert client1 is not client2

    def test_enforced_temperature_overrides(self):
        session, provider = _make_session()
        session.model_info.enforced_temperature = 0.0
        session._get_client(temperature=0.9)
        # The enforced temperature should be used in the config, not the provided one
        call_kwargs = provider.create_client.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_reasoning_effort_from_model_default(self):
        from llming_models.providers.llm_provider_models import ReasoningEffort
        session, provider = _make_session()
        session.model_info.reasoning = True
        session.model_info.default_reasoning_effort = ReasoningEffort.MEDIUM
        session._get_client()
        call_kwargs = provider.create_client.call_args[1]
        assert call_kwargs["reasoning_effort"] == ReasoningEffort.MEDIUM

    def test_mcp_server_key_in_cache(self):
        """MCP servers are included in the client cache key."""
        session, provider = _make_session()
        session.config.mcp_servers = [MCPServerConfig(command="python", args=["-m", "srv"])]
        provider.create_client.side_effect = [MagicMock(), MagicMock()]
        client1 = session._get_client()
        # Same mcp config -> cached
        client2 = session._get_client()
        assert client1 is client2
        assert provider.create_client.call_count == 1


# ---------------------------------------------------------------------------
# _build_toolboxes & tool_cost_callback
# ---------------------------------------------------------------------------


class TestBuildToolboxes:
    """Tests for _build_toolboxes and tool_cost_callback."""

    def test_tool_cost_callback_schedules_budget(self):
        """tool_cost_callback should schedule async budget operations."""
        mock_limit = MagicMock()
        mock_limit.reserve_budget_async = AsyncMock()
        mock_limit.log_usage_async = AsyncMock()
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.limits = {"test": mock_limit}

        session, provider = _make_session(budget_manager=mock_budget, user_id="u1")

        # Patch get_toolboxes_for_config inside toolbox_adapter module (lazy import target)
        captured_callback = None
        def mock_get_toolboxes(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("cost_callback")
            return []
        with patch("llming_models.tools.toolbox_adapter.get_toolboxes_for_config", side_effect=mock_get_toolboxes):
            session._build_toolboxes()

        # The callback should have been created
        assert captured_callback is not None

        # Call it in a running event loop context
        import asyncio
        loop = asyncio.new_event_loop()

        async def _test():
            captured_callback("generate_image", 0.05)
            # Allow scheduled tasks to run
            await asyncio.sleep(0.05)

        loop.run_until_complete(_test())
        loop.close()

    def test_tool_cost_callback_no_event_loop(self):
        """tool_cost_callback should silently skip when no event loop."""
        mock_budget = MagicMock(spec=LLMBudgetManager)
        mock_budget.limits = {"test": MagicMock()}

        session, provider = _make_session(budget_manager=mock_budget)

        captured_callback = None
        def mock_get_toolboxes(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("cost_callback")
            return []
        with patch("llming_models.tools.toolbox_adapter.get_toolboxes_for_config", side_effect=mock_get_toolboxes):
            session._build_toolboxes()

        # Calling outside event loop should not raise
        captured_callback("tool", 0.01)

    def test_build_toolboxes_openai_provider_creates_openai_client(self):
        """When provider is openai, _build_toolboxes should create an OpenAI client."""
        session, provider = _make_session()
        session.config.provider = "openai"

        with patch("llming_models.tools.toolbox_adapter.get_toolboxes_for_config", return_value=[]) as mock_gtfc:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
                with patch("llming_models.providers.openai.openai_client.OpenAI"):
                    with patch("llming_models.providers.openai.openai_client.AsyncOpenAI"):
                        session._build_toolboxes()
                        assert mock_gtfc.call_args[1].get("openai_client") is not None

    def test_build_toolboxes_azure_openai_creates_azure_client(self):
        """When provider is azure_openai, _build_toolboxes should create an Azure client."""
        session, provider = _make_session()
        session.config.provider = "azure_openai"

        with patch("llming_models.tools.toolbox_adapter.get_toolboxes_for_config", return_value=[]) as mock_gtfc:
            with patch.dict("os.environ", {
                "AZURE_OPENAI_API_KEY": "ak-test",
                "AZURE_OPENAI_ENDPOINT": "https://my.azure.endpoint",
            }):
                with patch("llming_models.providers.openai.openai_client.AzureOpenAI"):
                    with patch("llming_models.providers.openai.openai_client.AsyncAzureOpenAI"):
                        session._build_toolboxes()
                        assert mock_gtfc.call_args[1].get("openai_client") is not None


# ---------------------------------------------------------------------------
# MCP tool discovery
# ---------------------------------------------------------------------------


class TestDiscoverMCPTools:
    """Tests for _discover_mcp_tools and discover_tools."""

    @pytest.mark.asyncio
    async def test_discover_skips_if_already_discovered(self):
        session, _ = _make_session()
        session._mcp_tools_discovered = True
        session.config.mcp_servers = [MCPServerConfig(command="python")]
        await session._discover_mcp_tools()
        # No crash, no side effects

    @pytest.mark.asyncio
    async def test_discover_skips_without_mcp_servers(self):
        session, _ = _make_session()
        session.config.mcp_servers = None
        await session._discover_mcp_tools()
        assert session._mcp_tools_discovered is False

    @pytest.mark.asyncio
    async def test_discover_handles_connection_failure(self):
        """If an MCP server fails to connect, it logs error and continues."""
        from llming_models.tools.tool_definition import ToolDefinition, ToolSource

        session, _ = _make_session()
        session.config.mcp_servers = [
            MCPServerConfig(command="bad_server", label="failing"),
        ]

        with patch("llming_models.tools.mcp.create_connection") as mock_create:
            mock_conn = AsyncMock()
            mock_conn.start.side_effect = RuntimeError("Connection refused")
            mock_create.return_value = mock_conn

            with patch("llming_models.tools.tool_registry.get_default_registry") as mock_reg:
                mock_registry = MagicMock()
                mock_reg.return_value = mock_registry

                await session._discover_mcp_tools()

        assert session._mcp_tools_discovered is True

    @pytest.mark.asyncio
    async def test_discover_registers_tools(self):
        """MCP tools are registered in the registry and connections tracked."""
        from llming_models.tools.tool_definition import ToolDefinition, ToolSource, ToolUIMetadata

        session, _ = _make_session()
        session.config.mcp_servers = [
            MCPServerConfig(
                command="python",
                args=["-m", "server"],
                label="TestServer",
                enabled_by_default=True,
                category="Math",
            ),
        ]

        mock_tool = ToolDefinition(
            name="solve",
            description="Solve equations",
            source=ToolSource.MCP_STDIO,
            inputSchema={"type": "object", "properties": {}},
        )

        with patch("llming_models.tools.mcp.create_connection") as mock_create:
            mock_conn = AsyncMock()
            mock_conn.start = AsyncMock()
            mock_conn.list_tools = AsyncMock(return_value=[mock_tool])
            mock_create.return_value = mock_conn

            with patch("llming_models.tools.tool_registry.get_default_registry") as mock_reg:
                mock_registry = MagicMock()
                mock_registry._mcp_connections = {}
                mock_reg.return_value = mock_registry

                await session._discover_mcp_tools()

        assert session._mcp_tools_discovered is True
        assert "solve" in session._mcp_connections
        assert "TestServer" in session._mcp_server_groups
        group = session._mcp_server_groups["TestServer"]
        assert group["category"] == "Math"
        assert "solve" in group["tool_names"]
        # Tool should have been enabled (enabled_by_default=True, no default_tools filter)
        assert "solve" in session.config.tools

    @pytest.mark.asyncio
    async def test_discover_default_tools_filter(self):
        """Only tools in default_enabled_tools are enabled when specified."""
        from llming_models.tools.tool_definition import ToolDefinition, ToolSource

        session, _ = _make_session()
        session.config.mcp_servers = [
            MCPServerConfig(
                command="python",
                enabled_by_default=True,
                default_enabled_tools=["tool_a"],
            ),
        ]

        tool_a = ToolDefinition(name="tool_a", description="A", source=ToolSource.MCP_STDIO)
        tool_b = ToolDefinition(name="tool_b", description="B", source=ToolSource.MCP_STDIO)

        with patch("llming_models.tools.mcp.create_connection") as mock_create:
            mock_conn = AsyncMock()
            mock_conn.start = AsyncMock()
            mock_conn.list_tools = AsyncMock(return_value=[tool_a, tool_b])
            mock_create.return_value = mock_conn

            with patch("llming_models.tools.tool_registry.get_default_registry") as mock_reg:
                mock_registry = MagicMock()
                mock_registry._mcp_connections = {}
                mock_reg.return_value = mock_registry

                await session._discover_mcp_tools()

        assert "tool_a" in session.config.tools
        assert "tool_b" not in session.config.tools

    @pytest.mark.asyncio
    async def test_discover_collects_prompt_hints_and_renderers(self):
        """Prompt hints and client renderers from in-process servers are collected."""
        from llming_models.tools.tool_definition import ToolDefinition, ToolSource

        mock_server = AsyncMock()
        mock_server.list_tools = AsyncMock(return_value=[])
        mock_server.get_prompt_hints = AsyncMock(return_value=["Use ```math blocks"])
        mock_server.get_client_renderers = AsyncMock(return_value=[{"lang": "math", "js": "code"}])

        session, _ = _make_session()
        session.config.mcp_servers = [
            MCPServerConfig(server_instance=mock_server, label="MathSrv"),
        ]

        with patch("llming_models.tools.mcp.create_connection") as mock_create:
            mock_conn = AsyncMock()
            mock_conn.start = AsyncMock()
            mock_conn.list_tools = AsyncMock(return_value=[])
            mock_create.return_value = mock_conn

            with patch("llming_models.tools.tool_registry.get_default_registry") as mock_reg:
                mock_registry = MagicMock()
                mock_registry._mcp_connections = {}
                mock_reg.return_value = mock_registry

                await session._discover_mcp_tools()

        assert session.mcp_prompt_hints == ["Use ```math blocks"]
        assert session.mcp_client_renderers == [{"lang": "math", "js": "code"}]

    @pytest.mark.asyncio
    async def test_close_mcp_connections(self):
        session, _ = _make_session()
        mock_conn1 = AsyncMock()
        mock_conn2 = AsyncMock()
        mock_conn2.close.side_effect = RuntimeError("close error")
        session._mcp_connections = {"tool1": mock_conn1, "tool2": mock_conn2}
        session._mcp_tools_discovered = True

        await session.close_mcp_connections()

        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()
        assert session._mcp_connections == {}
        assert session._mcp_tools_discovered is False

    @pytest.mark.asyncio
    async def test_discover_tools_calls_discover_mcp_tools(self):
        session, _ = _make_session()
        session.config.mcp_servers = None

        await session.discover_tools()
        assert session._mcp_tools_discovered is False

    @pytest.mark.asyncio
    async def test_chat_async_triggers_mcp_discovery(self):
        """chat_async triggers MCP discovery on first call."""
        session, provider = _make_session()
        session.config.mcp_servers = [MCPServerConfig(command="python")]

        mock_client = MagicMock()
        mock_response = LlmAIMessage(content="Ok", response_metadata={})
        mock_client.ainvoke = AsyncMock(return_value=mock_response)
        provider.create_client.return_value = mock_client

        with patch.object(session, "_discover_mcp_tools", new_callable=AsyncMock) as mock_discover:
            await session.chat_async("Hi", streaming=False)
            mock_discover.assert_called_once()


# ---------------------------------------------------------------------------
# Condensation extended
# ---------------------------------------------------------------------------


class TestCondensationExtended:
    """Extended tests for condensation flow."""

    @pytest.mark.asyncio
    async def test_condense_with_existing_summary(self):
        """When condensing with an existing summary, it's included in conversation text."""
        session, provider = _make_session(system_prompt="Test")
        session._condensed_summary = "Previous summary of earlier conversation"
        session.add_message(Role.USER, "New question")
        session.add_message(Role.ASSISTANT, "New answer " * 50)

        mock_condense_client = MagicMock()
        captured_messages = None

        async def capture_stream(messages, **kwargs):
            nonlocal captured_messages
            captured_messages = messages
            yield LlmMessageChunk(
                content="Updated summary with new info from the conversation.",
                role=Role.ASSISTANT, index=0, is_final=True, response_metadata={},
            )

        mock_condense_client.astream = capture_stream
        provider.create_client.return_value = mock_condense_client
        provider.get_models.return_value = [FAKE_MODEL_INFO]

        await session.check_and_condense(force=True)

        # The human message passed to the condense model should mention previous summary
        assert captured_messages is not None
        human_content = captured_messages[1].content
        assert "Previous summary" in human_content

    @pytest.mark.asyncio
    async def test_condense_exception_handled(self):
        """If condensation fails, it logs error and returns True (finally block)."""
        session, provider = _make_session(system_prompt="Test")
        session.add_message(Role.USER, "Question")
        session.add_message(Role.ASSISTANT, "Answer")

        provider.get_models.return_value = [FAKE_MODEL_INFO]
        provider.create_client.side_effect = RuntimeError("Provider error")

        end_cb = MagicMock()
        session.on_condense_end = end_cb

        result = await session.check_and_condense(force=True)
        # The method should return True because it entered condensation
        assert result is True
        # on_condense_end should still be called (from finally block)
        end_cb.assert_called_once()
        assert session._is_condensing is False

    @pytest.mark.asyncio
    async def test_condense_selects_cheapest_model(self):
        """Auto-selects cheapest model with enough output capacity."""
        session, provider = _make_session(system_prompt="Test")
        session.add_message(Role.USER, "Q")
        session.add_message(Role.ASSISTANT, "A " * 50)

        expensive_model = LLMInfo(
            provider="fake", name="expensive", model="expensive-v1", label="Expensive",
            description="Expensive", input_token_price=10.0, output_token_price=20.0,
            max_output_tokens=8192,
        )
        cheap_model = LLMInfo(
            provider="fake", name="cheap", model="cheap-v1", label="Cheap",
            description="Cheap", input_token_price=0.1, output_token_price=0.2,
            max_output_tokens=8192,
        )
        provider.get_models.return_value = [FAKE_MODEL_INFO, expensive_model, cheap_model]

        mock_condense_client = MagicMock()
        async def fake_stream(messages, **kwargs):
            yield LlmMessageChunk(
                content="Summary of conversation.", role=Role.ASSISTANT,
                index=0, is_final=True, response_metadata={},
            )
        mock_condense_client.astream = fake_stream
        provider.create_client.return_value = mock_condense_client

        await session.check_and_condense(force=True)

        # The condense_model passed to create_client should be the cheapest
        call_kwargs = provider.create_client.call_args[1]
        assert call_kwargs["model"] == "cheap-v1"

    @pytest.mark.asyncio
    async def test_condense_progress_callback(self):
        """on_condense_progress is called during streaming."""
        session, provider = _make_session(system_prompt="Test")
        session.add_message(Role.USER, "Question")
        session.add_message(Role.ASSISTANT, "Answer " * 20)

        progress_calls = []
        session.on_condense_progress = lambda pct: progress_calls.append(pct)

        mock_condense_client = MagicMock()
        # The summary must be >= 50 chars to not be rejected
        long_summary = "This is a detailed summary of the conversation about the question and answer. " * 3
        async def fake_stream(messages, **kwargs):
            yield LlmMessageChunk(content=long_summary[:40], role=Role.ASSISTANT, index=0, is_final=False, response_metadata={})
            yield LlmMessageChunk(content=long_summary[40:], role=Role.ASSISTANT, index=1, is_final=True, response_metadata={})
        mock_condense_client.astream = fake_stream
        provider.create_client.return_value = mock_condense_client
        provider.get_models.return_value = [FAKE_MODEL_INFO]

        await session.check_and_condense(force=True)
        assert len(progress_calls) >= 1
        assert all(0 <= p <= 1.0 for p in progress_calls)

    def test_enforced_temperature_overrides(self):
        session, provider = _make_session()
        session.model_info.enforced_temperature = 0.0
        session._get_client(temperature=0.9)
        call_kwargs = provider.create_client.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# discover_tools / close_mcp_connections
# ---------------------------------------------------------------------------


class TestMCPToolDiscovery:
    """Tests for discover_tools and close_mcp_connections."""

    @pytest.mark.asyncio
    async def test_discover_tools_no_servers(self):
        session, _ = _make_session()
        session.config.mcp_servers = None
        await session.discover_tools()
        assert session._mcp_tools_discovered is False

    @pytest.mark.asyncio
    async def test_discover_tools_already_discovered(self):
        session, _ = _make_session()
        session._mcp_tools_discovered = True
        session.config.mcp_servers = [MCPServerConfig(command="python")]
        await session.discover_tools()
        # Should not re-discover
        assert session._mcp_tools_discovered is True

    @pytest.mark.asyncio
    async def test_close_mcp_connections(self):
        session, _ = _make_session()
        mock_conn = AsyncMock()
        session._mcp_connections = {"tool1": mock_conn}
        session._mcp_tools_discovered = True
        await session.close_mcp_connections()
        mock_conn.close.assert_called_once()
        assert session._mcp_connections == {}
        assert session._mcp_tools_discovered is False

    @pytest.mark.asyncio
    async def test_close_mcp_connections_handles_errors(self):
        session, _ = _make_session()
        mock_conn = AsyncMock()
        mock_conn.close.side_effect = Exception("Connection error")
        session._mcp_connections = {"tool1": mock_conn}
        session._mcp_tools_discovered = True
        # Should not raise
        await session.close_mcp_connections()
        assert session._mcp_connections == {}


# ---------------------------------------------------------------------------
# context_preamble and system_prompt_suffix setters/getters
# ---------------------------------------------------------------------------


class TestContextPreambleAndSuffix:
    """Tests for context_preamble and system_prompt_suffix."""

    def test_context_preamble_default_none(self):
        session, _ = _make_session()
        assert session._context_preamble is None

    def test_context_preamble_set_and_get(self):
        session, _ = _make_session()
        session._context_preamble = "User context"
        assert session._context_preamble == "User context"

    def test_system_prompt_suffix_default_none(self):
        session, _ = _make_session()
        assert session._system_prompt_suffix is None

    def test_system_prompt_suffix_set_and_get(self):
        session, _ = _make_session()
        session._system_prompt_suffix = "Knowledge base"
        assert session._system_prompt_suffix == "Knowledge base"


# ---------------------------------------------------------------------------
# _estimate_full_context_tokens
# ---------------------------------------------------------------------------


class TestEstimateFullContextTokens:
    """Tests for _estimate_full_context_tokens."""

    def test_counts_all_messages(self):
        session, _ = _make_session(system_prompt="System text")
        session.add_message(Role.USER, "Hello")
        session.add_message(Role.ASSISTANT, "Hi there")
        tokens = session._estimate_full_context_tokens()
        assert tokens > 0

    def test_skips_stale_content(self):
        session, _ = _make_session()
        session.add_message(Role.USER, "Old message " * 100)
        session.add_message(Role.ASSISTANT, "Old reply " * 100)
        tokens_before = session._estimate_full_context_tokens()
        session.history.messages[0].content_stale = True
        session.history.messages[1].content_stale = True
        tokens_after = session._estimate_full_context_tokens()
        assert tokens_after < tokens_before

    def test_includes_condensed_summary(self):
        session, _ = _make_session(system_prompt="Base")
        tokens_no_summary = session._estimate_full_context_tokens()
        session._condensed_summary = "This is a summary " * 50
        tokens_with_summary = session._estimate_full_context_tokens()
        assert tokens_with_summary > tokens_no_summary


# ---------------------------------------------------------------------------
# clear_history clears condensed summary
# ---------------------------------------------------------------------------


class TestClearHistoryExtended:
    """Tests that clear_history also resets condensed summary."""

    def test_clears_condensed_summary(self):
        session, _ = _make_session()
        session._condensed_summary = "Some old summary"
        session.add_message(Role.USER, "Hello")
        session.clear_history()
        assert session._condensed_summary is None
        assert len(session.history.messages) == 0


# ---------------------------------------------------------------------------
# mcp_prompt_hints / mcp_client_renderers properties
# ---------------------------------------------------------------------------


class TestMCPProperties:
    """Tests for mcp_prompt_hints and mcp_client_renderers."""

    def test_mcp_prompt_hints_empty_by_default(self):
        session, _ = _make_session()
        assert session.mcp_prompt_hints == []

    def test_mcp_prompt_hints_returns_stored(self):
        session, _ = _make_session()
        session._mcp_prompt_hints = ["hint1", "hint2"]
        assert session.mcp_prompt_hints == ["hint1", "hint2"]

    def test_mcp_client_renderers_empty_by_default(self):
        session, _ = _make_session()
        assert session.mcp_client_renderers == []

    def test_mcp_client_renderers_returns_stored(self):
        session, _ = _make_session()
        session._mcp_client_renderers = [{"type": "html", "code": "<div/>"}]
        assert len(session.mcp_client_renderers) == 1
