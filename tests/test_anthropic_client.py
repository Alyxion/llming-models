"""Unit tests for the Anthropic client (fully mocked SDK)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmSystemMessage,
)
from llming_models.llm_base_models import Role
from llming_models.tools.llm_tool import LlmTool
from llming_models.tools.llm_toolbox import LlmToolbox
from llming_models.tools.tool_call import ToolCallStatus


# ---------------------------------------------------------------------------
# Helpers to build realistic mock response objects and async context managers
# ---------------------------------------------------------------------------

class _AsyncStreamCtx:
    """A non-coroutine async context manager wrapping an async iterator + final message.

    ``self._aclient.messages.stream(**kw)`` is called *without* ``await`` and
    returns an async context manager directly.  ``AsyncMock`` would turn the
    call itself into a coroutine, which breaks ``async with``.  This helper
    avoids that problem.
    """

    def __init__(self, events, final_message):
        self._events = events
        self._final_message = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self._async_gen()

    async def _async_gen(self):
        for ev in self._events:
            yield ev

    async def get_final_message(self):
        return self._final_message

def _make_text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_tool_use_block(
    name: str, input_: dict, block_id: str = "toolu_1"
) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def _make_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def _make_response(
    text: str = "Hello!",
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-4-5-20250929",
    usage: Any = None,
    content: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content if content is not None else [_make_text_block(text)],
        stop_reason=stop_reason,
        model=model,
        usage=usage or _make_usage(),
    )


# ---------------------------------------------------------------------------
# _convert_messages tests
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def test_text_only_messages(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [
            LlmSystemMessage(content="Be helpful"),
            LlmHumanMessage(content="Hi"),
            LlmAIMessage(content="Hello"),
        ]
        system, converted = _convert_messages(msgs)
        assert system == "Be helpful"
        assert len(converted) == 2
        assert converted[0] == {"role": "user", "content": "Hi"}
        assert converted[1] == {"role": "assistant", "content": "Hello"}

    def test_no_system_message(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmHumanMessage(content="Hi")]
        system, converted = _convert_messages(msgs)
        assert system is None
        assert len(converted) == 1

    def test_user_message_with_images(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="Describe this", images=["iVBORbase64data"]),
        ]
        system, converted = _convert_messages(msgs)
        assert len(converted) == 1
        parts = converted[0]["content"]
        assert parts[0] == {"type": "text", "text": "Describe this"}
        assert parts[1]["type"] == "image"
        assert parts[1]["source"]["media_type"] == "image/png"
        assert parts[1]["source"]["data"] == "iVBORbase64data"

    def test_jpeg_image_detection(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmHumanMessage(content="pic", images=["/9j/fakedata"])]
        _, converted = _convert_messages(msgs)
        assert converted[0]["content"][1]["source"]["media_type"] == "image/jpeg"

    def test_gif_image_detection(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmHumanMessage(content="gif", images=["R0lGbase64"])]
        _, converted = _convert_messages(msgs)
        assert converted[0]["content"][1]["source"]["media_type"] == "image/gif"

    def test_webp_image_detection(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmHumanMessage(content="webp", images=["UklGbase64"])]
        _, converted = _convert_messages(msgs)
        assert converted[0]["content"][1]["source"]["media_type"] == "image/webp"

    def test_unknown_image_defaults_to_png(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmHumanMessage(content="x", images=["AAAA_unknown"])]
        _, converted = _convert_messages(msgs)
        assert converted[0]["content"][1]["source"]["media_type"] == "image/png"

    def test_image_limit(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="old", images=["iVBORold"]),
            LlmHumanMessage(content="new", images=["iVBORnew"]),
        ]
        _, converted = _convert_messages(msgs, max_image_history=1)
        # Old message should have images stripped (text only)
        assert isinstance(converted[0]["content"], str)
        # New message should keep images
        assert isinstance(converted[1]["content"], list)

    def test_multiple_images_in_one_message(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="multi", images=["iVBOR1", "/9j/2"]),
        ]
        _, converted = _convert_messages(msgs)
        parts = converted[0]["content"]
        assert len(parts) == 3  # 1 text + 2 images
        assert parts[1]["source"]["media_type"] == "image/png"
        assert parts[2]["source"]["media_type"] == "image/jpeg"

    def test_ai_message_no_images_simple_format(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_messages

        msgs = [LlmAIMessage(content="response")]
        _, converted = _convert_messages(msgs)
        assert converted[0] == {"role": "assistant", "content": "response"}


# ---------------------------------------------------------------------------
# _convert_tools tests
# ---------------------------------------------------------------------------

class TestConvertTools:
    def test_llm_tool(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools

        tool = LlmTool(
            name="calc", description="Calculate",
            func=lambda x: x,
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        toolbox = LlmToolbox(name="math", description="Math tools", tools=[tool])
        result = _convert_tools([toolbox])
        assert len(result) == 1
        assert result[0]["name"] == "calc"
        assert result[0]["description"] == "Calculate"
        assert result[0]["input_schema"]["type"] == "object"

    def test_string_web_search(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools

        toolbox = LlmToolbox(name="search", description="Search", tools=["web_search"])
        result = _convert_tools([toolbox])
        assert len(result) == 1
        assert result[0]["type"] == "web_search_20250305"
        assert result[0]["name"] == "web_search"
        assert result[0]["max_uses"] == 5

    def test_string_unknown_tool_warns(self, caplog):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools
        import logging

        toolbox = LlmToolbox(name="tb", description="d", tools=["unknown_tool"])
        with caplog.at_level(logging.WARNING):
            result = _convert_tools([toolbox])
        assert len(result) == 0
        assert "Unknown native tool" in caplog.text

    def test_dict_web_search_with_config(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools

        toolbox = LlmToolbox(name="tb", description="d", tools=[{
            "type": "web_search",
            "max_uses": 10,
            "allowed_domains": ["example.com"],
            "blocked_domains": ["bad.com"],
            "user_location": {"country": "US"},
        }])
        result = _convert_tools([toolbox])
        assert len(result) == 1
        assert result[0]["type"] == "web_search_20250305"
        assert result[0]["max_uses"] == 10
        assert result[0]["allowed_domains"] == ["example.com"]
        assert result[0]["blocked_domains"] == ["bad.com"]
        assert result[0]["user_location"] == {"country": "US"}

    def test_dict_web_search_20250305_type(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools

        toolbox = LlmToolbox(name="tb", description="d", tools=[{
            "type": "web_search_20250305",
        }])
        result = _convert_tools([toolbox])
        assert result[0]["type"] == "web_search_20250305"

    def test_dict_unknown_type_warns(self, caplog):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools
        import logging

        toolbox = LlmToolbox(name="tb", description="d", tools=[{"type": "unknown"}])
        with caplog.at_level(logging.WARNING):
            result = _convert_tools([toolbox])
        assert len(result) == 0
        assert "Unknown native tool type" in caplog.text

    def test_multiple_toolboxes(self):
        from llming_models.providers.anthropic.anthropic_client import _convert_tools

        tool1 = LlmTool(name="t1", description="d", func=lambda: None, parameters={"type": "object", "properties": {}})
        tool2 = LlmTool(name="t2", description="d", func=lambda: None, parameters={"type": "object", "properties": {}})
        tb1 = LlmToolbox(name="tb1", description="d", tools=[tool1])
        tb2 = LlmToolbox(name="tb2", description="d", tools=[tool2, "web_search"])
        result = _convert_tools([tb1, tb2])
        assert len(result) == 3
        names = [r.get("name") for r in result]
        assert "t1" in names
        assert "t2" in names
        assert "web_search" in names


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestAnthropicInit:
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_direct_api_key(self, mock_anthropic, mock_async_anthropic):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        assert client.model == "claude-sonnet-4-5-20250929"
        mock_anthropic.assert_called_once_with(api_key="sk-ant-test")
        mock_async_anthropic.assert_called_once_with(api_key="sk-ant-test")

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropicFoundry")
    @patch("llming_models.providers.anthropic.anthropic_client.AnthropicFoundry")
    def test_azure_base_url(self, mock_foundry, mock_async_foundry):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key="azure-key",
            model="claude-sonnet-4-5-20250929",
            azure_base_url="https://my-azure.example.com",
        )
        mock_foundry.assert_called_once_with(
            api_key="azure-key", base_url="https://my-azure.example.com"
        )
        mock_async_foundry.assert_called_once_with(
            api_key="azure-key", base_url="https://my-azure.example.com"
        )

    def test_missing_api_key_raises(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicClient(api_key=None, model="claude-sonnet-4-5-20250929")

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_toolboxes_stored(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        tb = LlmToolbox(name="tb", description="d", tools=[])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])
        assert client.toolboxes == [tb]

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_default_toolboxes_empty(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        assert client.toolboxes == []


# ---------------------------------------------------------------------------
# _mark_cache_control tests (extend beyond test_caching.py)
# ---------------------------------------------------------------------------

class TestMarkCacheControlExtended:
    """Additional tests for edge cases not covered in test_caching.py."""

    def test_non_dict_last_block_is_noop(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        # If the list contains non-dict items, should not crash
        msg = {"role": "user", "content": ["just a string"]}
        AnthropicClient._mark_cache_control(msg)
        # String in list is not a dict, so no cache_control added
        assert msg["content"] == ["just a string"]


# ---------------------------------------------------------------------------
# _apply_cache_control tests
# ---------------------------------------------------------------------------

class TestApplyCacheControl:
    def test_empty_messages(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        messages: list = []
        AnthropicClient._apply_cache_control(messages)
        assert messages == []

    def test_single_message(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        messages = [{"role": "user", "content": "hi"}]
        AnthropicClient._apply_cache_control(messages)
        # First message gets cache_control
        assert isinstance(messages[0]["content"], list)
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_three_messages(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        AnthropicClient._apply_cache_control(messages)
        # First message gets breakpoint
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Penultimate (index 1) gets breakpoint
        assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Last message: no breakpoint added by _apply_cache_control itself
        assert isinstance(messages[2]["content"], str)

    def test_two_messages_no_penultimate(self):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        AnthropicClient._apply_cache_control(messages)
        # First gets breakpoint
        assert isinstance(messages[0]["content"], list)
        # Only 2 messages, so no penultimate breakpoint (need >= 3)
        assert isinstance(messages[1]["content"], str)


# ---------------------------------------------------------------------------
# _build_kwargs tests
# ---------------------------------------------------------------------------

class TestBuildKwargs:
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_basic_kwargs(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        # reasoning=False so temperature is still passed through the API kwargs.
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929",
            temperature=0.5, max_tokens=2048, reasoning=False,
        )
        kwargs = client._build_kwargs([
            LlmSystemMessage(content="Be helpful"),
            LlmHumanMessage(content="Hi"),
        ])

        assert kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 2048
        assert "system" in kwargs
        assert kwargs["system"][0]["text"] == "Be helpful"
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_kwargs_without_system(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert "system" not in kwargs

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_kwargs_default_max_tokens(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert kwargs["max_tokens"] == 4096

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_kwargs_with_tools(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        tool = LlmTool(
            name="calc", description="Calculate", func=lambda: None,
            parameters={"type": "object", "properties": {}},
        )
        tb = LlmToolbox(name="math", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_kwargs_no_tools_when_empty(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert "tools" not in kwargs

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_reasoning_default_is_true(self, mock_anthropic, mock_async):
        """New-default guard: unspecified ``reasoning`` is the safe ``True``.

        Claude 4.x reasoning models reject the ``temperature`` parameter, so the
        safe default is to omit it. This test pins the default so a refactor
        can't accidentally flip it back to False.
        """
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-opus-4-8")
        assert client.reasoning is True

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_reasoning_model_omits_temperature(self, mock_anthropic, mock_async):
        """With ``reasoning=True``, ``temperature`` must NOT reach the API.

        Regression guard for the production 400 error:
        ``'`temperature` is deprecated for this model.'`` against Opus 4.8.
        """
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-opus-4-8",
            temperature=0.5, reasoning=True,
        )
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert "temperature" not in kwargs
        assert kwargs["model"] == "claude-opus-4-8"
        assert kwargs["max_tokens"] == 4096

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_non_reasoning_model_keeps_temperature(self, mock_anthropic, mock_async):
        """Legacy models that still accept ``temperature`` must keep receiving it."""
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-3-5-sonnet-20241022",
            temperature=0.3, reasoning=False,
        )
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert kwargs["temperature"] == 0.3


class TestLLMInfoReasoningDefault:
    """The dataclass default must stay ``True`` so that forgetting to set it
    on a new frontier model is the safe choice."""

    def test_llm_info_reasoning_defaults_to_true(self):
        from llming_models.providers.llm_provider_models import LLMInfo

        info = LLMInfo(
            provider="test", name="x", label="X", model="x",
            description="", input_token_price=0.0,
        )
        assert info.reasoning is True

    def test_legacy_providers_opt_out(self):
        """Mistral and DeepSeek models must keep ``reasoning=False`` so they
        continue to receive ``temperature``."""
        from llming_models.providers.mistral.mistral_models import MISTRAL_MODELS
        from llming_models.providers.together.deepseek.deepseek_models import (
            TOGETHER_DEEPSEEK_MODELS,
        )

        assert MISTRAL_MODELS, "MISTRAL_MODELS should not be empty"
        assert TOGETHER_DEEPSEEK_MODELS, "TOGETHER_DEEPSEEK_MODELS should not be empty"
        for m in MISTRAL_MODELS:
            assert m.reasoning is False, f"{m.name} should be non-reasoning"
        for m in TOGETHER_DEEPSEEK_MODELS:
            assert m.reasoning is False, f"{m.name} should be non-reasoning"


class TestAnthropicProviderReasoningPropagation:
    """The provider must look up the model's ``reasoning`` flag from model
    metadata and pass it to the client — otherwise the client falls back to
    its default and we lose the per-model opt-out story."""

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_provider_passes_reasoning_from_model_metadata(
        self, mock_anthropic, mock_async
    ):
        from llming_models.providers.anthropic.anthropic_provider import (
            AnthropicProvider,
        )
        from llming_models.providers.anthropic.anthropic_models import (
            ANTHROPIC_MODELS,
        )
        from llming_models.credentials import ProviderCredentials

        # Pick a real, declared Claude 4.x model and confirm the provider
        # reads its ``reasoning=True`` out of ANTHROPIC_MODELS and hands that
        # through to the constructed client.
        opus = next(m for m in ANTHROPIC_MODELS if m.name == "claude_opus")
        assert opus.reasoning is True

        provider = AnthropicProvider(
            credentials=ProviderCredentials(api_key="sk-ant-test")
        )
        client = provider.create_client(model=opus.model)
        assert client.reasoning is True


# ---------------------------------------------------------------------------
# _execute_tool tests
# ---------------------------------------------------------------------------

class TestExecuteTool:
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_execute_known_tool(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        tool = LlmTool(
            name="add", description="Add", func=lambda a, b: a + b,
            parameters={"type": "object", "properties": {}},
        )
        tb = LlmToolbox(name="math", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        result = client._execute_tool("add", {"a": 2, "b": 3})
        assert result == 5

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_execute_unknown_tool(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        result = client._execute_tool("nonexistent", {})
        assert result is None

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_execute_skips_string_tools(self, mock_anthropic, mock_async):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        tb = LlmToolbox(name="tb", description="d", tools=["web_search"])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        result = client._execute_tool("web_search", {})
        assert result is None


# ---------------------------------------------------------------------------
# invoke() tests
# ---------------------------------------------------------------------------

class TestInvoke:
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_invoke_simple_text(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        resp = _make_response("Hello world!")
        mock_client.messages.create.return_value = resp

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        result = client.invoke([LlmHumanMessage(content="Hi")])

        assert isinstance(result, LlmAIMessage)
        assert result.content == "Hello world!"
        assert result.response_metadata["input_tokens"] == 10
        assert result.response_metadata["output_tokens"] == 20
        assert result.response_metadata["model"] == "claude-sonnet-4-5-20250929"

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_invoke_tool_loop(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # First response: tool use
        tool_block = _make_tool_use_block("calc", {"expr": "2+2"})
        resp1 = _make_response(
            stop_reason="tool_use",
            content=[_make_text_block("Let me calculate."), tool_block],
        )
        # Second response: final text
        resp2 = _make_response("The answer is 4.")

        mock_client.messages.create.side_effect = [resp1, resp2]

        tool = LlmTool(
            name="calc", description="Calculate", func=lambda expr: 4,
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        tb = LlmToolbox(name="math", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        result = client.invoke([LlmHumanMessage(content="What is 2+2?")])

        assert result.content == "The answer is 4."
        assert mock_client.messages.create.call_count == 2

        # Verify tool results were appended to messages
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        messages = second_call_kwargs[1]["messages"]
        # Should have: original user msg + assistant msg + tool result msg
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"

    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    def test_invoke_concatenates_text_blocks(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        resp = _make_response(
            content=[_make_text_block("Part 1 "), _make_text_block("Part 2")]
        )
        mock_client.messages.create.return_value = resp

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        result = client.invoke([LlmHumanMessage(content="Hi")])
        assert result.content == "Part 1 Part 2"


# ---------------------------------------------------------------------------
# ainvoke() tests
# ---------------------------------------------------------------------------

class TestAInvoke:
    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_ainvoke_simple_text(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        resp = _make_response("Async hello!")
        mock_aclient.messages.create.return_value = resp

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        result = await client.ainvoke([LlmHumanMessage(content="Hi")])

        assert result.content == "Async hello!"
        assert result.response_metadata["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_ainvoke_tool_loop(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        tool_block = _make_tool_use_block("search", {"query": "test"})
        resp1 = _make_response(
            stop_reason="tool_use",
            content=[tool_block],
        )
        resp2 = _make_response("Found results.")
        mock_aclient.messages.create.side_effect = [resp1, resp2]

        tool = LlmTool(
            name="search", description="Search", func=lambda query: "results",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        result = await client.ainvoke([LlmHumanMessage(content="Search")])
        assert result.content == "Found results."
        assert mock_aclient.messages.create.call_count == 2

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_ainvoke_tool_result_null(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        tool_block = _make_tool_use_block("noop", {})
        resp1 = _make_response(stop_reason="tool_use", content=[tool_block])
        resp2 = _make_response("Done.")
        mock_aclient.messages.create.side_effect = [resp1, resp2]

        tool = LlmTool(
            name="noop", description="No-op", func=lambda: None,
            parameters={"type": "object", "properties": {}},
        )
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb]
        )
        result = await client.ainvoke([LlmHumanMessage(content="Do nothing")])

        # Check tool result content is "null"
        second_call_msgs = mock_aclient.messages.create.call_args_list[1][1]["messages"]
        tool_result = second_call_msgs[-1]["content"][0]
        assert tool_result["content"] == "null"


# ---------------------------------------------------------------------------
# astream() tests
# ---------------------------------------------------------------------------

class TestAStream:
    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_text_and_final_chunk(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        events = [
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hello ")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="world")),
        ]
        final_message = _make_response(
            "Hello world", stop_reason="end_turn",
            usage=_make_usage(input_tokens=50, output_tokens=30),
        )
        mock_aclient.messages.stream = MagicMock(return_value=_AsyncStreamCtx(events, final_message))

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(chunk)

        assert len(chunks) >= 3
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world"
        final = chunks[-1]
        assert final.is_final is True
        assert final.response_metadata["total_input_tokens"] == 50
        assert final.response_metadata["total_output_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_usage_callback(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        final_message = _make_response(
            "", stop_reason="end_turn",
            usage=_make_usage(
                input_tokens=100, output_tokens=50,
                cache_read_input_tokens=30, cache_creation_input_tokens=10,
            ),
        )
        mock_aclient.messages.stream = MagicMock(return_value=_AsyncStreamCtx([], final_message))

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        usage_cb = MagicMock()

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        # Usage callback should be called with total input (100 + 10 + 30 = 140) and output (50)
        usage_cb.assert_called_once()
        args = usage_cb.call_args
        assert args[0][0] == 140  # input_tokens + cache_creation + cache_read
        assert args[0][1] == 50   # output_tokens
        assert args[1]["cached_input_tokens"] == 30

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_usage_callback_error_handled(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        final_message = _make_response(
            "", stop_reason="end_turn",
            usage=_make_usage(input_tokens=10, output_tokens=5),
        )
        mock_aclient.messages.stream = MagicMock(return_value=_AsyncStreamCtx([], final_message))

        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929")
        usage_cb = MagicMock(side_effect=RuntimeError("callback boom"))

        # Should not raise despite callback error
        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_tool_call_flow(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        # First iteration events: tool call
        events_1 = [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="toolu_abc", name="calc")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json='{"expr": "2+2"}')),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(
            stop_reason="tool_use",
            content=[_make_text_block("Let me calc"), _make_tool_use_block("calc", {"expr": "2+2"}, "toolu_abc")],
            usage=_make_usage(input_tokens=20, output_tokens=10),
        )

        # Second iteration events: text response
        events_2 = [
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="4")),
        ]
        final_msg_2 = _make_response("4", stop_reason="end_turn", usage=_make_usage(input_tokens=30, output_tokens=5))

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events_1, final_msg_1),
            _AsyncStreamCtx(events_2, final_msg_2),
        ])

        tool = LlmTool(
            name="calc", description="Calculate", func=lambda expr: "4",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        tb = LlmToolbox(name="math", description="d", tools=[tool])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="2+2")]):
            chunks.append(chunk)

        pending_chunks = [c for c in chunks if c.tool_call and c.tool_call.status == ToolCallStatus.PENDING]
        completed_chunks = [c for c in chunks if c.tool_call and c.tool_call.status == ToolCallStatus.COMPLETED]
        assert len(pending_chunks) == 1
        assert pending_chunks[0].tool_call.name == "calc"
        assert len(completed_chunks) == 1
        assert completed_chunks[0].tool_call.result == "4"

        text_chunks = [c for c in chunks if c.content and not c.tool_call]
        assert any("4" in c.content for c in text_chunks)

        final = chunks[-1]
        assert final.is_final is True
        assert final.response_metadata["total_input_tokens"] == 50
        assert final.response_metadata["total_output_tokens"] == 15

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_tool_json_decode_error(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        events_1 = [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="toolu_bad", name="calc")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json="{broken json")),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(stop_reason="tool_use", content=[_make_tool_use_block("calc", {}, "toolu_bad")], usage=_make_usage())
        final_msg_2 = _make_response("ok", stop_reason="end_turn", usage=_make_usage())

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events_1, final_msg_1),
            _AsyncStreamCtx([], final_msg_2),
        ])

        tool = LlmTool(name="calc", description="Calculate", func=lambda: "result", parameters={"type": "object", "properties": {}})
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="test")]):
            chunks.append(chunk)

        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_tool_execution_error(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        events_1 = [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="toolu_err", name="boom")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json="{}")),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(stop_reason="tool_use", content=[_make_tool_use_block("boom", {}, "toolu_err")], usage=_make_usage())
        final_msg_2 = _make_response("Sorry, error.", stop_reason="end_turn", usage=_make_usage())

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events_1, final_msg_1),
            _AsyncStreamCtx([], final_msg_2),
        ])

        def failing_func():
            raise ValueError("tool exploded")

        tool = LlmTool(name="boom", description="Boom", func=failing_func, parameters={"type": "object", "properties": {}})
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="test")]):
            chunks.append(chunk)

        error_chunks = [c for c in chunks if c.tool_call and c.tool_call.status == ToolCallStatus.ERROR]
        assert len(error_chunks) == 1
        assert error_chunks[0].tool_call.error == "Tool execution failed"

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_rich_mcp_sends_summary(self, mock_anthropic_cls, mock_async_cls):
        """When a tool returns a __rich_mcp__ envelope, only the summary goes to the LLM."""
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        events_1 = [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="toolu_rich", name="plot")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json="{}")),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(stop_reason="tool_use", content=[_make_tool_use_block("plot", {}, "toolu_rich")], usage=_make_usage())
        final_msg_2 = _make_response("Great chart!", stop_reason="end_turn", usage=_make_usage())

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events_1, final_msg_1),
            _AsyncStreamCtx([], final_msg_2),
        ])

        rich_result = json.dumps({
            "__rich_mcp__": {
                "render": {"type": "chart", "title": "Sales"},
                "llm_summary": "Chart showing Q1 sales data",
            }
        })

        tool = LlmTool(name="plot", description="Plot", func=lambda: rich_result, parameters={"type": "object", "properties": {}})
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="plot")]):
            chunks.append(chunk)

        # Verify the tool result sent to the model uses the summary, not raw JSON
        second_call_kwargs = mock_aclient.messages.stream.call_args_list[1]
        stream_kwargs = second_call_kwargs[1]
        messages = stream_kwargs["messages"]
        tool_result_msg = messages[-1]
        tool_result_content = tool_result_msg["content"][0]["content"]
        assert tool_result_content == "Chart showing Q1 sales data"

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_astream_empty_tool_input(self, mock_anthropic_cls, mock_async_cls):
        """Tool with no JSON input should default to empty dict."""
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        events_1 = [
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="toolu_empty", name="noop")),
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(stop_reason="tool_use", content=[_make_tool_use_block("noop", {}, "toolu_empty")], usage=_make_usage())
        final_msg_2 = _make_response("done", stop_reason="end_turn", usage=_make_usage())

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events_1, final_msg_1),
            _AsyncStreamCtx([], final_msg_2),
        ])

        tool = LlmTool(name="noop", description="No-op", func=lambda: "ok", parameters={"type": "object", "properties": {}})
        tb = LlmToolbox(name="tb", description="d", tools=[tool])
        client = AnthropicClient(api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="test")]):
            chunks.append(chunk)

        completed = [c for c in chunks if c.tool_call and c.tool_call.status == ToolCallStatus.COMPLETED]
        assert len(completed) == 1
        assert completed[0].tool_call.arguments == {}


class TestAStreamArgumentStreaming:
    """``input_json_delta`` events must surface to callers as live ``STREAMING``
    chunks so the frontend can show progressive feedback while the model is
    still writing the tool call (used for the document side-pane UX)."""

    @pytest.mark.asyncio
    @patch("llming_models.providers.anthropic.anthropic_client.AsyncAnthropic")
    @patch("llming_models.providers.anthropic.anthropic_client.Anthropic")
    async def test_input_json_delta_emits_streaming_chunks(self, mock_anthropic_cls, mock_async_cls):
        from llming_models.providers.anthropic.anthropic_client import AnthropicClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        # Model emits the arguments as three JSON fragments before the tool fires.
        fragments = ['{"type":"text_doc",', ' "name":"Report",', ' "data":"hello"}']
        events = [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="tool_use", id="toolu_doc", name="create_document"),
            ),
            *[
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="input_json_delta", partial_json=frag),
                )
                for frag in fragments
            ],
            SimpleNamespace(type="content_block_stop"),
        ]
        final_msg_1 = _make_response(
            stop_reason="tool_use",
            content=[_make_tool_use_block(
                "create_document",
                {"type": "text_doc", "name": "Report", "data": "hello"},
                "toolu_doc",
            )],
            usage=_make_usage(),
        )
        final_msg_2 = _make_response("done.", stop_reason="end_turn", usage=_make_usage())

        mock_aclient.messages.stream = MagicMock(side_effect=[
            _AsyncStreamCtx(events, final_msg_1),
            _AsyncStreamCtx([], final_msg_2),
        ])

        tool = LlmTool(
            name="create_document",
            description="Create a document",
            func=lambda type, name, data: '{"status":"created","document_id":"d1"}',
            parameters={"type": "object", "properties": {}},
        )
        tb = LlmToolbox(name="docs", description="d", tools=[tool])
        client = AnthropicClient(
            api_key="sk-ant-test", model="claude-sonnet-4-5-20250929", toolboxes=[tb],
        )

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="draft a report")]):
            chunks.append(chunk)

        streaming_chunks = [
            c for c in chunks
            if c.tool_call and c.tool_call.status == ToolCallStatus.STREAMING
        ]
        # One streaming chunk per input_json_delta event.
        assert len(streaming_chunks) == len(fragments)
        # Concatenating their deltas reproduces the full argument JSON.
        assembled = "".join(c.tool_call.arguments_delta or "" for c in streaming_chunks)
        assert json.loads(assembled) == {
            "type": "text_doc", "name": "Report", "data": "hello",
        }
        # Each streaming chunk carries the tool identity so the UI can group
        # deltas by call_id across overlapping tool calls.
        for c in streaming_chunks:
            assert c.tool_call.name == "create_document"
            assert c.tool_call.call_id == "toolu_doc"
            # ``arguments`` stays None during streaming — only set on finalize.
            assert c.tool_call.arguments is None

        # The finalized tool call still fires — STREAMING chunks don't replace it.
        completed = [
            c for c in chunks
            if c.tool_call and c.tool_call.status == ToolCallStatus.COMPLETED
        ]
        assert len(completed) == 1
        assert completed[0].tool_call.arguments == {
            "type": "text_doc", "name": "Report", "data": "hello",
        }
