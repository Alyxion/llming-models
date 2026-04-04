"""Unit tests for the OpenAI Responses-API client (fully mocked SDK)."""
from __future__ import annotations

import asyncio
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
from llming_models.model_info import ReasoningEffort


# ---------------------------------------------------------------------------
# Async stream context manager helper
# ---------------------------------------------------------------------------

class _AsyncStreamCtx:
    """Non-coroutine async context manager for ``_aclient.responses.stream()``.

    The SDK's ``.stream()`` is called *without* ``await`` and returns an async
    context manager directly.  Using ``AsyncMock`` would turn the call itself
    into a coroutine, breaking ``async with``.
    """

    def __init__(self, events: list, final_response: Any = None):
        self._events = events
        self._final_response = final_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self._async_gen()

    async def _async_gen(self):
        for ev in self._events:
            yield ev

    async def get_final_response(self):
        return self._final_response


# ---------------------------------------------------------------------------
# Helpers to build realistic mock SDK objects
# ---------------------------------------------------------------------------

def _make_usage(**overrides: Any) -> MagicMock:
    defaults = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_tokens_details": None,
    }
    defaults.update(overrides)
    usage = MagicMock()
    for k, v in defaults.items():
        setattr(usage, k, v)
    usage.model_dump.return_value = {k: v for k, v in defaults.items()}
    return usage


def _make_text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _make_message_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[_make_text_part(text)])


def _make_function_call_output(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def _make_reasoning_output() -> SimpleNamespace:
    return SimpleNamespace(type="reasoning", content=[SimpleNamespace(type="thinking", text="hmm")])


def _make_response(text: str = "Hello!", usage: Any = None, output: list | None = None) -> MagicMock:
    resp = MagicMock()
    resp.output = output if output is not None else [_make_message_output(text)]
    resp.usage = usage or _make_usage()
    return resp


# ---------------------------------------------------------------------------
# _convert_messages tests
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def test_text_only(self):
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmSystemMessage(content="Be helpful"),
            LlmHumanMessage(content="Hi"),
            LlmAIMessage(content="Hello"),
        ]
        converted = _convert_messages(msgs)
        assert converted == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

    def test_user_message_with_images(self):
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="What is this?", images=["iVBORbase64data"]),
        ]
        converted = _convert_messages(msgs)
        assert len(converted) == 1
        parts = converted[0]["content"]
        assert parts[0] == {"type": "input_text", "text": "What is this?"}
        assert parts[1]["type"] == "input_image"
        assert "image/png" in parts[1]["image_url"]

    def test_user_jpeg_image(self):
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="pic", images=["/9j/fakedata"]),
        ]
        converted = _convert_messages(msgs)
        assert "image/jpeg" in converted[0]["content"][1]["image_url"]

    def test_assistant_message_with_images(self):
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmAIMessage(content="Generated this", images=["iVBORabc"]),
        ]
        converted = _convert_messages(msgs)
        # Assistant text + follow-up user message with image context
        assert len(converted) == 2
        assert converted[0] == {"role": "assistant", "content": "Generated this"}
        assert converted[1]["role"] == "user"
        assert converted[1]["content"][0]["type"] == "input_text"
        assert converted[1]["content"][1]["type"] == "input_image"

    def test_image_limit(self):
        """Only the last max_image_history messages should keep their images."""
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="old", images=["iVBORold1"]),
            LlmHumanMessage(content="old2", images=["iVBORold2"]),
            LlmHumanMessage(content="new", images=["iVBORnew"]),
        ]
        converted = _convert_messages(msgs, max_image_history=1)
        # Only the last one should be multimodal
        assert isinstance(converted[0]["content"], str)  # stripped
        assert isinstance(converted[1]["content"], str)  # stripped
        assert isinstance(converted[2]["content"], list)  # kept

    def test_unknown_image_type_defaults_to_png(self):
        from llming_models.providers.openai.openai_client import _convert_messages

        msgs = [
            LlmHumanMessage(content="x", images=["AAAA_unknown"]),
        ]
        converted = _convert_messages(msgs)
        assert "image/png" in converted[0]["content"][1]["image_url"]


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestOpenAIInit:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_openai_api_type(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", api_type="openai")
        assert client.api_type == "openai"
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()

    @patch("llming_models.providers.openai.openai_client.AsyncAzureOpenAI")
    @patch("llming_models.providers.openai.openai_client.AzureOpenAI")
    def test_azure_api_type(self, mock_azure, mock_async_azure):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key="azure-key",
            model="gpt-4o",
            api_type="azure",
            api_version="2024-06-01",
            base_url="https://myresource.openai.azure.com",
        )
        assert client.api_type == "azure"
        mock_azure.assert_called_once()
        mock_async_azure.assert_called_once()

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_missing_api_key_raises(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAILlmClient(api_key=None, model="gpt-4o")

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_reasoning_effort_stored(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key="sk-test", model="gpt-5-mini",
            reasoning_effort=ReasoningEffort.HIGH,
        )
        assert client.reasoning_effort == ReasoningEffort.HIGH


# ---------------------------------------------------------------------------
# _get_reasoning_kwargs / _skip_temperature / _is_reasoning_model tests
# ---------------------------------------------------------------------------

class TestReasoningHelpers:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_no_reasoning_effort(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        assert client._get_reasoning_kwargs() == {}

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_reasoning_effort_high(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key="sk-test", model="gpt-5-mini",
            reasoning_effort=ReasoningEffort.HIGH,
        )
        assert client._get_reasoning_kwargs() == {"reasoning": {"effort": "high"}}

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_reasoning_effort_none_maps_to_minimal(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(
            api_key="sk-test", model="gpt-5-mini",
            reasoning_effort=ReasoningEffort.NONE,
        )
        # No tools, so minimal stays minimal
        assert client._get_reasoning_kwargs() == {"reasoning": {"effort": "minimal"}}

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_reasoning_effort_none_with_tools_maps_to_low(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        tool = LlmTool(name="test", description="t", func=lambda: None, parameters={"type": "object", "properties": {}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(
            api_key="sk-test", model="gpt-5-mini",
            reasoning_effort=ReasoningEffort.NONE,
            toolboxes=[toolbox],
        )
        assert client._get_reasoning_kwargs() == {"reasoning": {"effort": "low"}}

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_is_reasoning_model(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        for model_name in ("gpt-5-mini", "o1-preview", "o3-mini", "o4-mini"):
            client = OpenAILlmClient(api_key="sk-test", model=model_name)
            assert client._is_reasoning_model() is True

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        assert client._is_reasoning_model() is False

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_skip_temperature_with_web_search(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        toolbox = LlmToolbox(name="tb", description="d", tools=["web_search"])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        assert client._skip_temperature() is True

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_skip_temperature_reasoning_model(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")
        assert client._skip_temperature() is True


# ---------------------------------------------------------------------------
# _enforce_strict_schema tests
# ---------------------------------------------------------------------------

class TestEnforceStrictSchema:
    def test_object_gets_required_and_additional_properties(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"a", "b"}

    def test_array_items_enforced(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"type": "array"}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert result["items"] == {"type": "string"}

    def test_default_moved_to_description(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"type": "string", "description": "Color", "default": "red"}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert "default" not in result
        assert '"red"' in result["description"]

    def test_default_bool_formatting(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"type": "boolean", "default": True}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert "true" in result.get("description", "")

    def test_default_empty_string(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"type": "string", "default": ""}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert "empty" in result.get("description", "")

    def test_no_type_gets_string(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"description": "unknown"}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        assert result["type"] == "string"

    def test_anyof_branches_recursed(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {"anyOf": [{"type": "object", "properties": {"x": {"type": "integer"}}}]}
        result = OpenAILlmClient._enforce_strict_schema(schema)
        inner = result["anyOf"][0]
        assert inner["additionalProperties"] is False

    def test_nested_object(self):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        schema = {
            "type": "object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {"val": {"type": "number"}},
                }
            },
        }
        result = OpenAILlmClient._enforce_strict_schema(schema)
        child = result["properties"]["child"]
        assert child["additionalProperties"] is False
        assert child["required"] == ["val"]


# ---------------------------------------------------------------------------
# _build_tools_list tests
# ---------------------------------------------------------------------------

class TestBuildToolsList:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_no_toolboxes(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        assert client._build_tools_list() == []

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_string_web_search_in_toolbox(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        toolbox = LlmToolbox(name="tb", description="d", tools=["web_search"])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        tools = client._build_tools_list()
        assert tools == [{"type": "web_search"}]

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_dict_tool_passthrough(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        toolbox = LlmToolbox(name="tb", description="d", tools=[{"type": "web_search", "extra": 1}])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        tools = client._build_tools_list()
        assert tools == [{"type": "web_search", "extra": 1}]

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_function_tool_strict_mode(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        tool = LlmTool(
            name="calc", description="Calculate",
            func=lambda x: x,
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        tools = client._build_tools_list()
        assert len(tools) == 1
        assert tools[0]["strict"] is True
        assert tools[0]["name"] == "calc"
        assert tools[0]["parameters"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# invoke() tests
# ---------------------------------------------------------------------------

class TestInvoke:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_text_response(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = _make_response("Hi there!")

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        msgs = [LlmHumanMessage(content="Hello")]
        result = client.invoke(msgs)

        assert isinstance(result, LlmAIMessage)
        assert result.content == "Hi there!"
        mock_client.responses.create.assert_called_once()

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_reasoning_output_skipped(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = _make_response(output=[_make_reasoning_output(), _make_message_output("Answer")])
        mock_client.responses.create.return_value = resp

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")
        result = client.invoke([LlmHumanMessage(content="Think hard")])
        assert result.content == "Answer"

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_function_call(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        fn_output = _make_function_call_output("calc", '{"expr": "2+2"}')
        resp = _make_response(output=[fn_output])
        mock_client.responses.create.return_value = resp

        tool = LlmTool(
            name="calc", description="Calculate", func=lambda expr: 4,
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        result = client.invoke([LlmHumanMessage(content="2+2")])
        parsed = json.loads(result.content)
        assert parsed["function_call_result"] == 4
        assert parsed["function"] == "calc"

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_temperature_skipped_for_reasoning(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.return_value = _make_response("ok")

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")
        client.invoke([LlmHumanMessage(content="hi")])

        call_kwargs = mock_client.responses.create.call_args
        assert "temperature" not in call_kwargs.kwargs and "temperature" not in (call_kwargs[1] if len(call_kwargs) > 1 else {})

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_usage_metadata(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        usage = _make_usage(input_tokens=50, output_tokens=100)
        mock_client.responses.create.return_value = _make_response("ok", usage=usage)

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = client.invoke([LlmHumanMessage(content="test")])
        assert result.response_metadata["input_tokens"] == 50
        assert result.response_metadata["output_tokens"] == 100


# ---------------------------------------------------------------------------
# ainvoke() tests
# ---------------------------------------------------------------------------

class TestAInvoke:
    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_text_response(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_aclient.responses.create.return_value = _make_response("Async hi!")

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = await client.ainvoke([LlmHumanMessage(content="hello")])
        assert result.content == "Async hi!"

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_function_call(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        fn_output = _make_function_call_output("search", '{"q": "test"}')
        resp = _make_response(output=[fn_output])
        mock_aclient.responses.create.return_value = resp

        tool = LlmTool(
            name="search", description="Search", func=lambda q: f"found: {q}",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        result = await client.ainvoke([LlmHumanMessage(content="search for test")])
        parsed = json.loads(result.content)
        assert parsed["function_call_result"] == "found: test"

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_invalid_schema_reraises(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_aclient.responses.create.side_effect = Exception("invalid_function_parameters: bad schema")

        tool = LlmTool(
            name="t", description="t", func=lambda: None,
            parameters={"type": "object", "properties": {}},
        )
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        with pytest.raises(Exception, match="invalid_function_parameters"):
            await client.ainvoke([LlmHumanMessage(content="test")])

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_reasoning_output_skipped(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        resp = _make_response(output=[_make_reasoning_output(), _make_message_output("Result")])
        mock_aclient.responses.create.return_value = resp

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")
        result = await client.ainvoke([LlmHumanMessage(content="think")])
        assert result.content == "Result"


# ---------------------------------------------------------------------------
# astream() tests
# ---------------------------------------------------------------------------

class TestAStream:
    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_astream_yields_final_chunk_with_usage(self, mock_openai_cls, mock_async_cls):
        """Test astream emits a final chunk with usage from get_final_response fallback."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        # Events as SimpleNamespace won't pass isinstance() checks for SDK types,
        # so they'll be silently consumed. Usage comes from get_final_response.
        completed_usage = _make_usage(input_tokens=30, output_tokens=40)
        completed_usage.input_tokens_details = None
        final_response = SimpleNamespace(usage=completed_usage)

        mock_aclient.responses.stream = MagicMock(
            return_value=_AsyncStreamCtx([], final_response)
        )

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        usage_cb = MagicMock()

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="test")], usage_callback=usage_cb):
            chunks.append(chunk)

        assert len(chunks) >= 1
        final = chunks[-1]
        assert final.is_final is True
        # Usage should come from get_final_response fallback
        assert final.response_metadata["total_input_tokens"] == 30
        assert final.response_metadata["total_output_tokens"] == 40

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_astream_usage_callback_called(self, mock_openai_cls, mock_async_cls):
        """Test that usage callback is invoked with correct token counts."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        usage = _make_usage(input_tokens=100, output_tokens=50)
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.input_tokens_details = SimpleNamespace(cached_tokens=20)
        final_response = SimpleNamespace(usage=usage)

        mock_aclient.responses.stream = MagicMock(
            return_value=_AsyncStreamCtx([], final_response)
        )

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        usage_cb = MagicMock()

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="test")], usage_callback=usage_cb):
            chunks.append(chunk)

        final = chunks[-1]
        assert final.is_final is True

        # Usage should be tracked via get_final_response fallback
        total_input = final.response_metadata.get("total_input_tokens", 0)
        total_output = final.response_metadata.get("total_output_tokens", 0)
        assert total_input == 100
        assert total_output == 50
        # Callback should have been called
        assert usage_cb.called

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_astream_tool_execution(self, mock_openai_cls, mock_async_cls):
        """Test astream tool execution flow with function call events."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import (
            ResponseOutputItemAddedEvent,
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseFunctionCallArgumentsDoneEvent,
            ResponseCompletedEvent,
            ResponseTextDeltaEvent,
        )

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        # First iteration events: tool call
        from openai.types.responses import ResponseFunctionToolCall
        item_added = MagicMock(spec=ResponseOutputItemAddedEvent)
        item_added.__class__ = ResponseOutputItemAddedEvent
        item_added.output_index = 0
        item_added.item = ResponseFunctionToolCall.model_construct(
            type="function_call", name="calc", call_id="call_abc", arguments="",
        )

        args_delta = MagicMock(spec=ResponseFunctionCallArgumentsDeltaEvent)
        args_delta.__class__ = ResponseFunctionCallArgumentsDeltaEvent
        args_delta.output_index = 0
        args_delta.delta = '{"expr": "2+2"}'

        args_done = MagicMock(spec=ResponseFunctionCallArgumentsDoneEvent)
        args_done.__class__ = ResponseFunctionCallArgumentsDoneEvent
        args_done.output_index = 0

        completed1_usage = _make_usage(input_tokens=20, output_tokens=10)
        completed1_usage.input_tokens = 20
        completed1_usage.output_tokens = 10
        completed1_usage.input_tokens_details = None
        completed1_resp = SimpleNamespace(usage=completed1_usage)
        completed1 = MagicMock(spec=ResponseCompletedEvent)
        completed1.__class__ = ResponseCompletedEvent
        completed1.response = completed1_resp

        # Second iteration events: text + completed
        text_delta = MagicMock(spec=ResponseTextDeltaEvent)
        text_delta.__class__ = ResponseTextDeltaEvent
        text_delta.model_dump.return_value = {"delta": "The answer is 4"}

        completed2_usage = _make_usage(input_tokens=30, output_tokens=15)
        completed2_usage.input_tokens = 30
        completed2_usage.output_tokens = 15
        completed2_usage.input_tokens_details = None
        completed2_resp = SimpleNamespace(usage=completed2_usage)
        completed2 = MagicMock(spec=ResponseCompletedEvent)
        completed2.__class__ = ResponseCompletedEvent
        completed2.response = completed2_resp

        mock_aclient.responses.stream = MagicMock(side_effect=[
            _AsyncStreamCtx([item_added, args_delta, args_done, completed1], completed1_resp),
            _AsyncStreamCtx([text_delta, completed2], completed2_resp),
        ])

        tool = LlmTool(
            name="calc", description="Calculate", func=lambda expr: "4",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="2+2")]):
            chunks.append(chunk)

        # Should have: pending tool, completed tool, text, final
        tool_chunks = [c for c in chunks if c.tool_call is not None]
        assert len(tool_chunks) >= 1  # At least the pending+completed pair

        final = chunks[-1]
        assert final.is_final is True


# ---------------------------------------------------------------------------
# generate_image_sync() tests
# ---------------------------------------------------------------------------

class TestGenerateImageSync:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_generate_image_sync_gpt_image(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = SimpleNamespace(b64_json="iVBORw0KGgoAAAANSUhEU==")
        mock_client.images.generate.return_value = SimpleNamespace(data=[mock_data])

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = client.generate_image_sync("A cat", model="gpt-image-1")
        assert result == "iVBORw0KGgoAAAANSUhEU=="

        call_kwargs = mock_client.images.generate.call_args[1]
        assert "response_format" not in call_kwargs  # gpt-image-1 doesn't need it

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_generate_image_sync_dalle(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_data = SimpleNamespace(b64_json="base64data")
        mock_client.images.generate.return_value = SimpleNamespace(data=[mock_data])

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = client.generate_image_sync("A cat", model="dall-e-3")

        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"


# ---------------------------------------------------------------------------
# generate_image() async tests
# ---------------------------------------------------------------------------

class TestGenerateImageAsync:
    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_generate_image_async(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        mock_data = SimpleNamespace(b64_json="async_image_data")
        mock_aclient.images.generate.return_value = SimpleNamespace(data=[mock_data])

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = await client.generate_image("A dog")
        assert result == "async_image_data"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_propagates_api_error(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.side_effect = RuntimeError("API is down")

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        with pytest.raises(RuntimeError, match="API is down"):
            client.invoke([LlmHumanMessage(content="hi")])

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_propagates_api_error(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_aclient.responses.create.side_effect = RuntimeError("timeout")

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        with pytest.raises(RuntimeError, match="timeout"):
            await client.ainvoke([LlmHumanMessage(content="hi")])


# ---------------------------------------------------------------------------
# _has_web_search tests
# ---------------------------------------------------------------------------

class TestHasWebSearch:
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_string_toolbox_web_search(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=["web_search"])
        assert client._has_web_search() is True

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_dict_tool_web_search(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        toolbox = LlmToolbox(name="tb", description="d", tools=[{"type": "web_search"}])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])
        assert client._has_web_search() is True

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_no_web_search(self, mock_openai, mock_async):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        assert client._has_web_search() is False


# ---------------------------------------------------------------------------
# Synchronous stream() method
# ---------------------------------------------------------------------------

class TestSyncStream:
    """Tests for the synchronous stream() method.

    Uses model_construct() to create SDK event objects that pass isinstance checks
    without triggering pydantic validation.
    """

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_stream_text_chunks(self, mock_openai_cls, mock_async_cls):
        """sync stream yields text delta events."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        delta_evt = ResponseTextDeltaEvent.model_construct(
            type="response.text.delta", content_index=0, item_id="item_1",
            output_index=0, delta="Hello", logprobs=None, sequence_number=0,
        )
        completed_evt = ResponseCompletedEvent.model_construct(
            type="response.completed", response=MagicMock(output=[], usage=MagicMock()),
            sequence_number=1,
        )

        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=iter([delta_evt, completed_evt]))
        stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.responses.stream.return_value = stream_ctx

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")

        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))
        assert len(chunks) >= 2
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[-1].is_final is True

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_stream_with_function_calls(self, mock_openai_cls, mock_async_cls):
        """sync stream handles function call events."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import (
            ResponseOutputItemAddedEvent,
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseFunctionCallArgumentsDoneEvent,
            ResponseCompletedEvent,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        added_item = SimpleNamespace(type="function_call", name="my_tool", id="item_1")
        added_evt = ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=added_item, output_index=0, sequence_number=0,
        )
        delta_evt = ResponseFunctionCallArgumentsDeltaEvent.model_construct(
            type="response.function_call_arguments.delta", delta='{"x": 1}',
            item_id="item_1", output_index=0, sequence_number=1,
        )
        done_evt = ResponseFunctionCallArgumentsDoneEvent.model_construct(
            type="response.function_call_arguments.done", arguments='{"x": 1}',
            item_id="item_1", output_index=0, name="my_tool", sequence_number=2,
        )
        completed_evt = ResponseCompletedEvent.model_construct(
            type="response.completed", response=MagicMock(output=[], usage=MagicMock()),
            sequence_number=3,
        )

        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=iter([added_evt, delta_evt, done_evt, completed_evt]))
        stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.responses.stream.return_value = stream_ctx

        tool = LlmTool(name="my_tool", description="Test", func=lambda x: f"result_{x}", parameters={"type": "object", "properties": {"x": {"type": "integer"}}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        chunks = list(client.stream([LlmHumanMessage(content="call tool")]))

        tool_chunks = [c for c in chunks if c.tool_call]
        assert len(tool_chunks) == 2
        assert tool_chunks[0].tool_call.status == ToolCallStatus.PENDING
        assert tool_chunks[1].tool_call.status == ToolCallStatus.COMPLETED
        assert tool_chunks[1].tool_call.result == "result_1"

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_stream_tool_error(self, mock_openai_cls, mock_async_cls):
        """sync stream handles tool execution errors."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import (
            ResponseOutputItemAddedEvent,
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseFunctionCallArgumentsDoneEvent,
            ResponseCompletedEvent,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        added_item = SimpleNamespace(type="function_call", name="failing_tool", id="item_1")
        added_evt = ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added", item=added_item, output_index=0, sequence_number=0,
        )
        delta_evt = ResponseFunctionCallArgumentsDeltaEvent.model_construct(
            type="response.function_call_arguments.delta", delta='{}', item_id="item_1", output_index=0, sequence_number=1,
        )
        done_evt = ResponseFunctionCallArgumentsDoneEvent.model_construct(
            type="response.function_call_arguments.done", arguments='{}', item_id="item_1", output_index=0, name="failing_tool", sequence_number=2,
        )
        completed_evt = ResponseCompletedEvent.model_construct(
            type="response.completed", response=MagicMock(output=[], usage=MagicMock()), sequence_number=3,
        )

        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=iter([added_evt, delta_evt, done_evt, completed_evt]))
        stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.responses.stream.return_value = stream_ctx

        def failing_func():
            raise ValueError("boom")

        tool = LlmTool(name="failing_tool", description="Fails", func=failing_func, parameters={"type": "object", "properties": {}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        chunks = list(client.stream([LlmHumanMessage(content="call")]))
        error_chunks = [c for c in chunks if c.tool_call and c.tool_call.status == ToolCallStatus.ERROR]
        assert len(error_chunks) == 1
        assert error_chunks[0].tool_call.error is not None

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_stream_reasoning_events_skipped(self, mock_openai_cls, mock_async_cls):
        """Reasoning model events are consumed but not yielded."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import (
            ResponseTextDeltaEvent,
            ResponseCompletedEvent,
            ResponseReasoningTextDeltaEvent,
            ResponseReasoningSummaryTextDoneEvent,
            ResponseContentPartAddedEvent,
            ResponseContentPartDoneEvent,
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        reasoning_evt = ResponseReasoningTextDeltaEvent.model_construct(
            type="response.reasoning.text.delta", item_id="item_1", output_index=0,
            content_index=0, delta="thinking...", sequence_number=0,
        )
        summary_evt = ResponseReasoningSummaryTextDoneEvent.model_construct(
            type="response.reasoning_summary.text.done", item_id="item_1", output_index=0,
            summary_index=0, text="summary", sequence_number=1,
        )
        content_part_added = ResponseContentPartAddedEvent.model_construct(
            type="response.content_part.added", item_id="item_1", output_index=0,
            content_index=0, part=SimpleNamespace(type="text", text=""), sequence_number=2,
        )
        content_part_done = ResponseContentPartDoneEvent.model_construct(
            type="response.content_part.done", item_id="item_1", output_index=0,
            content_index=0, part=SimpleNamespace(type="text", text="done"), sequence_number=3,
        )
        text_evt = ResponseTextDeltaEvent.model_construct(
            type="response.text.delta", content_index=0, item_id="item_1",
            output_index=0, delta="Answer", logprobs=None, sequence_number=4,
        )
        completed_evt = ResponseCompletedEvent.model_construct(
            type="response.completed", response=MagicMock(output=[], usage=MagicMock()),
            sequence_number=5,
        )

        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=iter([
            reasoning_evt, summary_evt, content_part_added, content_part_done,
            text_evt, completed_evt
        ]))
        stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.responses.stream.return_value = stream_ctx

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")

        chunks = list(client.stream([LlmHumanMessage(content="Think about this")]))
        text_chunks = [c for c in chunks if c.content and not c.is_final]
        assert len(text_chunks) == 1
        assert text_chunks[0].content == "Answer"


# ---------------------------------------------------------------------------
# invoke / ainvoke function call handling in non-streaming
# ---------------------------------------------------------------------------

class TestNonStreamingFunctionCalls:
    """Tests for function call handling in invoke() and ainvoke()."""

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_function_call(self, mock_openai_cls, mock_async_cls):
        """invoke() executes function calls and returns result."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Response with function call output
        fn_output = _make_function_call_output("add_nums", '{"a": 1, "b": 2}', "call_1")
        resp = _make_response(output=[fn_output])
        mock_client.responses.create.return_value = resp

        tool = LlmTool(name="add_nums", description="Add", func=lambda a, b: a + b, parameters={"type": "object", "properties": {}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        result = client.invoke([LlmHumanMessage(content="add 1+2")])
        assert isinstance(result, LlmAIMessage)
        parsed = json.loads(result.content)
        assert parsed["function"] == "add_nums"
        assert parsed["function_call_result"] == 3

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_invoke_with_reasoning_output(self, mock_openai_cls, mock_async_cls):
        """invoke() skips reasoning output items."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        reasoning = _make_reasoning_output()
        message = _make_message_output("The answer is 42")
        resp = _make_response(output=[reasoning, message])
        mock_client.responses.create.return_value = resp

        client = OpenAILlmClient(api_key="sk-test", model="gpt-5-mini")
        result = client.invoke([LlmHumanMessage(content="What is 6*7?")])
        assert result.content == "The answer is 42"

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_function_call(self, mock_openai_cls, mock_async_cls):
        """ainvoke() executes function calls and returns result."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        fn_output = _make_function_call_output("multiply", '{"x": 3, "y": 4}', "call_2")
        resp = _make_response(output=[fn_output])
        mock_aclient.responses.create = AsyncMock(return_value=resp)

        tool = LlmTool(name="multiply", description="Mul", func=lambda x, y: x * y, parameters={"type": "object", "properties": {}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        result = await client.ainvoke([LlmHumanMessage(content="3*4")])
        parsed = json.loads(result.content)
        assert parsed["function"] == "multiply"
        assert parsed["function_call_result"] == 12

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_ainvoke_schema_error_reraises(self, mock_openai_cls, mock_async_cls):
        """ainvoke() logs tool schemas on schema validation errors."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_aclient.responses.create = AsyncMock(
            side_effect=RuntimeError("invalid_function_parameters: bad schema")
        )

        tool = LlmTool(name="t", description="d", func=lambda: None, parameters={"type": "object", "properties": {"x": {"type": "string"}}})
        toolbox = LlmToolbox(name="tb", description="d", tools=[tool])
        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o", toolboxes=[toolbox])

        with pytest.raises(RuntimeError, match="invalid_function_parameters"):
            await client.ainvoke([LlmHumanMessage(content="hi")])


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

class TestImageGeneration:
    """Tests for generate_image and generate_image_sync."""

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_generate_image_async(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="base64imagedata")]
        mock_aclient.images.generate = AsyncMock(return_value=mock_response)

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = await client.generate_image("A cat", size="1024x1024", quality="medium")
        assert result == "base64imagedata"
        mock_aclient.images.generate.assert_called_once()

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_generate_image_dalle_format(self, mock_openai_cls, mock_async_cls):
        """DALL-E models should include response_format=b64_json."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="dalle_img")]
        mock_aclient.images.generate = AsyncMock(return_value=mock_response)

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        await client.generate_image("A cat", model="dall-e-3")
        call_kwargs = mock_aclient.images.generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_generate_image_sync(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="sync_image_data")]
        mock_client.images.generate.return_value = mock_response

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        result = client.generate_image_sync("A dog")
        assert result == "sync_image_data"

    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    def test_generate_image_sync_dalle(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai.openai_client import OpenAILlmClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="dalle_sync")]
        mock_client.images.generate.return_value = mock_response

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")
        client.generate_image_sync("A dog", model="dall-e-2")
        call_kwargs = mock_client.images.generate.call_args[1]
        assert call_kwargs["response_format"] == "b64_json"


# ---------------------------------------------------------------------------
# astream with tool continuation and image handling
# ---------------------------------------------------------------------------

class TestAstreamToolContinuation:
    """Tests for astream() with tool execution and image data handling."""

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai.openai_client.AsyncOpenAI")
    @patch("llming_models.providers.openai.openai_client.OpenAI")
    async def test_astream_incomplete_event_captures_usage(self, mock_openai_cls, mock_async_cls):
        """ResponseIncompleteEvent should capture usage data."""
        from llming_models.providers.openai.openai_client import OpenAILlmClient
        from openai.types.responses import ResponseIncompleteEvent, ResponseTextDeltaEvent

        mock_aclient = MagicMock()
        mock_async_cls.return_value = mock_aclient

        text_evt = ResponseTextDeltaEvent.model_construct(
            type="response.text.delta",
            content_index=0, item_id="item_1", output_index=0,
            delta="Partial", logprobs=None, sequence_number=0,
        )

        # Create a mock response with usage for the incomplete event
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.input_tokens_details = None
        mock_response = MagicMock()
        mock_response.usage = mock_usage

        incomplete_evt = ResponseIncompleteEvent.model_construct(
            type="response.incomplete",
            response=mock_response, sequence_number=1,
        )

        events = [text_evt, incomplete_evt]
        stream_ctx = _AsyncStreamCtx(events)
        mock_aclient.responses.stream.return_value = stream_ctx

        client = OpenAILlmClient(api_key="sk-test", model="gpt-4o")

        usage_data = {}
        def usage_cb(inp, out, cached_input_tokens=0):
            usage_data["input"] = inp
            usage_data["output"] = out

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        # Should have captured usage from incomplete event
        assert usage_data.get("input") == 100
        assert usage_data.get("output") == 50
