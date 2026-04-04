"""Unit tests for the OpenAI-compatible Chat Completions client (fully mocked SDK)."""
from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmSystemMessage,
)
from llming_models.llm_base_models import Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_usage(prompt_tokens: int = 10, completion_tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _make_choice(content: str = "Hello!") -> SimpleNamespace:
    return SimpleNamespace(message=SimpleNamespace(content=content))


def _make_response(content: str = "Hello!", usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[_make_choice(content)],
        usage=usage or _make_usage(),
    )


def _make_stream_chunk(content: str | None, usage: Any = None, has_choices: bool = True) -> SimpleNamespace:
    """Build a single chunk for streaming tests."""
    if has_choices:
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice], usage=usage)
    else:
        return SimpleNamespace(choices=[], usage=usage)


# ---------------------------------------------------------------------------
# _convert_messages tests
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def test_role_mapping(self):
        from llming_models.providers.openai_compat_client import _convert_messages

        msgs = [
            LlmSystemMessage(content="System prompt"),
            LlmHumanMessage(content="User message"),
            LlmAIMessage(content="AI response"),
        ]
        converted = _convert_messages(msgs)
        assert converted == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "AI response"},
        ]

    def test_single_message(self):
        from llming_models.providers.openai_compat_client import _convert_messages

        msgs = [LlmHumanMessage(content="Hi")]
        converted = _convert_messages(msgs)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_empty_messages(self):
        from llming_models.providers.openai_compat_client import _convert_messages

        converted = _convert_messages([])
        assert converted == []


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestCompatInit:
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_default_init(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        assert client.model == "deepseek-chat"
        assert client.temperature == 0.7
        assert client.max_tokens is None
        assert client.streaming is False
        mock_openai.assert_called_once_with(api_key="sk-test", base_url=None)
        mock_async_openai.assert_called_once_with(api_key="sk-test", base_url=None)

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_custom_base_url(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="sk-test", model="mistral-large",
            base_url="https://api.mistral.ai/v1",
        )
        mock_openai.assert_called_once_with(api_key="sk-test", base_url="https://api.mistral.ai/v1")
        mock_async_openai.assert_called_once_with(api_key="sk-test", base_url="https://api.mistral.ai/v1")

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_all_parameters(self, mock_openai, mock_async_openai):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="sk-test", model="together-llama",
            temperature=0.3, max_tokens=1024, streaming=True,
            base_url="https://api.together.xyz",
        )
        assert client.model == "together-llama"
        assert client.temperature == 0.3
        assert client.max_tokens == 1024
        assert client.streaming is True


# ---------------------------------------------------------------------------
# _build_kwargs tests
# ---------------------------------------------------------------------------

class TestBuildKwargs:
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_basic_kwargs(self, mock_openai, mock_async):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="sk-test", model="deepseek-chat", temperature=0.5,
        )
        msgs = [LlmHumanMessage(content="Hi")]
        kwargs = client._build_kwargs(msgs)

        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["temperature"] == 0.5
        assert "max_tokens" not in kwargs
        assert "stream" not in kwargs

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_kwargs_with_max_tokens(self, mock_openai, mock_async):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            api_key="sk-test", model="deepseek-chat", max_tokens=2048,
        )
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")])
        assert kwargs["max_tokens"] == 2048

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_kwargs_with_stream(self, mock_openai, mock_async):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        kwargs = client._build_kwargs([LlmHumanMessage(content="Hi")], stream=True)
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_kwargs_messages_converted(self, mock_openai, mock_async):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        msgs = [
            LlmSystemMessage(content="Be helpful"),
            LlmHumanMessage(content="Hello"),
        ]
        kwargs = client._build_kwargs(msgs)
        assert kwargs["messages"] == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]


# ---------------------------------------------------------------------------
# invoke() tests
# ---------------------------------------------------------------------------

class TestInvoke:
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_invoke_text_response(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_response("Answer!")

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = client.invoke([LlmHumanMessage(content="Question")])

        assert isinstance(result, LlmAIMessage)
        assert result.content == "Answer!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_invoke_usage_metadata(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        usage = _make_usage(prompt_tokens=50, completion_tokens=100)
        mock_client.chat.completions.create.return_value = _make_response("ok", usage=usage)

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = client.invoke([LlmHumanMessage(content="test")])
        assert result.response_metadata["input_tokens"] == 50
        assert result.response_metadata["output_tokens"] == 100

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_invoke_no_usage(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = SimpleNamespace(
            choices=[_make_choice("text")],
            usage=None,
        )
        mock_client.chat.completions.create.return_value = resp

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = client.invoke([LlmHumanMessage(content="test")])
        assert result.response_metadata == {}

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_invoke_none_content(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=_make_usage(),
        )
        mock_client.chat.completions.create.return_value = resp

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = client.invoke([LlmHumanMessage(content="test")])
        assert result.content == ""

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_invoke_propagates_error(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        with pytest.raises(RuntimeError, match="API error"):
            client.invoke([LlmHumanMessage(content="test")])


# ---------------------------------------------------------------------------
# ainvoke() tests
# ---------------------------------------------------------------------------

class TestAInvoke:
    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_ainvoke_text_response(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        mock_aclient.chat.completions.create.return_value = _make_response("Async answer!")

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = await client.ainvoke([LlmHumanMessage(content="Question")])

        assert result.content == "Async answer!"
        mock_aclient.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_ainvoke_usage_metadata(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        usage = _make_usage(prompt_tokens=75, completion_tokens=200)
        mock_aclient.chat.completions.create.return_value = _make_response("ok", usage=usage)

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = await client.ainvoke([LlmHumanMessage(content="test")])
        assert result.response_metadata["input_tokens"] == 75
        assert result.response_metadata["output_tokens"] == 200

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_ainvoke_no_usage(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        resp = SimpleNamespace(choices=[_make_choice("text")], usage=None)
        mock_aclient.chat.completions.create.return_value = resp

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = await client.ainvoke([LlmHumanMessage(content="test")])
        assert result.response_metadata == {}

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_ainvoke_none_content(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=_make_usage(),
        )
        mock_aclient.chat.completions.create.return_value = resp

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        result = await client.ainvoke([LlmHumanMessage(content="test")])
        assert result.content == ""


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------

class TestStream:
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_stream_text_chunks(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunks_data = [
            _make_stream_chunk("Hello "),
            _make_stream_chunk("world"),
            _make_stream_chunk(None),  # no content
        ]
        mock_client.chat.completions.create.return_value = iter(chunks_data)

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))

        # 2 text chunks + 1 final
        assert len(chunks) == 3
        assert chunks[0].content == "Hello "
        assert chunks[0].is_final is False
        assert chunks[1].content == "world"
        assert chunks[1].is_final is False
        assert chunks[2].content == ""
        assert chunks[2].is_final is True

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_stream_empty_choices_skipped(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunks_data = [
            _make_stream_chunk(None, has_choices=False),  # no choices
            _make_stream_chunk("text"),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks_data)

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))

        # 1 text chunk + 1 final
        assert len(chunks) == 2
        assert chunks[0].content == "text"

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_stream_calls_with_stream_options(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([])

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        list(client.stream([LlmHumanMessage(content="Hi")]))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"] == {"include_usage": True}

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_stream_chunk_indices(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        chunks_data = [
            _make_stream_chunk("a"),
            _make_stream_chunk("b"),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks_data)

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))

        # Indices should be sequential: 0, 1, 2 (final)
        assert chunks[0].index == 0
        assert chunks[1].index == 1
        assert chunks[2].index == 2  # final

    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    def test_stream_all_roles_assistant(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = iter([_make_stream_chunk("text")])

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))
        for chunk in chunks:
            assert chunk.role == Role.ASSISTANT


# ---------------------------------------------------------------------------
# astream() tests
# ---------------------------------------------------------------------------

class TestAStream:
    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_text_and_usage(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        # Chunks: text, text, final with usage
        chunks_data = [
            _make_stream_chunk("Hello "),
            _make_stream_chunk("world"),
            _make_stream_chunk(None, usage=_make_usage(prompt_tokens=30, completion_tokens=40), has_choices=False),
        ]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(chunk)

        # 2 text + 1 final
        assert len(chunks) == 3
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world"
        final = chunks[-1]
        assert final.is_final is True
        assert final.response_metadata["total_input_tokens"] == 30
        assert final.response_metadata["total_output_tokens"] == 40

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_usage_callback_called(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        chunks_data = [
            _make_stream_chunk("ok"),
            _make_stream_chunk(None, usage=_make_usage(prompt_tokens=10, completion_tokens=5), has_choices=False),
        ]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        usage_cb = MagicMock()

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        usage_cb.assert_called_once_with(10, 5)

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_usage_callback_error_handled(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        chunks_data = [
            _make_stream_chunk("text"),
            _make_stream_chunk(None, usage=_make_usage(prompt_tokens=10, completion_tokens=5), has_choices=False),
        ]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        usage_cb = MagicMock(side_effect=RuntimeError("callback error"))

        # Should not raise
        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_no_usage_no_callback(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        # All chunks without usage
        chunks_data = [_make_stream_chunk("text")]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")
        usage_cb = MagicMock()

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(chunk)

        # Callback should NOT be called (no usage data)
        usage_cb.assert_not_called()
        final = chunks[-1]
        assert final.is_final is True
        assert final.response_metadata["total_input_tokens"] == 0
        assert final.response_metadata["total_output_tokens"] == 0

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_empty_delta_skipped(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        # Chunk with empty string content should be skipped
        chunks_data = [
            _make_stream_chunk(""),
            _make_stream_chunk("real content"),
        ]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(chunk)

        # Only "real content" + final
        text_chunks = [c for c in chunks if c.content and not c.is_final]
        assert len(text_chunks) == 1
        assert text_chunks[0].content == "real content"

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_none_delta_skipped(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        # Chunk with delta.content=None
        chunk_with_none = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            usage=None,
        )

        async def async_iter():
            yield chunk_with_none

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(chunk)

        # Only the final chunk
        assert len(chunks) == 1
        assert chunks[0].is_final is True

    @pytest.mark.asyncio
    @patch("llming_models.providers.openai_compat_client.AsyncOpenAI")
    @patch("llming_models.providers.openai_compat_client.OpenAI")
    async def test_astream_chunk_indices_sequential(self, mock_openai_cls, mock_async_cls):
        from llming_models.providers.openai_compat_client import OpenAICompatibleClient

        mock_aclient = AsyncMock()
        mock_async_cls.return_value = mock_aclient

        chunks_data = [
            _make_stream_chunk("a"),
            _make_stream_chunk("b"),
            _make_stream_chunk("c"),
        ]

        async def async_iter():
            for c in chunks_data:
                yield c

        mock_aclient.chat.completions.create.return_value = async_iter()

        client = OpenAICompatibleClient(api_key="sk-test", model="deepseek-chat")

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(chunk)

        # Indices should be 0, 1, 2 for text + 3 for final
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
