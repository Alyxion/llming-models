"""Unit tests for the Google Gemini client (fully mocked SDK)."""
from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.llm_base_models import Role
from llming_models.messages import (
    LlmAIMessage,
    LlmHumanMessage,
    LlmMessageChunk,
    LlmSystemMessage,
)


# ---------------------------------------------------------------------------
# _build_contents tests
# ---------------------------------------------------------------------------


class TestBuildContents:
    """Tests for _build_contents helper."""

    def test_system_message_extracted(self):
        from llming_models.providers.google.google_client import _build_contents

        msgs = [
            LlmSystemMessage(content="Be helpful"),
            LlmHumanMessage(content="Hi"),
        ]
        system_instruction, contents = _build_contents(msgs)
        assert system_instruction == "Be helpful"
        assert len(contents) == 1  # Only the human message

    def test_user_and_model_messages(self):
        from llming_models.providers.google.google_client import _build_contents

        msgs = [
            LlmHumanMessage(content="Hello"),
            LlmAIMessage(content="Hi there"),
        ]
        system_instruction, contents = _build_contents(msgs)
        assert system_instruction is None
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"

    def test_user_message_with_images(self):
        from llming_models.providers.google.google_client import _build_contents
        import base64

        # Create proper base64 strings that can be decoded
        fake_png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        fake_jpeg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 50
        fake_png_b64 = "iVBOR" + base64.b64encode(fake_png_data).decode()[5:]
        fake_jpeg_b64 = "/9j/" + base64.b64encode(fake_jpeg_data).decode()[4:]
        msgs = [
            LlmHumanMessage(content="Look at these", images=[fake_png_b64, fake_jpeg_b64]),
        ]
        _, contents = _build_contents(msgs)
        assert len(contents) == 1
        # Should have text + 2 image parts
        parts = contents[0].parts
        assert len(parts) == 3  # text + 2 images


# ---------------------------------------------------------------------------
# _extract_text_and_images tests
# ---------------------------------------------------------------------------


class TestExtractTextAndImages:
    """Tests for _extract_text_and_images helper."""

    def test_text_only(self):
        from llming_models.providers.google.google_client import _extract_text_and_images

        part = SimpleNamespace(text="Hello world", inline_data=None)
        text, images = _extract_text_and_images([part])
        assert text == "Hello world"
        assert images == []

    def test_inline_image_bytes(self):
        from llming_models.providers.google.google_client import _extract_text_and_images

        raw_data = b"fake_image_data"
        inline_data = SimpleNamespace(data=raw_data, mime_type="image/png")
        part = SimpleNamespace(text=None, inline_data=inline_data)
        text, images = _extract_text_and_images([part])
        assert text == ""
        assert len(images) == 1
        assert images[0].startswith("data:image/png;base64,")
        # Verify the base64 decodes correctly
        b64_part = images[0].split(",", 1)[1]
        assert base64.b64decode(b64_part) == raw_data

    def test_inline_image_string(self):
        from llming_models.providers.google.google_client import _extract_text_and_images

        inline_data = SimpleNamespace(data="already_base64", mime_type="image/jpeg")
        part = SimpleNamespace(text=None, inline_data=inline_data)
        text, images = _extract_text_and_images([part])
        assert len(images) == 1
        assert "image/jpeg" in images[0]
        assert "already_base64" in images[0]

    def test_mixed_text_and_images(self):
        from llming_models.providers.google.google_client import _extract_text_and_images

        parts = [
            SimpleNamespace(text="Here is the image:", inline_data=None),
            SimpleNamespace(text=None, inline_data=SimpleNamespace(data=b"\x89PNG", mime_type="image/png")),
            SimpleNamespace(text=" Done.", inline_data=None),
        ]
        text, images = _extract_text_and_images(parts)
        assert text == "Here is the image: Done."
        assert len(images) == 1


# ---------------------------------------------------------------------------
# GoogleClient init
# ---------------------------------------------------------------------------


class TestGoogleClientInit:
    """Tests for GoogleClient initialization."""

    @patch("llming_models.providers.google.google_client.genai")
    def test_init(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        client = GoogleClient(api_key="test-key", model="gemini-2.0-flash", temperature=0.5, max_tokens=1024)
        assert client.model == "gemini-2.0-flash"
        assert client.temperature == 0.5
        assert client.max_tokens == 1024
        mock_genai.Client.assert_called_once_with(api_key="test-key")


# ---------------------------------------------------------------------------
# GoogleClient invoke (sync)
# ---------------------------------------------------------------------------


class TestGoogleClientInvoke:
    """Tests for GoogleClient.invoke (synchronous)."""

    @patch("llming_models.providers.google.google_client.genai")
    def test_invoke_text_response(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        # Build a realistic response
        text_part = SimpleNamespace(text="Hello!", inline_data=None)
        content = SimpleNamespace(parts=[text_part])
        candidate = SimpleNamespace(content=content)
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        response = SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        mock_client_inst.models.generate_content.return_value = response

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        result = client.invoke([LlmHumanMessage(content="Hi")])

        assert isinstance(result, LlmAIMessage)
        assert result.content == "Hello!"
        assert result.response_metadata["input_tokens"] == 10
        assert result.response_metadata["output_tokens"] == 5

    @patch("llming_models.providers.google.google_client.genai")
    def test_invoke_with_images(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        # Response with inline image
        img_data = SimpleNamespace(data=b"img", mime_type="image/png")
        img_part = SimpleNamespace(text=None, inline_data=img_data)
        text_part = SimpleNamespace(text="Generated!", inline_data=None)
        content = SimpleNamespace(parts=[text_part, img_part])
        candidate = SimpleNamespace(content=content)
        response = SimpleNamespace(candidates=[candidate], usage_metadata=None)

        mock_client_inst.models.generate_content.return_value = response

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        result = client.invoke([LlmHumanMessage(content="Generate")])

        assert result.content == "Generated!"
        assert result.images is not None
        assert len(result.images) == 1

    @patch("llming_models.providers.google.google_client.genai")
    def test_invoke_empty_candidates(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        response = SimpleNamespace(candidates=[], usage_metadata=None)
        mock_client_inst.models.generate_content.return_value = response

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        result = client.invoke([LlmHumanMessage(content="Hi")])
        assert result.content == ""
        assert result.images is None


# ---------------------------------------------------------------------------
# GoogleClient ainvoke (async)
# ---------------------------------------------------------------------------


class TestGoogleClientAinvoke:
    """Tests for GoogleClient.ainvoke (asynchronous)."""

    @pytest.mark.asyncio
    @patch("llming_models.providers.google.google_client.genai")
    async def test_ainvoke_text_response(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        text_part = SimpleNamespace(text="Async hello!", inline_data=None)
        content = SimpleNamespace(parts=[text_part])
        candidate = SimpleNamespace(content=content)
        usage = SimpleNamespace(prompt_token_count=15, candidates_token_count=8)
        response = SimpleNamespace(candidates=[candidate], usage_metadata=usage)

        mock_client_inst.aio.models.generate_content = AsyncMock(return_value=response)

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        result = await client.ainvoke([LlmHumanMessage(content="Hi")])

        assert result.content == "Async hello!"
        assert result.response_metadata["input_tokens"] == 15

    @pytest.mark.asyncio
    @patch("llming_models.providers.google.google_client.genai")
    async def test_ainvoke_empty_candidates(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        response = SimpleNamespace(candidates=[], usage_metadata=None)
        mock_client_inst.aio.models.generate_content = AsyncMock(return_value=response)

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        result = await client.ainvoke([LlmHumanMessage(content="Hi")])
        assert result.content == ""


# ---------------------------------------------------------------------------
# GoogleClient stream (sync)
# ---------------------------------------------------------------------------


class TestGoogleClientStream:
    """Tests for GoogleClient.stream (synchronous streaming)."""

    @patch("llming_models.providers.google.google_client.genai")
    def test_stream_text_chunks(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        # Build chunk responses
        chunk1_part = SimpleNamespace(text="Hello", inline_data=None)
        chunk1_content = SimpleNamespace(parts=[chunk1_part])
        chunk1_candidate = SimpleNamespace(content=chunk1_content)
        chunk1 = SimpleNamespace(candidates=[chunk1_candidate], usage_metadata=None)

        chunk2_part = SimpleNamespace(text=" world", inline_data=None)
        chunk2_content = SimpleNamespace(parts=[chunk2_part])
        chunk2_candidate = SimpleNamespace(content=chunk2_content)
        chunk2 = SimpleNamespace(candidates=[chunk2_candidate], usage_metadata=None)

        mock_client_inst.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        chunks = list(client.stream([LlmHumanMessage(content="Hi")]))

        # Should have 2 text chunks + 1 final
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].content == " world"
        assert chunks[2].content == ""
        assert chunks[2].is_final is True

    @patch("llming_models.providers.google.google_client.genai")
    def test_stream_with_images(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        text_part = SimpleNamespace(text="Image:", inline_data=None)
        img_part = SimpleNamespace(text=None, inline_data=SimpleNamespace(data=b"img", mime_type="image/png"))
        content = SimpleNamespace(parts=[text_part, img_part])
        candidate = SimpleNamespace(content=content)
        chunk = SimpleNamespace(candidates=[candidate], usage_metadata=None)

        mock_client_inst.models.generate_content_stream.return_value = iter([chunk])

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")
        chunks = list(client.stream([LlmHumanMessage(content="Draw")]))

        assert len(chunks) == 2  # content + final
        assert chunks[0].images is not None


# ---------------------------------------------------------------------------
# GoogleClient astream (async streaming)
# ---------------------------------------------------------------------------


class TestGoogleClientAstream:
    """Tests for GoogleClient.astream (asynchronous streaming)."""

    @pytest.mark.asyncio
    @patch("llming_models.providers.google.google_client.genai")
    async def test_astream_text_chunks(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        chunk1_part = SimpleNamespace(text="Hi", inline_data=None)
        chunk1_content = SimpleNamespace(parts=[chunk1_part])
        chunk1_candidate = SimpleNamespace(content=chunk1_content)
        usage1 = SimpleNamespace(prompt_token_count=10, candidates_token_count=2)
        chunk1 = SimpleNamespace(candidates=[chunk1_candidate], usage_metadata=usage1)

        chunk2_part = SimpleNamespace(text=" there", inline_data=None)
        chunk2_content = SimpleNamespace(parts=[chunk2_part])
        chunk2_candidate = SimpleNamespace(content=chunk2_content)
        usage2 = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        chunk2 = SimpleNamespace(candidates=[chunk2_candidate], usage_metadata=usage2)

        async def fake_stream(*args, **kwargs):
            for c in [chunk1, chunk2]:
                yield c

        mock_client_inst.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")

        chunks = []
        async for chunk in client.astream([LlmHumanMessage(content="Hello")]):
            chunks.append(chunk)

        # 2 text chunks + 1 final
        assert len(chunks) == 3
        assert chunks[0].content == "Hi"
        assert chunks[1].content == " there"
        assert chunks[2].is_final is True
        # Final should have usage metadata
        assert chunks[2].response_metadata["total_input_tokens"] == 10
        assert chunks[2].response_metadata["total_output_tokens"] == 5

    @pytest.mark.asyncio
    @patch("llming_models.providers.google.google_client.genai")
    async def test_astream_usage_callback(self, mock_genai):
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        chunk_part = SimpleNamespace(text="Ok", inline_data=None)
        chunk_content = SimpleNamespace(parts=[chunk_part])
        chunk_candidate = SimpleNamespace(content=chunk_content)
        usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=10)
        chunk = SimpleNamespace(candidates=[chunk_candidate], usage_metadata=usage)

        async def fake_stream(*args, **kwargs):
            yield chunk

        mock_client_inst.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")

        usage_data = {}
        def usage_cb(inp, out):
            usage_data["input"] = inp
            usage_data["output"] = out

        chunks = []
        async for c in client.astream([LlmHumanMessage(content="Hi")], usage_callback=usage_cb):
            chunks.append(c)

        assert usage_data["input"] == 20
        assert usage_data["output"] == 10

    @pytest.mark.asyncio
    @patch("llming_models.providers.google.google_client.genai")
    async def test_astream_no_usage_no_callback(self, mock_genai):
        """When there's no usage and no callback, astream still works."""
        from llming_models.providers.google.google_client import GoogleClient

        mock_client_inst = MagicMock()
        mock_genai.Client.return_value = mock_client_inst

        chunk_part = SimpleNamespace(text="Hello", inline_data=None)
        chunk_content = SimpleNamespace(parts=[chunk_part])
        chunk_candidate = SimpleNamespace(content=chunk_content)
        chunk = SimpleNamespace(candidates=[chunk_candidate], usage_metadata=None)

        async def fake_stream(*args, **kwargs):
            yield chunk

        mock_client_inst.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

        client = GoogleClient(api_key="k", model="gemini-2.0-flash")

        chunks = []
        async for c in client.astream([LlmHumanMessage(content="Hi")]):
            chunks.append(c)

        assert len(chunks) == 2  # 1 text + 1 final
        # No usage callback should not crash
