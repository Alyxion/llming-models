"""Tests for conversation persistence models."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from llming_models.llm_base_models import ChatMessage, Role
from llming_models.messages import LlmAIMessage, LlmHumanMessage
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_models.conversation import (
    AvatarOverride,
    Conversation,
    ConversationMeta,
    FileRef,
    serialize_message,
)


# ---------------------------------------------------------------------------
# ToolCallInfo.to_serialized_dict
# ---------------------------------------------------------------------------

class TestToolCallSerialization:
    def test_minimal(self):
        tc = ToolCallInfo(
            name="web_search",
            call_id="call_123",
            status=ToolCallStatus.COMPLETED,
        )
        d = tc.to_serialized_dict()
        assert d["name"] == "web_search"
        assert d["call_id"] == "call_123"
        assert d["display_name"] == "Web Search"
        assert d["status"] == "completed"
        assert d["is_image_generation"] is False
        assert "arguments" not in d
        assert "result" not in d
        assert "sources" not in d
        assert "error" not in d

    def test_full(self):
        tc = ToolCallInfo(
            name="generate_image",
            call_id="call_456",
            status=ToolCallStatus.COMPLETED,
            arguments={"prompt": "a cat"},
            result="data:image/png;base64,abc",
            sources=[{"url": "https://example.com"}],
        )
        d = tc.to_serialized_dict()
        assert d["is_image_generation"] is True
        assert d["arguments"] == {"prompt": "a cat"}
        assert d["result"] == "data:image/png;base64,abc"
        assert d["sources"] == [{"url": "https://example.com"}]

    def test_error(self):
        tc = ToolCallInfo(
            name="broken_tool",
            call_id="call_789",
            status=ToolCallStatus.ERROR,
            error="timeout",
        )
        d = tc.to_serialized_dict()
        assert d["status"] == "error"
        assert d["error"] == "timeout"


# ---------------------------------------------------------------------------
# serialize_message
# ---------------------------------------------------------------------------

class TestSerializeMessage:
    def test_user_message_basic(self):
        msg = LlmHumanMessage(content="Hello")
        d = serialize_message(msg)
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "tool_calls" not in d
        assert "avatar_override" not in d

    def test_assistant_with_tool_calls(self):
        msg = LlmAIMessage(content="Let me search.")
        tc = ToolCallInfo(
            name="web_search", call_id="c1", status=ToolCallStatus.COMPLETED,
            result="found it",
        )
        d = serialize_message(msg, tool_calls=[tc])
        assert d["tool_calls"][0]["name"] == "web_search"
        assert d["tool_calls"][0]["display_name"] == "Web Search"
        assert d["tool_calls"][0]["result"] == "found it"

    def test_assistant_with_dict_tool_calls(self):
        msg = LlmAIMessage(content="Done.")
        raw_tc = {"name": "custom", "call_id": "c2", "status": "completed"}
        d = serialize_message(msg, tool_calls=[raw_tc])
        assert d["tool_calls"][0] == raw_tc

    def test_avatar_override_model(self):
        msg = LlmAIMessage(content="Hi")
        av = AvatarOverride(icon="models/lisa.gif", label="LISA")
        d = serialize_message(msg, avatar_override=av)
        assert d["avatar_override"]["icon"] == "models/lisa.gif"
        assert d["avatar_override"]["label"] == "LISA"

    def test_avatar_override_dict(self):
        msg = LlmAIMessage(content="Hi")
        d = serialize_message(msg, avatar_override={"icon": "x", "label": "Y"})
        assert d["avatar_override"] == {"icon": "x", "label": "Y"}

    def test_content_blocks_inlined(self):
        msg = LlmAIMessage(content="Here's a chart:")
        blocks = [("plotly", {"data": [{"y": [1, 2, 3]}]})]
        d = serialize_message(msg, content_blocks=blocks)
        assert "```plotly" in d["content"]
        assert '"data"' in d["content"]

    def test_exclude_none_fields(self):
        msg = LlmAIMessage(content="Test")
        d = serialize_message(msg)
        assert "images" not in d
        assert "tool_calls" not in d
        assert "avatar_override" not in d
        assert "function_call" not in d


# ---------------------------------------------------------------------------
# FileRef
# ---------------------------------------------------------------------------

class TestFileRef:
    def test_round_trip(self):
        ref = FileRef(hash="abc123", name="doc.pdf", mime_type="application/pdf", size=1024, type="document")
        d = ref.model_dump()
        restored = FileRef.model_validate(d)
        assert restored.hash == "abc123"
        assert restored.type == "document"


# ---------------------------------------------------------------------------
# ConversationMeta
# ---------------------------------------------------------------------------

class TestConversationMeta:
    def test_fields(self):
        meta = ConversationMeta(
            id="conv-1", title="Test", created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z", message_count=5,
            first_user_snippet="Hello there",
        )
        d = meta.model_dump()
        assert d["id"] == "conv-1"
        assert d["message_count"] == 5
        assert d["project_id"] is None


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class TestConversation:
    def _make_conversation(self, **kwargs):
        defaults = dict(
            id="conv-1",
            title="Test Chat",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            model="claude-3-haiku",
            provider="anthropic",
            messages=[
                {"role": "user", "content": "Hi", "timestamp": "2026-01-01T00:00:00Z"},
                {"role": "assistant", "content": "Hello!", "timestamp": "2026-01-01T00:00:01Z"},
            ],
            config={"provider": "anthropic", "model": "claude-3-haiku", "temperature": 0.7},
        )
        defaults.update(kwargs)
        return Conversation(**defaults)

    def test_round_trip(self):
        conv = self._make_conversation()
        d = conv.model_dump(mode="json")
        restored = Conversation.model_validate(d)
        assert restored.id == "conv-1"
        assert len(restored.messages) == 2
        assert restored.provider == "anthropic"

    def test_json_round_trip(self):
        conv = self._make_conversation()
        json_str = conv.model_dump_json()
        restored = Conversation.model_validate_json(json_str)
        assert restored.id == conv.id

    def test_to_meta(self):
        conv = self._make_conversation()
        meta = conv.to_meta()
        assert meta.id == "conv-1"
        assert meta.title == "Test Chat"
        assert meta.message_count == 2
        assert meta.first_user_snippet == "Hi"

    def test_to_meta_skips_stale(self):
        conv = self._make_conversation(messages=[
            {"role": "user", "content": "Old message", "content_stale": True},
            {"role": "user", "content": "New message"},
            {"role": "assistant", "content": "Reply"},
        ])
        meta = conv.to_meta()
        assert meta.message_count == 2  # stale message excluded
        assert meta.first_user_snippet == "New message"

    def test_file_refs(self):
        conv = self._make_conversation(file_refs=[
            {"hash": "abc", "name": "test.pdf", "mime_type": "application/pdf", "size": 100, "type": "document"},
        ])
        assert len(conv.file_refs) == 1
        assert conv.file_refs[0].hash == "abc"

    def test_optional_fields_default(self):
        conv = Conversation(model="test", provider="test", config={})
        assert conv.condensed_summary is None
        assert conv.project_id is None
        assert conv.nudge_id is None
        assert conv.auto_underlying_model is None
        assert conv.file_refs == []
        assert conv.documents == []
        assert conv.enabled_tools == []


class TestConversationFromSession:
    def _make_mock_session(self):
        """Build a mock ChatSession with enough structure for from_session()."""
        from llming_models.llm_base_models import ChatHistory

        session = MagicMock()
        session.config = MagicMock()
        session.config.model = "claude-3-haiku"
        session.config.provider = "anthropic"
        session.config.model_dump.return_value = {
            "provider": "anthropic", "model": "claude-3-haiku", "temperature": 0.7,
        }

        msg1 = LlmHumanMessage(content="What is 2+2?")
        msg2 = LlmAIMessage(content="4")
        history = ChatHistory()
        history.add_message(msg1)
        history.add_message(msg2)
        session.history = history

        session._condensed_summary = None
        session._system_prompt = "You are a helpful assistant."

        return session

    def test_basic(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(session, session_id="test-id")
        assert conv.id == "test-id"
        assert conv.provider == "anthropic"
        assert conv.model == "claude-3-haiku"
        assert len(conv.messages) == 2
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[1]["role"] == "assistant"

    def test_auto_title(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(session)
        assert "What is 2+2?" in conv.title

    def test_explicit_title(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(session, title="My Chat")
        assert conv.title == "My Chat"

    def test_condensed_summary(self):
        session = self._make_mock_session()
        session._condensed_summary = "User asked math questions."
        conv = Conversation.from_session(session)
        assert conv.condensed_summary == "User asked math questions."

    def test_metadata_passthrough(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(
            session,
            project_id="proj-1",
            nudge_id="nudge-1",
            enabled_tools=["web_search", "generate_image"],
            model_defaults_version=3,
        )
        assert conv.project_id == "proj-1"
        assert conv.nudge_id == "nudge-1"
        assert conv.enabled_tools == ["web_search", "generate_image"]
        assert conv.model_defaults_version == 3

    def test_system_prompt(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(session)
        assert conv.base_system_prompt == "You are a helpful assistant."

    def test_config_serialized(self):
        session = self._make_mock_session()
        conv = Conversation.from_session(session)
        assert conv.config["provider"] == "anthropic"
        assert conv.config["model"] == "claude-3-haiku"
