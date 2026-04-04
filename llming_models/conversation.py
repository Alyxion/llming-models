"""Conversation persistence models.

Defines the canonical conversation format stored in IndexedDB (client-side)
and used for import/export between apps.  Wire-compatible with llming-lodge.

A simple chat app built on llming-models can produce, store, and load
conversations in the same format — rich content blocks that the app does not
support remain as inert fenced code blocks in the message content string.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field

from .content_blocks import encode_content_block
from .llm_base_models import ChatMessage, Role
from .tools.tool_call import ToolCallInfo

if TYPE_CHECKING:
    from .session import ChatSession


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------

class AvatarOverride(BaseModel):
    """Custom avatar for MCP-provided responses."""
    icon: str
    label: str


class FileRef(BaseModel):
    """Reference to a content-addressable file in IDB."""
    hash: str
    name: str
    mime_type: str
    size: int
    type: str  # "document" | "image"


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

def serialize_message(
    msg: ChatMessage,
    *,
    tool_calls: list[ToolCallInfo | dict[str, Any]] | None = None,
    avatar_override: AvatarOverride | dict[str, str] | None = None,
    content_blocks: list[tuple[str, dict[str, Any] | list[Any]]] | None = None,
) -> dict[str, Any]:
    """Serialize a ChatMessage to the canonical dict stored in IndexedDB.

    Parameters
    ----------
    msg:
        The message to serialize.
    tool_calls:
        Optional tool calls to attach (assistant messages only).
        Accepts ``ToolCallInfo`` objects (calls ``to_serialized_dict()``)
        or pre-built dicts.
    avatar_override:
        Optional custom avatar ``{icon, label}``.
    content_blocks:
        Optional list of ``(block_type, data)`` tuples to inline as fenced
        code blocks in the content string.
    """
    data = msg.model_dump(mode="json", exclude_none=True)

    # Inline content blocks into the content string
    if content_blocks:
        for block_type, block_data in content_blocks:
            data["content"] += encode_content_block(block_type, block_data)

    # Attach tool calls
    if tool_calls:
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, ToolCallInfo):
                serialized.append(tc.to_serialized_dict())
            else:
                serialized.append(tc)
        data["tool_calls"] = serialized

    # Attach avatar override
    if avatar_override is not None:
        if isinstance(avatar_override, AvatarOverride):
            data["avatar_override"] = avatar_override.model_dump()
        else:
            data["avatar_override"] = avatar_override

    return data


# ---------------------------------------------------------------------------
# Conversation metadata (lightweight, for sidebar listing)
# ---------------------------------------------------------------------------

class ConversationMeta(BaseModel):
    """Lightweight metadata matching the IDB ``conv_meta`` store."""
    id: str
    title: str
    created_at: str  # ISO 8601
    updated_at: str  # ISO 8601
    message_count: int = 0
    first_user_snippet: str = ""
    project_id: str | None = None
    nudge_id: str | None = None
    favorited: bool | None = None


# ---------------------------------------------------------------------------
# Full conversation
# ---------------------------------------------------------------------------

class Conversation(BaseModel):
    """Full conversation as stored in IndexedDB.

    Wire-compatible with llming-lodge's ``conversations`` IDB store.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = ""
    created_at: str = ""  # ISO 8601
    updated_at: str = ""  # ISO 8601
    model: str = ""
    auto_underlying_model: str | None = None
    model_defaults_version: int = 1
    provider: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    condensed_summary: str | None = None
    base_system_prompt: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    enabled_tools: list[str] = Field(default_factory=list)
    project_id: str | None = None
    nudge_id: str | None = None
    file_refs: list[FileRef] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)

    def to_meta(self) -> ConversationMeta:
        """Extract lightweight metadata for sidebar listing."""
        first_snippet = ""
        count = 0
        for m in self.messages:
            if m.get("content_stale"):
                continue
            count += 1
            if not first_snippet and m.get("role") == "user":
                first_snippet = (m.get("content") or "")[:60]

        return ConversationMeta(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            message_count=count,
            first_user_snippet=first_snippet,
            project_id=self.project_id,
            nudge_id=self.nudge_id,
        )

    @classmethod
    def from_session(
        cls,
        session: ChatSession,
        *,
        session_id: str | None = None,
        title: str | None = None,
        auto_underlying_model: str | None = None,
        model_defaults_version: int = 1,
        base_system_prompt: str | None = None,
        enabled_tools: list[str] | None = None,
        project_id: str | None = None,
        nudge_id: str | None = None,
    ) -> Conversation:
        """Build a Conversation from a live ChatSession.

        This serializes the session's history and config into the canonical
        IndexedDB-compatible format.
        """
        messages_raw = session.history.get_messages()

        # Serialize each message
        serialized_messages: list[dict[str, Any]] = []
        for msg in messages_raw:
            serialized_messages.append(
                msg.model_dump(mode="json", exclude_none=True)
            )

        # Auto-generate title from first user message if not provided
        auto_title = title or ""
        if not auto_title:
            for msg in messages_raw:
                if msg.role == Role.USER and msg.content:
                    snippet = msg.content[:30]
                    if len(msg.content) > 30:
                        snippet += "\u2026"
                    auto_title = snippet
                    break

        # Timestamps
        now = datetime.now(timezone.utc).isoformat()
        created = now
        if messages_raw:
            created = messages_raw[0].timestamp.isoformat()

        return cls(
            id=session_id or str(uuid4()),
            title=auto_title,
            created_at=created,
            updated_at=now,
            model=session.config.model,
            auto_underlying_model=auto_underlying_model,
            model_defaults_version=model_defaults_version,
            provider=session.config.provider,
            messages=serialized_messages,
            condensed_summary=session._condensed_summary,
            base_system_prompt=base_system_prompt or session._system_prompt or "",
            config=session.config.model_dump(mode="json"),
            enabled_tools=enabled_tools or [],
            project_id=project_id,
            nudge_id=nudge_id,
        )
