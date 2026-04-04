"""Models for chat handling."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Role in a chat conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    FUNCTION_PENDING = "function_pending"  # Used to indicate a tool is being executed


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    id: UUID = Field(default_factory=uuid4)
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str | None = None
    function_call: dict | None = None
    images: list[str] | None = Field(default=None, description="Base64-encoded images attached to this message")
    images_stale: bool = Field(default=False, description="If True, images exceeded the history limit and will be skipped")
    content_stale: bool = Field(default=False, description="If True, message was condensed into a summary and will be skipped")


class ChatHistory(BaseModel):
    """History of chat messages."""
    messages: list[ChatMessage] = Field(default_factory=list)

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def get_messages(self) -> list[ChatMessage]:
        """Get all messages in the history."""
        return self.messages

    def get_last_message(self) -> ChatMessage | None:
        """Get the last message in the history."""
        return self.messages[-1] if self.messages else None

    def clear(self) -> None:
        """Clear all messages from the history."""
        self.messages.clear()
