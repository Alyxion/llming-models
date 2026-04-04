"""Tool call status tracking and structured tool call info."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCallStatus(str, Enum):
    """Status of a tool call."""
    PENDING = "pending"        # Tool call initiated, waiting to execute
    EXECUTING = "executing"    # Tool is currently executing
    COMPLETED = "completed"    # Tool completed successfully
    ERROR = "error"           # Tool execution failed


class ToolCallInfo(BaseModel):
    """Structured information about a tool call.

    Used by clients to emit tool events that the UI can render
    without parsing JSON strings.
    """
    name: str = Field(description="Tool/function name")
    call_id: str = Field(description="Unique identifier for this call")
    status: ToolCallStatus = Field(description="Current status of the tool call")
    arguments: dict[str, Any] | None = Field(default=None, description="Arguments passed to the tool")
    result: Any | None = Field(default=None, description="Result from tool execution")
    error: str | None = Field(default=None, description="Error message if failed")
    sources: list[dict[str, Any]] | None = Field(default=None, description="Source attribution data (e.g. flux sources)")

    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        return self.name.replace('_', ' ').title()

    @property
    def is_image_generation(self) -> bool:
        """Check if this is an image generation tool."""
        return self.name == 'generate_image'

    def to_serialized_dict(self) -> dict[str, Any]:
        """Produce the dict shape stored in IndexedDB.

        Materializes computed properties (display_name, is_image_generation)
        as concrete values and converts status enum to string.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "call_id": self.call_id,
            "display_name": self.display_name,
            "status": self.status.value,
            "is_image_generation": self.is_image_generation,
        }
        if self.arguments:
            d["arguments"] = self.arguments
        if self.result is not None:
            d["result"] = self.result
        if self.sources:
            d["sources"] = self.sources
        if self.error:
            d["error"] = self.error
        return d
