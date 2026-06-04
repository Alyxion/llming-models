"""Configuration for connecting to MCP servers."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """Configuration for connecting to an MCP server."""
    # Stdio mode
    command: str | None = Field(default=None, description="Command to execute for stdio transport")
    args: list[str] | None = Field(default=None, description="Arguments for the command")
    env: dict[str, str] | None = Field(default=None, description="Environment variables to set")
    cwd: str | None = Field(default=None, description="Working directory for the command")

    # HTTP mode
    url: str | None = Field(default=None, description="URL for HTTP/SSE transport")
    api_key: str | None = Field(default=None, description="API key for HTTP authentication")
    headers: dict[str, str] | None = Field(default=None, description="Additional HTTP headers")

    # In-process mode
    server_instance: Any | None = Field(default=None, exclude=True, description="In-process MCP server instance")

    # UI / toggle metadata
    label: str | None = Field(default=None, description="Display name (e.g. 'Stock Agent')")
    description: str | None = Field(default=None, description="Short description for UI")
    category: str | None = Field(default=None, description="Grouping category (e.g. 'Experimental')")
    enabled_by_default: bool = Field(default=False, description="If True, tools are enabled on discovery; otherwise opt-in")
    default_enabled_tools: list[str] | None = Field(default=None, description="When set, only these tools are enabled by default (per-tool control). Requires enabled_by_default=True.")
    exclude_providers: list[str] | None = Field(default=None, description="Providers this MCP does NOT support. None = all providers.")
    requires_providers: list[str] | None = Field(default=None, description="If set, MCP tools only work with these providers (respects PROVIDER_COMPAT). None = all providers.")
    collapse_tools: bool = Field(default=False, description="If True, all tools are shown as a single toggle in the UI instead of individual entries.")
    flyout: bool = Field(default=False, description="If True, tools get their own top-level flyout in the plus menu")
    hidden: bool = Field(default=False, description="If True, the MCP server and all its tools are omitted from the UI tool list. Tools still execute when enabled (e.g. because a nudge auto-activates them); useful for nudge-bound MCPs that should be invisible to the user.")
    avatar: str | None = Field(default=None, description="Custom avatar icon path (relative to staticBase, e.g. 'models/lisa-avatar.gif'). When set, replaces the model icon in chat message headers when this MCP's tools are used.")
    auto_activate_keywords: list[str] | None = Field(default=None, description="Keywords that trigger auto-activation of this MCP. When a user message matches any keyword (case-insensitive), the MCP tools are enabled for the session.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def is_stdio(self) -> bool:
        """Check if this is a stdio-based connection."""
        return self.command is not None

    def is_http(self) -> bool:
        """Check if this is an HTTP-based connection."""
        return self.url is not None

    def is_inprocess(self) -> bool:
        """Check if this is an in-process server."""
        return self.server_instance is not None
