"""Central registry for LLM tools.

This module provides a ToolRegistry that manages tool registration,
discovery, and execution across different sources (builtin, MCP, provider-native).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .tool_definition import (
    PROVIDER_COMPAT,
    ToolDefinition,
    ToolSource,
    ToolUIMetadata,
    MCPServerConfig,
    DEFAULT_WEB_SEARCH_TOOL,
    OPENAI_WEB_SEARCH_TOOL,
    ANTHROPIC_WEB_SEARCH_TOOL,
    DEFAULT_IMAGE_GENERATION_TOOL,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for managing LLM tools.

    Provides methods for:
    - Registering tools from various sources
    - Loading tools from JSON configuration
    - Querying available tools
    - Executing tools (delegating to appropriate handler)
    """

    def __init__(self, auto_register_defaults: bool = True):
        """Initialize the tool registry.

        Args:
            auto_register_defaults: If True, register default tools (web_search, generate_image)
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._mcp_connections: dict[str, Any] = {}  # MCPConnection instances
        self._event_loop: Any = None  # Event loop for MCP async operations

        if auto_register_defaults:
            self._register_default_tools()

    def set_event_loop(self, loop) -> None:
        """Set the event loop for async MCP operations."""
        self._event_loop = loop

    def get_event_loop(self) -> Any:
        """Get the event loop for async MCP operations."""
        return self._event_loop

    def _register_default_tools(self) -> None:
        """Register the default built-in tools."""
        # Web search - register provider-specific versions
        # Both registered under same name but with different requires_provider
        # The adapter will pick the right one based on provider
        self._tools["web_search:openai"] = OPENAI_WEB_SEARCH_TOOL
        self._tools["web_search:anthropic"] = ANTHROPIC_WEB_SEARCH_TOOL
        # Also register default for backward compat
        self.register(DEFAULT_WEB_SEARCH_TOOL)

        # Image generation - provider-specific versions
        # OpenAI uses DALL-E, Google uses Imagen 3
        self.register(DEFAULT_IMAGE_GENERATION_TOOL)  # DALL-E for OpenAI

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        Args:
            tool: The tool definition to register
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name} (source={tool.source.value})")

    def register_builtin(
        self,
        name: str,
        description: str,
        callback: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
        ui: ToolUIMetadata | None = None,
        fixed_cost_usd: float | None = None,
        requires_provider: str | None = None,
        realtime_enabled: bool = False,
    ) -> ToolDefinition:
        """Register a built-in Python tool.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            callback: Python function to execute
            parameters: JSON Schema for tool parameters
            ui: UI display metadata
            fixed_cost_usd: Fixed cost per invocation
            requires_provider: If set, only works with this provider
            realtime_enabled: If True, tool is available in live voice (Realtime API) sessions

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.BUILTIN,
            callback=callback,
            ui=ui,
            fixed_cost_usd=fixed_cost_usd,
            requires_provider=requires_provider,
            realtime_enabled=realtime_enabled,
        )
        self.register(tool)
        return tool

    def register_provider_native(
        self,
        name: str,
        description: str,
        provider_config: dict[str, Any],
        ui: ToolUIMetadata | None = None,
        requires_provider: str | None = None,
    ) -> ToolDefinition:
        """Register a provider-native tool (e.g., OpenAI's web_search).

        Args:
            name: Unique tool identifier
            description: Human-readable description
            provider_config: Provider-specific configuration
            ui: UI display metadata
            requires_provider: Provider this tool requires

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            source=ToolSource.PROVIDER_NATIVE,
            provider_config=provider_config,
            ui=ui,
            requires_provider=requires_provider,
        )
        self.register(tool)
        return tool

    def register_mcp_stdio(
        self,
        name: str,
        description: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        parameters: dict[str, Any] | None = None,
        ui: ToolUIMetadata | None = None,
    ) -> ToolDefinition:
        """Register an MCP tool via stdio transport.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            command: Command to execute
            args: Command arguments
            env: Environment variables
            cwd: Working directory
            parameters: JSON Schema for tool parameters
            ui: UI display metadata

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.MCP_STDIO,
            mcp_server=MCPServerConfig(
                command=command,
                args=args,
                env=env,
                cwd=cwd,
            ),
            ui=ui,
        )
        self.register(tool)
        return tool

    def register_mcp_http(
        self,
        name: str,
        description: str,
        url: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        parameters: dict[str, Any] | None = None,
        ui: ToolUIMetadata | None = None,
    ) -> ToolDefinition:
        """Register an MCP tool via HTTP transport.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            url: Server URL
            api_key: API key for authentication
            headers: Additional HTTP headers
            parameters: JSON Schema for tool parameters
            ui: UI display metadata

        Returns:
            The registered ToolDefinition
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            inputSchema=parameters or {"type": "object", "properties": {}},
            source=ToolSource.MCP_HTTP,
            mcp_server=MCPServerConfig(
                url=url,
                api_key=api_key,
                headers=headers,
            ),
            ui=ui,
        )
        self.register(tool)
        return tool

    def load_from_json(self, path: Path) -> list[ToolDefinition]:
        """Load tool definitions from a JSON file.

        Expected format:
        {
            "tools": [
                {
                    "name": "tool_name",
                    "description": "...",
                    "source": "builtin|mcp_stdio|mcp_http|provider_native",
                    "inputSchema": {...},
                    "ui": {...},
                    ...
                }
            ]
        }

        Args:
            path: Path to JSON file

        Returns:
            List of loaded ToolDefinitions
        """
        loaded = []
        try:
            with open(path) as f:
                data = json.load(f)

            tools_data = data.get("tools", [])
            for tool_data in tools_data:
                # Convert nested objects
                if "ui" in tool_data and tool_data["ui"]:
                    tool_data["ui"] = ToolUIMetadata(**tool_data["ui"])
                if "mcp_server" in tool_data and tool_data["mcp_server"]:
                    tool_data["mcp_server"] = MCPServerConfig(**tool_data["mcp_server"])
                if "source" in tool_data:
                    tool_data["source"] = ToolSource(tool_data["source"])

                tool = ToolDefinition(**tool_data)
                self.register(tool)
                loaded.append(tool)

            logger.info(f"Loaded {len(loaded)} tools from {path}")
        except Exception as e:
            logger.error(f"Failed to load tools from {path}: {e}")

        return loaded

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str, provider: str | None = None) -> ToolDefinition | None:
        """Get a tool by name, optionally for a specific provider.

        Args:
            name: Tool name
            provider: Optional provider name to get provider-specific version

        Returns:
            ToolDefinition if found, None otherwise
        """
        # If provider specified, try provider-specific key first
        if provider:
            provider_key = f"{name}:{provider}"
            tool = self._tools.get(provider_key)
            if tool:
                return tool
            # Try compatible provider (e.g. azure_openai → openai)
            compat = PROVIDER_COMPAT.get(provider)
            if compat:
                tool = self._tools.get(f"{name}:{compat}")
                if tool:
                    return tool

        # Fall back to generic name
        return self._tools.get(name)

    def get_all(self) -> list[ToolDefinition]:
        """Get all registered tools.

        Returns:
            List of all ToolDefinitions
        """
        return list(self._tools.values())

    def get_visible(self) -> list[ToolDefinition]:
        """Get all tools that should be shown in UI.

        Returns:
            List of visible ToolDefinitions
        """
        return [t for t in self._tools.values() if t.is_visible()]

    def get_by_category(self, category: str) -> list[ToolDefinition]:
        """Get all tools in a specific category.

        Args:
            category: Category name

        Returns:
            List of ToolDefinitions in the category
        """
        return [
            t for t in self._tools.values()
            if t.ui and t.ui.category == category
        ]

    def get_for_provider(self, provider: str) -> list[ToolDefinition]:
        """Get all tools available for a specific provider.

        Args:
            provider: Provider name (openai, anthropic, etc.)

        Returns:
            List of ToolDefinitions available for the provider
        """
        return [
            t for t in self._tools.values()
            if t.is_available_for_provider(provider)
        ]

    def get_names(self) -> list[str]:
        """Get all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if registered
        """
        return name in self._tools

    def set_callback(self, name: str, callback: Callable[..., Any]) -> bool:
        """Set the callback for a builtin tool.

        This is useful for tools like generate_image where the callback
        needs to be created with runtime dependencies (like an OpenAI client).

        Args:
            name: Tool name
            callback: Callback function

        Returns:
            True if tool exists and callback was set
        """
        tool = self._tools.get(name)
        if tool and tool.source == ToolSource.BUILTIN:
            tool.callback = callback
            return True
        return False

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or source not supported

        Logging
        -------
        Two-tier logging at the dispatcher boundary, designed to be
        privacy-safe by default:

        * INFO  — metadata only: tool name, arg KEYS (not values), and
          elapsed ms. Safe to leave on in production. Provides the "did
          the LLM actually call this tool?" signal without ever writing
          user content to logs.
        * DEBUG — full truncated arg / result / error payload. Off by
          default. Local devs can flip the ``llming_models.tools.tool_registry``
          logger to DEBUG when they need to see what the LLM actually
          sent.

        This split exists because tool call args & results routinely
        contain user-authored document content, customer rows, etc. —
        material that under GDPR / EU rules must not land in long-lived
        log streams.
        """
        import logging as _logging
        import time
        tool = self.get(name)
        if not tool:
            logger.warning("MCP tool call: '%s' (NOT FOUND)", name)
            raise ValueError(f"Tool not found: {name}")

        arg_keys = sorted(arguments.keys()) if isinstance(arguments, dict) else []
        logger.info("MCP tool call: %s keys=%s", name, arg_keys)
        # Full payload only at DEBUG. ``isEnabledFor`` skips the
        # _format_for_log call entirely when DEBUG is off, so the
        # formatting cost stays at zero in production.
        if logger.isEnabledFor(_logging.DEBUG):
            logger.debug("  args=%s", _format_for_log(arguments, limit=400))
        started = time.monotonic()

        try:
            if tool.source == ToolSource.BUILTIN:
                if not tool.callback:
                    raise ValueError(f"Tool '{name}' has no callback configured")
                # Call the callback (may be sync or async)
                import asyncio
                if asyncio.iscoroutinefunction(tool.callback):
                    result = await tool.callback(**arguments)
                else:
                    result = tool.callback(**arguments)

            elif tool.source == ToolSource.PROVIDER_NATIVE:
                # Provider-native tools are handled by the provider, not here
                raise ValueError(f"Provider-native tool '{name}' must be executed by the provider")

            elif tool.source in (ToolSource.MCP_STDIO, ToolSource.MCP_HTTP, ToolSource.MCP_INPROCESS):
                # MCP tools require a connection
                connection = self._mcp_connections.get(name)
                if not connection:
                    raise ValueError(f"MCP connection not established for tool '{name}'")
                result = await connection.call_tool(name, arguments)

            elif tool.source == ToolSource.MCP_BROWSER:
                # Browser-hosted MCP tools — proxied via WebSocket to a Web Worker
                connection = self._mcp_connections.get(name)
                if not connection:
                    raise ValueError(f"Browser MCP connection not established for tool '{name}'")
                result = await connection.call_tool(name, arguments)

            else:
                raise ValueError(f"Unsupported tool source: {tool.source}")

        except Exception as exc:
            elapsed_ms = (time.monotonic() - started) * 1000
            # WARN gets only the exception TYPE — exception messages can
            # echo back the args (e.g. "no key 'foo' in {...}") and would
            # leak content. Full detail is at DEBUG.
            logger.warning(
                "MCP tool error: %s in %.0fms — %s",
                name, elapsed_ms, type(exc).__name__,
            )
            if logger.isEnabledFor(_logging.DEBUG):
                logger.debug("  exc detail: %s", exc)
            raise

        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info("MCP tool result: %s in %.0fms", name, elapsed_ms)
        if logger.isEnabledFor(_logging.DEBUG):
            logger.debug("  result=%s", _format_for_log(result, limit=400))
        return result


def _format_for_log(value: Any, limit: int = 400) -> str:
    """Render ``value`` as a single line for log output, truncating at
    ``limit`` characters. Used only by DEBUG-level lines (see
    :meth:`ToolRegistry.execute`); INFO never sees user-content values.
    Avoids blowing up logs with large MCP payloads (e.g. a full XLSX
    query result) while still preserving enough of the structure to
    read at a glance."""
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    else:
        text = str(value)
    text = text.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + f"...[+{len(text) - limit} chars]"
    return text


# Global default registry instance
_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Get the default global tool registry.

    Creates the registry on first access with default tools registered.

    Returns:
        The default ToolRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(auto_register_defaults=True)
    return _default_registry


def reset_default_registry() -> None:
    """Reset the default registry (primarily for testing)."""
    global _default_registry
    _default_registry = None
