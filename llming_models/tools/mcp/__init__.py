"""MCP (Model Context Protocol) integration — config, connections, and test utilities.

Connection classes and MCPTestClient are imported from their submodules directly
to avoid circular imports with tool_definition.py.
"""
from __future__ import annotations

from typing import Any

from llming_models.tools.mcp.config import MCPServerConfig

__all__ = [
    "MCPServerConfig",
]


def __getattr__(name: str) -> Any:
    """Lazy imports for connection classes to break circular dependency."""
    _connection_names = {
        "MCPConnection", "MCPStdioConnection", "MCPHTTPConnection",
        "MCPInProcessConnection", "InProcessMCPServer", "MCPError",
        "create_connection",
    }
    if name in _connection_names:
        from llming_models.tools.mcp import connection as _conn
        return getattr(_conn, name)
    if name == "MCPTestClient":
        from llming_models.tools.mcp.test_client import MCPTestClient
        return MCPTestClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
