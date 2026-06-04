# MCP Integration

llming-models has first-class support for the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), enabling tool discovery and execution across local processes, remote servers, and in-process instances.

## MCPServerConfig

All MCP connections are configured via `MCPServerConfig`:

```python
from llming_models.tools.tool_definition import MCPServerConfig
```

## Connection Types

### Stdio (Local Process)

Launch an MCP server as a subprocess and communicate via stdin/stdout:

```python
config = MCPServerConfig(
    command="python",
    args=["-m", "my_mcp_server"],
    env={"API_KEY": "..."},
    cwd="/path/to/server",
    label="My MCP Server",
    enabled_by_default=True,
)
```

### HTTP/SSE (Remote Server)

Connect to a remote MCP server over HTTP:

```python
config = MCPServerConfig(
    url="https://api.example.com/mcp",
    api_key="your-key",
    headers={"Authorization": "Bearer ..."},
    label="Remote MCP",
)
```

### In-Process

Use a Python MCP server instance directly (no subprocess):

```python
from my_mcp_server import create_server

config = MCPServerConfig(
    server_instance=create_server(),
    label="In-Process MCP",
)
```

## Using MCP with ChatSession

Add MCP servers to your session config:

```python
from llming_models import ChatSession, LLMConfig
from llming_models.tools.tool_definition import MCPServerConfig

session = ChatSession(
    config=LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        mcp_servers=[
            MCPServerConfig(
                command="python",
                args=["-m", "llming_models.tools.math_mcp"],
                label="Math Tools",
                enabled_by_default=True,
            ),
        ],
    ),
)
```

## MCPConnection

The connection layer handles protocol communication:

```python
from llming_models.tools.mcp.connection import MCPStdioConnection

# Low-level usage (usually handled by ChatSession)
conn = MCPStdioConnection(config)
await conn.start()

tools = await conn.list_tools()   # returns list[ToolDefinition]
result = await conn.call_tool("calculate", {"expression": "2+2"})

await conn.close()
```

## UI Integration

`MCPServerConfig` includes rich UI metadata for chat interfaces:

```python
config = MCPServerConfig(
    command="python",
    args=["-m", "stock_agent"],
    label="Stock Agent",
    description="Real-time stock data and analysis",
    category="Finance",
    enabled_by_default=False,
    collapse_tools=True,         # show as single toggle
    flyout=True,                 # own top-level menu
    avatar="models/stock-avatar.png",
    auto_activate_keywords=["stock", "market", "ticker"],
)
```

| Field | Description |
|---|---|
| `label` | Display name in UI |
| `category` | Grouping category |
| `enabled_by_default` | Auto-enable on discovery |
| `default_enabled_tools` | Per-tool default enable list |
| `collapse_tools` | Single toggle vs. individual tools |
| `flyout` | Dedicated top-level menu |
| `avatar` | Custom chat avatar when tools are used |
| `auto_activate_keywords` | Keywords that trigger activation |

## Browser MCP

For browser-hosted MCP servers (via WebSocket + Web Worker), use the browser connection:

```python
from llming_models.tools.mcp.browser_connection import MCPBrowserConnection
# Used for MCP servers running in the browser context
```

!!! tip "Built-in MCP servers"
    llming-models ships with built-in MCP servers in `llming_models.tools`: `math_mcp` for mathematical operations and `gemini_image` for image generation.

---

## Session Isolation — Critical Security Requirement

**MCP tool names are NOT globally unique across sessions.** Two users can both have `create_document`, `list_files`, or any other tool name registered simultaneously. This creates a critical isolation hazard that must be understood before building multi-session servers.

### The global registry trap

`ToolRegistry` maintains a global `_mcp_connections` dict keyed by tool name. Every session that registers a tool overwrites the same entry. The last session to connect wins:

```
Session A registers create_document → registry._mcp_connections["create_document"] = conn_A
Session B registers create_document → registry._mcp_connections["create_document"] = conn_B  ← overwrites!

Session A's LLM calls create_document → registry.execute("create_document", ...) → uses conn_B ← WRONG USER
```

This is a **cross-user data leak**: Session A's LLM tool call executes against Session B's MCP instance. Any state that MCP touches (document stores, file systems, databases, callbacks, WebSockets) is exposed across user boundaries.

**This bug was confirmed in production** — a document created in one user's session appeared live in another user's browser because the MCP execution resolved to the wrong session's connection.

### The correct pattern

`ChatSession` keeps its own per-session connection dict (`self._mcp_connections`). Tool execution must resolve connections from this dict, never from the global registry:

```python
# WRONG — uses last-writer-wins global map
connection = registry._mcp_connections.get(tool_name)

# CORRECT — uses session-specific dict
connection = session._mcp_connections.get(tool_name)
```

`_build_toolboxes()` automatically passes `mcp_connections=self._mcp_connections` to `get_toolboxes_for_config`, so the standard `ChatSession` path is safe. The fix is **already in place** — do not bypass it.

### Rules for new MCP execution paths

- **Never call `get_default_registry().execute(tool_name, ...)` inside a per-session or per-user code path.** This reads from the global `_mcp_connections` map and violates isolation.
- **Never introduce a module-level dict keyed by tool name** that maps to anything session-specific (connections, stores, callbacks, WebSockets, user IDs). Tool names are not unique across sessions; session identity must be carried explicitly.
- **Always thread `mcp_connections`** (the session's own dict) to any new execution path that invokes MCP tools.
- When adding a new top-level execution path (e.g. a realtime voice pipeline, a batch runner), verify that it resolves connections from the calling session's dict, not the global registry.

### What the global registry IS for

`registry._mcp_connections` is still written to so that:
- Standalone scripts with a single session continue to work without threading a connections dict.
- The realtime Azure voice path (which does not have a `ChatSession` object) can function.

It is a fallback, not the primary path. The presence of an entry in `registry._mcp_connections` does NOT mean that entry is the right connection for the current session.
