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
