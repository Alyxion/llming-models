# Tool System

llming-models provides an MCP-compatible tool system that supports built-in Python callbacks, provider-native tools, and MCP server integrations.

## ToolDefinition

Every tool is described by a `ToolDefinition` -- an MCP-compatible data model:

```python
from llming_models.tools.tool_definition import ToolDefinition, ToolSource

tool = ToolDefinition(
    name="calculate",
    description="Perform a mathematical calculation",
    inputSchema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            }
        },
        "required": ["expression"],
    },
    source=ToolSource.BUILTIN,
    callback=lambda expression: eval(expression),
)
```

## Tool Sources

| Source | Description | Example |
|---|---|---|
| `BUILTIN` | Direct Python callback | Custom functions |
| `PROVIDER_NATIVE` | Handled by the LLM provider | OpenAI web search |
| `MCP_STDIO` | Local MCP server via stdio | `python -m my_mcp_server` |
| `MCP_HTTP` | Remote MCP server via HTTP/SSE | `https://api.example.com/mcp` |
| `MCP_INPROCESS` | In-process MCP server | Direct Python MCP instance |
| `MCP_BROWSER` | Browser-hosted MCP via WebSocket | Web Worker MCP |

## Built-in Tools

### Web Search

Provider-native web search (OpenAI and Anthropic):

```python
from llming_models.tools.tool_definition import get_web_search_tool_for_provider

# Automatically selects the right tool for the provider
tool = get_web_search_tool_for_provider("openai")    # OpenAI native
tool = get_web_search_tool_for_provider("anthropic")  # Brave Search
```

### Image Generation

GPT Image generation tool:

```python
from llming_models.tools.tool_definition import DEFAULT_IMAGE_GENERATION_TOOL
# Fixed cost: ~$0.042 per generation
```

## Provider Filtering

Tools can be restricted to specific providers:

```python
tool = ToolDefinition(
    name="my_tool",
    description="Only works with OpenAI",
    requires_providers=["openai"],      # whitelist
    exclude_providers=["together"],     # blacklist
)

# Check compatibility
tool.is_available_for_provider("openai")      # True
tool.is_available_for_provider("anthropic")   # False
tool.is_available_for_provider("azure_openai") # True (via PROVIDER_COMPAT)
```

## UI Metadata

Tools can carry UI metadata for chat interfaces:

```python
from llming_models.tools.tool_definition import ToolUIMetadata

tool = ToolDefinition(
    name="generate_image",
    description="Generate images",
    ui=ToolUIMetadata(
        icon="image",
        display_name="Generate Image",
        category="media",
        hidden=False,
    ),
)

tool.get_display_name()     # "Generate Image"
tool.get_icon()             # "image"
tool.is_visible()           # True
```

## Tool Registry

The `ToolRegistry` manages a collection of tools:

```python
from llming_models.tools.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register(tool)

# Get tools available for a provider
tools = registry.get_tools_for_provider("openai")
```

## LlmToolbox

`LlmToolbox` wraps tools into a provider-compatible format:

```python
from llming_models.tools.llm_toolbox import LlmToolbox

toolbox = LlmToolbox(tools=[tool1, tool2])
# Pass to provider's create_client() as toolboxes=[toolbox]
```

!!! tip "MCP tools"
    For MCP server integration, see [MCP Integration](mcp.md). MCP tools are automatically wrapped as `ToolDefinition` objects with the appropriate source type.
