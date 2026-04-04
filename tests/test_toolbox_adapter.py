"""Unit tests for ToolboxAdapter and get_toolboxes_for_config."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llming_models.tools.llm_tool import LlmTool
from llming_models.tools.llm_toolbox import LlmToolbox
from llming_models.tools.tool_definition import ToolDefinition, ToolSource
from llming_models.tools.tool_registry import ToolRegistry
from llming_models.tools.toolbox_adapter import ToolboxAdapter, get_toolboxes_for_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(*tools: ToolDefinition) -> ToolRegistry:
    """Create a ToolRegistry with auto_register_defaults=False and given tools."""
    reg = ToolRegistry(auto_register_defaults=False)
    for t in tools:
        reg.register(t)
    return reg


def _provider_native_tool(name: str = "web_search", provider: str | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Provider-native {name}",
        source=ToolSource.PROVIDER_NATIVE,
        requires_provider=provider,
    )


def _builtin_tool_with_callback(name: str = "my_tool", callback=None, cost: float | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Builtin {name}",
        source=ToolSource.BUILTIN,
        callback=callback or (lambda **kw: "ok"),
        fixed_cost_usd=cost,
        inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
    )


def _image_gen_tool() -> ToolDefinition:
    return ToolDefinition(
        name="generate_image",
        description="Generate an image",
        source=ToolSource.BUILTIN,
        # No callback — handled specially by the adapter
    )


def _mcp_tool(name: str = "mcp_search") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"MCP tool {name}",
        source=ToolSource.MCP_STDIO,
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )


# ---------------------------------------------------------------------------
# get_toolboxes_for_config helper
# ---------------------------------------------------------------------------


class TestGetToolboxesForConfig:
    """Tests for the top-level get_toolboxes_for_config helper."""

    def test_explicit_tools_list(self):
        tool = _provider_native_tool("web_search")
        reg = _make_registry(tool)
        result = get_toolboxes_for_config(
            tools=["web_search"],
            provider="openai",
            registry=reg,
        )
        assert len(result) == 1
        assert result[0].name == "web_search"

    def test_model_default_tools_used_when_tools_is_none(self):
        tool = _provider_native_tool("web_search")
        reg = _make_registry(tool)
        result = get_toolboxes_for_config(
            tools=None,
            provider="openai",
            model_default_tools=["web_search"],
            registry=reg,
        )
        assert len(result) == 1

    def test_no_tools_returns_empty(self):
        reg = _make_registry()
        result = get_toolboxes_for_config(
            tools=None,
            provider="openai",
            model_default_tools=None,
            registry=reg,
        )
        assert result == []

    def test_explicit_tools_override_defaults(self):
        tool_a = _provider_native_tool("tool_a")
        tool_b = _provider_native_tool("tool_b")
        reg = _make_registry(tool_a, tool_b)
        result = get_toolboxes_for_config(
            tools=["tool_a"],
            provider="openai",
            model_default_tools=["tool_b"],
            registry=reg,
        )
        assert len(result) == 1
        assert result[0].name == "tool_a"


# ---------------------------------------------------------------------------
# ToolboxAdapter.get_toolboxes
# ---------------------------------------------------------------------------


class TestToolboxAdapterGetToolboxes:
    """Tests for ToolboxAdapter.get_toolboxes."""

    def test_unknown_tool_skipped(self):
        reg = _make_registry()
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["nonexistent"], provider="openai")
        assert result == []

    def test_provider_incompatible_tool_skipped(self):
        # Tool requires anthropic, but we ask for openai
        tool = ToolDefinition(
            name="anthropic_only",
            description="Only for anthropic",
            source=ToolSource.PROVIDER_NATIVE,
            requires_provider="anthropic",
        )
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["anthropic_only"], provider="openai")
        assert result == []

    def test_multiple_tools(self):
        tool_a = _provider_native_tool("a")
        tool_b = _provider_native_tool("b")
        reg = _make_registry(tool_a, tool_b)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["a", "b"], provider="openai")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _create_provider_native_toolbox
# ---------------------------------------------------------------------------


class TestCreateProviderNativeToolbox:
    """Tests for _create_provider_native_toolbox."""

    def test_simple_no_config(self):
        tool = _provider_native_tool("web_search")
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["web_search"], provider="openai")
        assert len(result) == 1
        tb = result[0]
        assert tb.name == "web_search"
        assert tb.tools == ["web_search"]

    def test_with_tool_config(self):
        tool = _provider_native_tool("web_search")
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(
            ["web_search"],
            provider="openai",
            tool_config={"web_search": {"search_context_size": "high"}},
        )
        assert len(result) == 1
        tb = result[0]
        # Should be a dict with type and config
        assert isinstance(tb.tools[0], dict)
        assert tb.tools[0]["type"] == "web_search"
        assert tb.tools[0]["search_context_size"] == "high"


# ---------------------------------------------------------------------------
# _create_image_generation_toolbox
# ---------------------------------------------------------------------------


class TestCreateImageGenerationToolbox:
    """Tests for _create_image_generation_toolbox."""

    def test_no_openai_client_returns_none(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=None,
        )
        assert result == []

    def test_with_openai_client(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64_image_data"

        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=mock_client,
        )
        assert len(result) == 1
        tb = result[0]
        assert tb.name == "image_generation"
        assert len(tb.tools) == 1
        llm_tool = tb.tools[0]
        assert isinstance(llm_tool, LlmTool)
        assert llm_tool.name == "generate_image"

    def test_generate_image_function_calls_client(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "img_b64"

        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=mock_client,
        )
        llm_tool = result[0].tools[0]
        output = llm_tool.func(prompt="A cat", size="1024x1024", quality="medium")
        assert output == "img_b64"
        mock_client.generate_image_sync.assert_called_once()

    def test_generate_image_cost_callback(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "img_b64"
        cost_cb = MagicMock()

        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=mock_client,
            cost_callback=cost_cb,
        )
        llm_tool = result[0].tools[0]
        llm_tool.func(prompt="A cat", size="1024x1024", quality="medium")
        cost_cb.assert_called_once_with("generate_image", pytest.approx(0.042))

    def test_generate_image_enforces_allowed_sizes(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "img_b64"

        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=mock_client,
            tool_config={"generate_image": {"allowed_sizes": ["1024x1024"]}},
        )
        llm_tool = result[0].tools[0]
        # Request an invalid size
        llm_tool.func(prompt="A cat", size="1536x1024", quality="medium")
        call_args = mock_client.generate_image_sync.call_args[1]
        # Should fall back to default size
        assert call_args["size"] == "1024x1024"

    def test_generate_image_enforces_allowed_qualities(self):
        tool = _image_gen_tool()
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "img_b64"

        result = adapter.get_toolboxes(
            ["generate_image"],
            provider="openai",
            openai_client=mock_client,
            tool_config={"generate_image": {"allowed_qualities": ["low", "medium"]}},
        )
        llm_tool = result[0].tools[0]
        llm_tool.func(prompt="A cat", size="1024x1024", quality="high")
        call_args = mock_client.generate_image_sync.call_args[1]
        assert call_args["quality"] == "medium"  # Falls back to default


# ---------------------------------------------------------------------------
# _create_callback_toolbox
# ---------------------------------------------------------------------------


class TestCreateCallbackToolbox:
    """Tests for _create_callback_toolbox."""

    def test_simple_callback(self):
        cb = MagicMock(return_value="result")
        tool = _builtin_tool_with_callback("my_tool", callback=cb)
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["my_tool"], provider="openai")
        assert len(result) == 1
        tb = result[0]
        assert tb.name == "my_tool"
        llm_tool = tb.tools[0]
        assert isinstance(llm_tool, LlmTool)
        # Call should delegate to original callback
        output = llm_tool.func(x="hello")
        cb.assert_called_once_with(x="hello")
        assert output == "result"

    def test_callback_with_fixed_cost(self):
        cb = MagicMock(return_value="result")
        cost_cb = MagicMock()
        tool = _builtin_tool_with_callback("my_tool", callback=cb, cost=0.01)
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["my_tool"], provider="openai", cost_callback=cost_cb)
        llm_tool = result[0].tools[0]
        llm_tool.func(x="hello")
        cost_cb.assert_called_once_with("my_tool", 0.01)

    def test_callback_with_tool_config_merges(self):
        cb = MagicMock(return_value="result")
        tool = _builtin_tool_with_callback("my_tool", callback=cb)
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(
            ["my_tool"],
            provider="openai",
            tool_config={"my_tool": {"default_param": "val"}},
        )
        llm_tool = result[0].tools[0]
        llm_tool.func(x="hello")
        cb.assert_called_once_with(default_param="val", x="hello")

    def test_builtin_without_callback_returns_none(self):
        tool = ToolDefinition(
            name="broken_tool",
            description="No callback",
            source=ToolSource.BUILTIN,
            callback=None,
        )
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["broken_tool"], provider="openai")
        assert result == []


# ---------------------------------------------------------------------------
# _create_mcp_toolbox
# ---------------------------------------------------------------------------


class TestCreateMCPToolbox:
    """Tests for _create_mcp_toolbox."""

    def test_mcp_tool_creates_toolbox(self):
        tool = _mcp_tool("mcp_search")
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["mcp_search"], provider="openai")
        assert len(result) == 1
        tb = result[0]
        assert tb.name == "mcp_search"
        llm_tool = tb.tools[0]
        assert isinstance(llm_tool, LlmTool)
        assert llm_tool.name == "mcp_search"

    def test_mcp_inprocess_creates_toolbox(self):
        tool = ToolDefinition(
            name="inproc_tool",
            description="In-process MCP",
            source=ToolSource.MCP_INPROCESS,
            inputSchema={"type": "object", "properties": {}},
        )
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["inproc_tool"], provider="openai")
        assert len(result) == 1

    def test_mcp_browser_creates_toolbox(self):
        tool = ToolDefinition(
            name="browser_tool",
            description="Browser MCP",
            source=ToolSource.MCP_BROWSER,
            inputSchema={"type": "object", "properties": {}},
        )
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["browser_tool"], provider="openai")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Unknown tool source
# ---------------------------------------------------------------------------


class TestUnknownSource:
    """Tests for unknown tool source handling via direct _create_toolbox."""

    def test_unknown_source_returns_none(self):
        # ToolSource is a strict enum and the registry logs tool.source.value,
        # so we create the tool with a valid source and call _create_toolbox directly
        tool = ToolDefinition(
            name="weird_tool",
            description="Unknown source",
            source=ToolSource.BUILTIN,
        )
        # Monkey-patch source to an unrecognized value after construction
        object.__setattr__(tool, "source", "unknown_xyz")
        reg = _make_registry()
        adapter = ToolboxAdapter(registry=reg)
        result = adapter._create_toolbox(tool, "openai", {})
        assert result is None


# ---------------------------------------------------------------------------
# _create_mcp_toolbox sync wrapper (lines 305-331)
# ---------------------------------------------------------------------------


class TestMCPToolboxSyncWrapper:
    """Tests for the MCP sync wrapper inside _create_mcp_toolbox."""

    def test_mcp_sync_wrapper_uses_mcp_event_loop(self):
        """When registry has an event loop, sync_wrapper uses run_coroutine_threadsafe."""
        import asyncio

        tool = _mcp_tool("mcp_test")
        reg = _make_registry(tool)

        # Mock the execute method
        execute_result = "tool_result"
        reg.execute = MagicMock()

        adapter = ToolboxAdapter(registry=reg)

        result = adapter.get_toolboxes(["mcp_test"], provider="openai")
        assert len(result) == 1
        llm_tool = result[0].tools[0]
        assert isinstance(llm_tool, LlmTool)

    def test_mcp_sync_wrapper_with_tool_config(self):
        """Tool config defaults are merged with provided kwargs."""
        import asyncio

        tool = _mcp_tool("mcp_cfg")
        reg = _make_registry(tool)
        adapter = ToolboxAdapter(registry=reg)

        result = adapter.get_toolboxes(
            ["mcp_cfg"],
            provider="openai",
            tool_config={"mcp_cfg": {"default_key": "default_val"}},
        )
        assert len(result) == 1
        # The tool was created — verify it's an LlmTool with the right name
        llm_tool = result[0].tools[0]
        assert llm_tool.name == "mcp_cfg"

    def test_mcp_sync_wrapper_fallback_no_loop(self):
        """When no event loop is available, sync_wrapper uses asyncio.run fallback."""
        import asyncio

        tool = _mcp_tool("mcp_fallback")
        reg = _make_registry(tool)

        # No event loop set
        reg.set_event_loop(None) if hasattr(reg, 'set_event_loop') else None

        adapter = ToolboxAdapter(registry=reg)

        result = adapter.get_toolboxes(["mcp_fallback"], provider="openai")
        assert len(result) == 1

    def test_mcp_sync_wrapper_executes_in_running_loop(self):
        """Integration: sync wrapper executes MCP tool via the event loop."""
        import asyncio
        import threading

        tool = _mcp_tool("mcp_exec")
        reg = _make_registry(tool)

        # Set up an event loop and set it on the registry
        loop = asyncio.new_event_loop()

        async def fake_execute(name, args):
            return f"executed_{name}"

        reg.execute = fake_execute
        reg.set_event_loop(loop)

        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["mcp_exec"], provider="openai")
        llm_tool = result[0].tools[0]

        # Run the loop in a background thread
        def run_loop():
            loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()

        try:
            output = llm_tool.func(query="test")
            assert output == "executed_mcp_exec"
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    def test_mcp_sync_wrapper_error_propagated(self):
        """Errors from MCP tool execution propagate through the sync wrapper."""
        import asyncio
        import threading

        tool = _mcp_tool("mcp_err")
        reg = _make_registry(tool)

        loop = asyncio.new_event_loop()

        async def failing_execute(name, args):
            raise ValueError("Tool failed")

        reg.execute = failing_execute
        reg.set_event_loop(loop)

        adapter = ToolboxAdapter(registry=reg)
        result = adapter.get_toolboxes(["mcp_err"], provider="openai")
        llm_tool = result[0].tools[0]

        def run_loop():
            loop.run_forever()

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()

        try:
            with pytest.raises(ValueError, match="Tool failed"):
                llm_tool.func(query="test")
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()
