"""Tests for the tool system: definitions, registry, enums, and related models."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from llming_models.tools.builtin_tools import (
    IMAGE_GEN_PRICING,
    DALLE3_PRICING,
    create_image_generation_tool,
    create_web_search_toolbox,
    create_image_generation_toolbox,
)
from llming_models.tools.llm_tool import LlmTool
from llming_models.tools.llm_toolbox import LlmToolbox
from llming_models.tools.mcp.config import MCPServerConfig
from llming_models.tools.tool_call import ToolCallInfo, ToolCallStatus
from llming_models.tools.tool_definition import (
    PROVIDER_COMPAT,
    ToolDefinition,
    ToolSource,
    ToolUIMetadata,
    OPENAI_WEB_SEARCH_TOOL,
    ANTHROPIC_WEB_SEARCH_TOOL,
    DEFAULT_IMAGE_GENERATION_TOOL,
    get_web_search_tool_for_provider,
)
from llming_models.tools.tool_registry import (
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)


# ---------------------------------------------------------------------------
# ToolSource enum
# ---------------------------------------------------------------------------


class TestToolSource:
    """Tests for ToolSource enum."""

    def test_values(self):
        assert ToolSource.BUILTIN == "builtin"
        assert ToolSource.MCP_STDIO == "mcp_stdio"
        assert ToolSource.MCP_HTTP == "mcp_http"
        assert ToolSource.MCP_INPROCESS == "mcp_inprocess"
        assert ToolSource.MCP_BROWSER == "mcp_browser"
        assert ToolSource.PROVIDER_NATIVE == "provider_native"

    def test_all_values_are_strings(self):
        for member in ToolSource:
            assert isinstance(member.value, str)

    def test_membership(self):
        assert ToolSource("builtin") == ToolSource.BUILTIN


# ---------------------------------------------------------------------------
# ToolUIMetadata
# ---------------------------------------------------------------------------


class TestToolUIMetadata:
    """Tests for ToolUIMetadata."""

    def test_defaults(self):
        meta = ToolUIMetadata()
        assert meta.icon is None
        assert meta.display_name is None
        assert meta.description is None
        assert meta.category is None
        assert meta.hidden is False
        assert meta.color is None

    def test_construction_with_values(self):
        meta = ToolUIMetadata(
            icon="search",
            display_name="Web Search",
            description="Search the web",
            category="search",
            hidden=True,
            color="#ff0000",
        )
        assert meta.icon == "search"
        assert meta.display_name == "Web Search"
        assert meta.hidden is True
        assert meta.color == "#ff0000"


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    """Tests for ToolDefinition construction and methods."""

    def test_minimal_construction(self):
        tool = ToolDefinition(name="test_tool", description="A test tool")
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.source == ToolSource.BUILTIN  # default
        assert tool.inputSchema == {"type": "object", "properties": {}}

    def test_full_construction(self):
        tool = ToolDefinition(
            name="my_tool",
            description="Does something",
            inputSchema={"type": "object", "properties": {"x": {"type": "integer"}}},
            source=ToolSource.MCP_HTTP,
            requires_provider="openai",
            exclude_providers=["google"],
            fixed_cost_usd=0.05,
            realtime_enabled=True,
        )
        assert tool.source == ToolSource.MCP_HTTP
        assert tool.requires_provider == "openai"
        assert tool.exclude_providers == ["google"]
        assert tool.fixed_cost_usd == 0.05
        assert tool.realtime_enabled is True

    def test_to_mcp_dict(self):
        tool = ToolDefinition(
            name="test",
            description="desc",
            inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        d = tool.to_mcp_dict()
        assert d == {
            "name": "test",
            "description": "desc",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }

    def test_to_mcp_dict_excludes_extra_fields(self):
        tool = ToolDefinition(
            name="test", description="desc",
            source=ToolSource.MCP_STDIO,
            requires_provider="openai",
        )
        d = tool.to_mcp_dict()
        assert "source" not in d
        assert "requires_provider" not in d

    def test_get_display_name_with_ui(self):
        tool = ToolDefinition(
            name="snake_case_name", description="d",
            ui=ToolUIMetadata(display_name="Pretty Name"),
        )
        assert tool.get_display_name() == "Pretty Name"

    def test_get_display_name_fallback(self):
        tool = ToolDefinition(name="snake_case_name", description="d")
        assert tool.get_display_name() == "Snake Case Name"

    def test_get_ui_description_with_ui(self):
        tool = ToolDefinition(
            name="t", description="Full description",
            ui=ToolUIMetadata(description="Short desc"),
        )
        assert tool.get_ui_description() == "Short desc"

    def test_get_ui_description_fallback(self):
        tool = ToolDefinition(name="t", description="Full description")
        assert tool.get_ui_description() == "Full description"

    def test_get_icon(self):
        tool = ToolDefinition(name="t", description="d", ui=ToolUIMetadata(icon="star"))
        assert tool.get_icon() == "star"

    def test_get_icon_none(self):
        tool = ToolDefinition(name="t", description="d")
        assert tool.get_icon() is None

    def test_is_visible_default(self):
        tool = ToolDefinition(name="t", description="d")
        assert tool.is_visible() is True

    def test_is_visible_hidden(self):
        tool = ToolDefinition(name="t", description="d", ui=ToolUIMetadata(hidden=True))
        assert tool.is_visible() is False

    def test_is_visible_not_hidden(self):
        tool = ToolDefinition(name="t", description="d", ui=ToolUIMetadata(hidden=False))
        assert tool.is_visible() is True


# ---------------------------------------------------------------------------
# ToolDefinition.is_available_for_provider — extended tests
# ---------------------------------------------------------------------------


class TestToolAvailabilityExtended:
    """Extended tests beyond test_tool_provider_compat.py."""

    def test_requires_providers_list_allows_listed(self):
        tool = ToolDefinition(name="t", description="d", requires_providers=["openai", "anthropic"])
        assert tool.is_available_for_provider("openai")
        assert tool.is_available_for_provider("anthropic")
        assert not tool.is_available_for_provider("google")

    def test_requires_providers_list_with_compat(self):
        """azure_openai is compat with openai, so should pass if openai is in list."""
        tool = ToolDefinition(name="t", description="d", requires_providers=["openai"])
        assert tool.is_available_for_provider("azure_openai")

    def test_requires_providers_takes_precedence_over_requires_provider(self):
        """requires_providers (list) takes precedence over singular requires_provider."""
        tool = ToolDefinition(
            name="t", description="d",
            requires_provider="google",
            requires_providers=["openai"],
        )
        # requires_providers wins: google NOT in list
        assert not tool.is_available_for_provider("google")
        assert tool.is_available_for_provider("openai")

    def test_exclude_providers_with_no_requires(self):
        tool = ToolDefinition(name="t", description="d", exclude_providers=["mistral", "together"])
        assert tool.is_available_for_provider("openai")
        assert not tool.is_available_for_provider("mistral")
        assert not tool.is_available_for_provider("together")

    def test_azure_anthropic_compat(self):
        """azure_anthropic is compat with anthropic."""
        tool = ToolDefinition(name="t", description="d", requires_provider="anthropic")
        assert tool.is_available_for_provider("azure_anthropic")

    def test_provider_compat_mapping(self):
        assert PROVIDER_COMPAT["azure_openai"] == "openai"
        assert PROVIDER_COMPAT["azure_anthropic"] == "anthropic"


# ---------------------------------------------------------------------------
# ToolCallStatus / ToolCallInfo
# ---------------------------------------------------------------------------


class TestToolCallStatus:
    """Tests for ToolCallStatus enum."""

    def test_values(self):
        assert ToolCallStatus.PENDING == "pending"
        assert ToolCallStatus.EXECUTING == "executing"
        assert ToolCallStatus.COMPLETED == "completed"
        assert ToolCallStatus.ERROR == "error"


class TestToolCallInfo:
    """Tests for ToolCallInfo."""

    def test_construction(self):
        info = ToolCallInfo(
            name="generate_image",
            call_id="call_123",
            status=ToolCallStatus.COMPLETED,
            arguments={"prompt": "A cat"},
            result="base64data",
        )
        assert info.name == "generate_image"
        assert info.call_id == "call_123"
        assert info.status == ToolCallStatus.COMPLETED

    def test_display_name(self):
        info = ToolCallInfo(name="generate_image", call_id="c1", status=ToolCallStatus.PENDING)
        assert info.display_name == "Generate Image"

    def test_is_image_generation(self):
        info = ToolCallInfo(name="generate_image", call_id="c1", status=ToolCallStatus.PENDING)
        assert info.is_image_generation is True

    def test_is_not_image_generation(self):
        info = ToolCallInfo(name="web_search", call_id="c1", status=ToolCallStatus.PENDING)
        assert info.is_image_generation is False

    def test_optional_fields_default_none(self):
        info = ToolCallInfo(name="t", call_id="c1", status=ToolCallStatus.PENDING)
        assert info.arguments is None
        assert info.result is None
        assert info.error is None

    def test_error_field(self):
        info = ToolCallInfo(
            name="t", call_id="c1", status=ToolCallStatus.ERROR,
            error="Something went wrong",
        )
        assert info.error == "Something went wrong"


# ---------------------------------------------------------------------------
# LlmTool
# ---------------------------------------------------------------------------


class TestLlmTool:
    """Tests for LlmTool construction."""

    def test_construction(self):
        def my_func(x: int) -> int:
            return x * 2
        tool = LlmTool(name="double", description="Doubles a number", func=my_func)
        assert tool.name == "double"
        assert tool.description == "Doubles a number"
        assert tool.func is my_func
        assert tool.parameters == {}

    def test_construction_with_parameters(self):
        tool = LlmTool(
            name="t", description="d", func=lambda: None,
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        assert "x" in tool.parameters["properties"]


# ---------------------------------------------------------------------------
# LlmToolbox
# ---------------------------------------------------------------------------


class TestLlmToolbox:
    """Tests for LlmToolbox construction."""

    def test_construction_with_tool_objects(self):
        tool = LlmTool(name="t1", description="d1", func=lambda: None)
        toolbox = LlmToolbox(name="box", description="A toolbox", tools=[tool])
        assert toolbox.name == "box"
        assert len(toolbox.tools) == 1
        assert toolbox.tools[0].name == "t1"

    def test_construction_with_strings(self):
        toolbox = LlmToolbox(name="search", description="Search tools", tools=["web_search"])
        assert toolbox.tools == ["web_search"]

    def test_construction_with_mixed_types(self):
        tool = LlmTool(name="t1", description="d1", func=lambda: None)
        toolbox = LlmToolbox(name="mixed", description="Mixed", tools=[tool, "web_search", {"type": "custom"}])
        assert len(toolbox.tools) == 3


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_creation_with_defaults(self):
        reg = ToolRegistry(auto_register_defaults=True)
        assert reg.has("web_search")
        assert reg.has("generate_image")

    def test_creation_without_defaults(self):
        reg = ToolRegistry(auto_register_defaults=False)
        assert not reg.has("web_search")
        assert not reg.has("generate_image")

    def test_register_and_get(self):
        reg = ToolRegistry(auto_register_defaults=False)
        tool = ToolDefinition(name="my_tool", description="Custom tool")
        reg.register(tool)
        assert reg.get("my_tool") is tool
        assert reg.has("my_tool")

    def test_get_nonexistent_returns_none(self):
        reg = ToolRegistry(auto_register_defaults=False)
        assert reg.get("nonexistent") is None

    def test_get_all(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="a", description="d"))
        reg.register(ToolDefinition(name="b", description="d"))
        all_tools = reg.get_all()
        assert len(all_tools) == 2
        names = [t.name for t in all_tools]
        assert "a" in names
        assert "b" in names

    def test_get_names(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="x", description="d"))
        assert "x" in reg.get_names()

    def test_has(self):
        reg = ToolRegistry(auto_register_defaults=False)
        assert not reg.has("x")
        reg.register(ToolDefinition(name="x", description="d"))
        assert reg.has("x")

    def test_unregister(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="x", description="d"))
        assert reg.unregister("x") is True
        assert not reg.has("x")

    def test_unregister_nonexistent(self):
        reg = ToolRegistry(auto_register_defaults=False)
        assert reg.unregister("nonexistent") is False

    def test_register_builtin(self):
        reg = ToolRegistry(auto_register_defaults=False)
        tool = reg.register_builtin(
            name="custom_fn",
            description="Custom function",
            callback=lambda x: x,
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        assert tool.source == ToolSource.BUILTIN
        assert tool.callback is not None
        assert reg.has("custom_fn")

    def test_register_provider_native(self):
        reg = ToolRegistry(auto_register_defaults=False)
        tool = reg.register_provider_native(
            name="native_tool",
            description="A native tool",
            provider_config={"type": "web_search"},
            requires_provider="openai",
        )
        assert tool.source == ToolSource.PROVIDER_NATIVE
        assert tool.requires_provider == "openai"

    def test_register_mcp_stdio(self):
        reg = ToolRegistry(auto_register_defaults=False)
        tool = reg.register_mcp_stdio(
            name="mcp_tool",
            description="An MCP tool",
            command="python",
            args=["-m", "my_server"],
        )
        assert tool.source == ToolSource.MCP_STDIO
        assert tool.mcp_server is not None
        assert tool.mcp_server.command == "python"

    def test_register_mcp_http(self):
        reg = ToolRegistry(auto_register_defaults=False)
        tool = reg.register_mcp_http(
            name="remote_tool",
            description="Remote MCP tool",
            url="https://mcp.example.com",
            api_key="sk-test",
        )
        assert tool.source == ToolSource.MCP_HTTP
        assert tool.mcp_server is not None
        assert tool.mcp_server.url == "https://mcp.example.com"

    def test_get_visible(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="visible", description="d"))
        reg.register(ToolDefinition(name="hidden", description="d", ui=ToolUIMetadata(hidden=True)))
        visible = reg.get_visible()
        names = [t.name for t in visible]
        assert "visible" in names
        assert "hidden" not in names

    def test_get_by_category(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="s1", description="d", ui=ToolUIMetadata(category="search")))
        reg.register(ToolDefinition(name="m1", description="d", ui=ToolUIMetadata(category="media")))
        reg.register(ToolDefinition(name="no_cat", description="d"))
        search_tools = reg.get_by_category("search")
        assert len(search_tools) == 1
        assert search_tools[0].name == "s1"

    def test_get_for_provider(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="oai", description="d", requires_provider="openai"))
        reg.register(ToolDefinition(name="anth", description="d", requires_provider="anthropic"))
        reg.register(ToolDefinition(name="any", description="d"))

        oai_tools = reg.get_for_provider("openai")
        oai_names = [t.name for t in oai_tools]
        assert "oai" in oai_names
        assert "any" in oai_names
        assert "anth" not in oai_names

    def test_set_callback(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="t", description="d", source=ToolSource.BUILTIN))
        new_cb = lambda: "result"
        assert reg.set_callback("t", new_cb) is True
        assert reg.get("t").callback is new_cb

    def test_set_callback_wrong_source(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="t", description="d", source=ToolSource.PROVIDER_NATIVE))
        assert reg.set_callback("t", lambda: None) is False

    def test_set_callback_missing_tool(self):
        reg = ToolRegistry(auto_register_defaults=False)
        assert reg.set_callback("nonexistent", lambda: None) is False

    def test_set_event_loop(self):
        reg = ToolRegistry(auto_register_defaults=False)
        loop = MagicMock()
        reg.set_event_loop(loop)
        assert reg.get_event_loop() is loop

    @pytest.mark.asyncio
    async def test_execute_builtin(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(
            name="add", description="Add numbers",
            callback=lambda a, b: a + b,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        )
        result = await reg.execute("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_builtin_async_callback(self):
        async def async_fn(x: int) -> int:
            return x * 10

        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(name="mul", description="Multiply", callback=async_fn)
        result = await reg.execute("mul", {"x": 5})
        assert result == 50

    @pytest.mark.asyncio
    async def test_execute_missing_tool_raises(self):
        reg = ToolRegistry(auto_register_defaults=False)
        with pytest.raises(ValueError, match="Tool not found"):
            await reg.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_builtin_no_callback_raises(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="empty", description="d", source=ToolSource.BUILTIN))
        with pytest.raises(ValueError, match="no callback"):
            await reg.execute("empty", {})

    @pytest.mark.asyncio
    async def test_execute_provider_native_raises(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="native", description="d", source=ToolSource.PROVIDER_NATIVE))
        with pytest.raises(ValueError, match="must be executed by the provider"):
            await reg.execute("native", {})

    @pytest.mark.asyncio
    async def test_execute_mcp_no_connection_raises(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="mcp_t", description="d", source=ToolSource.MCP_STDIO))
        with pytest.raises(ValueError, match="MCP connection not established"):
            await reg.execute("mcp_t", {})

    @pytest.mark.asyncio
    async def test_execute_browser_mcp_no_connection_raises(self):
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register(ToolDefinition(name="bm", description="d", source=ToolSource.MCP_BROWSER))
        with pytest.raises(ValueError, match="Browser MCP connection not established"):
            await reg.execute("bm", {})

    # ------------------------------------------------------------------
    # Tool-call logging at the dispatcher boundary
    # ------------------------------------------------------------------
    #
    # Two-tier logging:
    #   * INFO — metadata only (tool name + arg KEYS + elapsed). Safe in
    #     production; never writes user content.
    #   * DEBUG — full truncated payload. For local diagnostics only.
    #
    # The privacy guarantee at INFO is the load-bearing assertion — if
    # ``test_info_log_does_not_leak_arg_values`` ever fails, somebody put
    # a value into an INFO line and that goes straight into long-lived
    # log streams.

    @pytest.mark.asyncio
    async def test_info_log_metadata_only(self, caplog):
        import logging
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(
            name="echo", description="Echo back",
            callback=lambda x: f"got:{x}",
        )
        with caplog.at_level(logging.INFO,
                             logger="llming_models.tools.tool_registry"):
            await reg.execute("echo", {"x": "hello"})

        info_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.INFO]
        assert any("MCP tool call: echo" in m and "keys=['x']" in m
                   for m in info_msgs), info_msgs
        assert any("MCP tool result: echo in" in m and "ms" in m
                   for m in info_msgs), info_msgs

    @pytest.mark.asyncio
    async def test_info_log_does_not_leak_arg_values(self, caplog):
        """Privacy regression: at INFO level, arg VALUES must never
        appear. Prod runs at INFO; values landing here would mean user
        content streams straight into Azure log storage."""
        import logging
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(
            name="store", description="Store something",
            callback=lambda secret_payload: "ok",
        )
        sentinel = "SENSITIVE-CUSTOMER-CONTENT-12345"
        with caplog.at_level(logging.INFO,
                             logger="llming_models.tools.tool_registry"):
            await reg.execute("store", {"secret_payload": sentinel})

        info_records = [r for r in caplog.records
                        if r.levelno == logging.INFO]
        joined = " | ".join(r.getMessage() for r in info_records)
        assert sentinel not in joined, (
            f"INFO logs leaked arg value: {joined!r}"
        )
        assert "secret_payload" in joined

    @pytest.mark.asyncio
    async def test_info_log_does_not_leak_result_values(self, caplog):
        """Same privacy bar for return values — query_table results,
        section content, etc. all go through this path."""
        import logging
        sentinel = "RESULT-CONTAINS-CUSTOMER-NAME-Hannah-Mustermann"
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(name="query", description="Returns rows",
                             callback=lambda: {"rows": [{"name": sentinel}]})
        with caplog.at_level(logging.INFO,
                             logger="llming_models.tools.tool_registry"):
            await reg.execute("query", {})
        info_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.INFO]
        joined = " | ".join(info_msgs)
        assert sentinel not in joined, (
            f"INFO logs leaked result value: {joined!r}"
        )

    @pytest.mark.asyncio
    async def test_debug_log_includes_full_payload(self, caplog):
        """At DEBUG, full truncated args & results show up — that's the
        opt-in diagnostic mode developers use locally."""
        import logging
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(
            name="echo", description="Echo back",
            callback=lambda x: f"got:{x}",
        )
        with caplog.at_level(logging.DEBUG,
                             logger="llming_models.tools.tool_registry"):
            await reg.execute("echo", {"x": "hello"})
        debug_msgs = [r.getMessage() for r in caplog.records
                      if r.levelno == logging.DEBUG]
        joined = " | ".join(debug_msgs)
        assert '"x": "hello"' in joined or "'x': 'hello'" in joined, joined
        assert "got:hello" in joined, joined

    @pytest.mark.asyncio
    async def test_warn_on_exception_omits_message(self, caplog):
        """The exception TYPE goes to WARN; the message (which can echo
        args back to the user) is gated to DEBUG."""
        import logging
        sentinel = "PRIVATE-PATH-customer-12345"
        def boom(**_):
            raise ValueError(f"path '{sentinel}' not found")
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(name="boom", description="Fails", callback=boom)
        with caplog.at_level(logging.WARNING,
                             logger="llming_models.tools.tool_registry"):
            with pytest.raises(ValueError):
                await reg.execute("boom", {})
        warn_msgs = [r.getMessage() for r in caplog.records
                     if r.levelno == logging.WARNING]
        joined = " | ".join(warn_msgs)
        assert "MCP tool error: boom" in joined
        assert "ValueError" in joined
        assert sentinel not in joined, (
            f"WARN log leaked exception detail: {joined!r}"
        )

    @pytest.mark.asyncio
    async def test_debug_log_truncates_huge_payloads(self, caplog):
        """Tool calls returning a very large payload (e.g. a full XLSX
        query result) should not blow up the log — the formatter
        truncates to a fixed limit so DEBUG logs stay readable."""
        import logging
        big = "X" * 10_000
        reg = ToolRegistry(auto_register_defaults=False)
        reg.register_builtin(name="big", description="Returns huge string",
                             callback=lambda: big)
        with caplog.at_level(logging.DEBUG,
                             logger="llming_models.tools.tool_registry"):
            await reg.execute("big", {})
        result_lines = [r.getMessage() for r in caplog.records
                        if "result=" in r.getMessage()]
        assert result_lines, caplog.records
        assert "[+" in result_lines[0]
        assert len(result_lines[0]) < 1000

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_warns(self, caplog):
        import logging
        reg = ToolRegistry(auto_register_defaults=False)
        with caplog.at_level(logging.WARNING,
                             logger="llming_models.tools.tool_registry"):
            with pytest.raises(ValueError, match="Tool not found"):
                await reg.execute("ghost", {})
        warnings = [r.getMessage() for r in caplog.records
                    if r.levelno >= logging.WARNING]
        assert any("MCP tool call: 'ghost' (NOT FOUND)" in m
                   for m in warnings), warnings

    def test_get_with_provider_specific_key(self):
        reg = ToolRegistry(auto_register_defaults=True)
        # Default registry registers web_search:openai and web_search:anthropic
        tool = reg.get("web_search", provider="openai")
        assert tool is not None
        assert tool.requires_provider == "openai"

    def test_get_with_compat_provider(self):
        reg = ToolRegistry(auto_register_defaults=True)
        # azure_openai should resolve to openai via compat
        tool = reg.get("web_search", provider="azure_openai")
        assert tool is not None
        assert tool.requires_provider == "openai"

    def test_load_from_json(self, tmp_path):
        import json
        data = {
            "tools": [
                {
                    "name": "json_tool",
                    "description": "A tool from JSON",
                    "source": "builtin",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
        path = tmp_path / "tools.json"
        path.write_text(json.dumps(data))

        reg = ToolRegistry(auto_register_defaults=False)
        loaded = reg.load_from_json(path)
        assert len(loaded) == 1
        assert loaded[0].name == "json_tool"
        assert reg.has("json_tool")


# ---------------------------------------------------------------------------
# get_default_registry / reset_default_registry
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    """Tests for singleton registry."""

    def test_returns_singleton(self):
        reset_default_registry()
        a = get_default_registry()
        b = get_default_registry()
        assert a is b

    def test_reset_clears_singleton(self):
        reset_default_registry()
        a = get_default_registry()
        reset_default_registry()
        b = get_default_registry()
        assert a is not b


# ---------------------------------------------------------------------------
# builtin_tools.IMAGE_GEN_PRICING
# ---------------------------------------------------------------------------


class TestImageGenPricing:
    """Tests for IMAGE_GEN_PRICING constants."""

    def test_has_expected_keys(self):
        expected = [
            ("1024x1024", "low"),
            ("1024x1024", "medium"),
            ("1024x1024", "high"),
            ("1536x1024", "low"),
            ("1024x1536", "low"),
            ("1536x1024", "medium"),
            ("1024x1536", "medium"),
            ("1536x1024", "high"),
            ("1024x1536", "high"),
        ]
        for key in expected:
            assert key in IMAGE_GEN_PRICING, f"Missing key: {key}"

    def test_all_values_are_positive_floats(self):
        for key, value in IMAGE_GEN_PRICING.items():
            assert isinstance(value, float)
            assert value > 0

    def test_backwards_compat_alias(self):
        assert DALLE3_PRICING is IMAGE_GEN_PRICING


# ---------------------------------------------------------------------------
# MCPServerConfig
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    """Tests for MCPServerConfig construction and type detection."""

    def test_stdio_config(self):
        cfg = MCPServerConfig(command="python", args=["-m", "server"])
        assert cfg.is_stdio() is True
        assert cfg.is_http() is False
        assert cfg.is_inprocess() is False

    def test_http_config(self):
        cfg = MCPServerConfig(url="https://mcp.example.com", api_key="sk-test")
        assert cfg.is_stdio() is False
        assert cfg.is_http() is True
        assert cfg.is_inprocess() is False

    def test_inprocess_config(self):
        mock_server = MagicMock()
        cfg = MCPServerConfig(server_instance=mock_server)
        assert cfg.is_stdio() is False
        assert cfg.is_http() is False
        assert cfg.is_inprocess() is True

    def test_ui_metadata_fields(self):
        cfg = MCPServerConfig(
            command="python",
            label="My Server",
            description="Does things",
            category="Experimental",
            enabled_by_default=True,
            collapse_tools=True,
            flyout=True,
        )
        assert cfg.label == "My Server"
        assert cfg.description == "Does things"
        assert cfg.category == "Experimental"
        assert cfg.enabled_by_default is True
        assert cfg.collapse_tools is True
        assert cfg.flyout is True

    def test_default_values(self):
        cfg = MCPServerConfig()
        assert cfg.command is None
        assert cfg.url is None
        assert cfg.server_instance is None
        assert cfg.enabled_by_default is False
        assert cfg.collapse_tools is False
        assert cfg.flyout is False
        assert cfg.exclude_providers is None
        assert cfg.requires_providers is None

    def test_env_and_cwd(self):
        cfg = MCPServerConfig(
            command="node",
            env={"NODE_ENV": "production"},
            cwd="/app",
        )
        assert cfg.env == {"NODE_ENV": "production"}
        assert cfg.cwd == "/app"

    def test_headers(self):
        cfg = MCPServerConfig(
            url="https://mcp.example.com",
            headers={"X-Custom": "value"},
        )
        assert cfg.headers == {"X-Custom": "value"}

    def test_default_enabled_tools(self):
        cfg = MCPServerConfig(
            command="python",
            enabled_by_default=True,
            default_enabled_tools=["tool_a", "tool_b"],
        )
        assert cfg.default_enabled_tools == ["tool_a", "tool_b"]

    def test_auto_activate_keywords(self):
        cfg = MCPServerConfig(
            command="python",
            auto_activate_keywords=["math", "calculate"],
        )
        assert cfg.auto_activate_keywords == ["math", "calculate"]

    def test_avatar(self):
        cfg = MCPServerConfig(command="python", avatar="models/custom-avatar.gif")
        assert cfg.avatar == "models/custom-avatar.gif"


# ---------------------------------------------------------------------------
# get_web_search_tool_for_provider
# ---------------------------------------------------------------------------


class TestGetWebSearchToolForProvider:
    """Tests for get_web_search_tool_for_provider factory."""

    def test_anthropic_returns_anthropic_tool(self):
        tool = get_web_search_tool_for_provider("anthropic")
        assert tool is ANTHROPIC_WEB_SEARCH_TOOL

    def test_openai_returns_openai_tool(self):
        tool = get_web_search_tool_for_provider("openai")
        assert tool is OPENAI_WEB_SEARCH_TOOL

    def test_unknown_returns_openai_tool(self):
        tool = get_web_search_tool_for_provider("mistral")
        assert tool is OPENAI_WEB_SEARCH_TOOL


# ---------------------------------------------------------------------------
# create_image_generation_tool / create_web_search_toolbox
# ---------------------------------------------------------------------------


class TestBuiltinToolFactories:
    """Tests for factory functions in builtin_tools."""

    def test_create_image_generation_tool(self):
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64_image_data"
        tool = create_image_generation_tool(mock_client)
        assert tool.name == "generate_image"
        assert "prompt" in tool.parameters["properties"]

    def test_create_image_generation_tool_callback_works(self):
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "base64_result"
        tool = create_image_generation_tool(mock_client)
        result = tool.func(prompt="A cat", size="1024x1024", quality="medium")
        assert result == "base64_result"

    def test_create_image_generation_tool_with_cost_callback(self):
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "data"
        cost_calls = []
        tool = create_image_generation_tool(mock_client, cost_callback=lambda name, cost: cost_calls.append((name, cost)))
        tool.func(prompt="A cat", size="1024x1024", quality="medium")
        assert len(cost_calls) == 1
        assert cost_calls[0][0] == "generate_image"
        assert cost_calls[0][1] == 0.042  # medium 1024x1024

    def test_create_web_search_toolbox(self):
        toolbox = create_web_search_toolbox()
        assert toolbox.name == "web_search"
        assert "web_search" in toolbox.tools

    def test_create_image_generation_toolbox(self):
        mock_client = MagicMock()
        mock_client.generate_image_sync.return_value = "data"
        toolbox = create_image_generation_toolbox(mock_client)
        assert toolbox.name == "image_generation"
        assert len(toolbox.tools) == 1
        assert toolbox.tools[0].name == "generate_image"
