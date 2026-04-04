"""Tests for MCP connections: stdio, HTTP, in-process, and factory function."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.tools.mcp.config import MCPServerConfig
from llming_models.tools.mcp.connection import (
    MCPConnection,
    MCPError,
    MCPHTTPConnection,
    MCPInProcessConnection,
    MCPStdioConnection,
    InProcessMCPServer,
    create_connection,
)
from llming_models.tools.tool_definition import ToolDefinition, ToolSource


# ---------------------------------------------------------------------------
# MCPError
# ---------------------------------------------------------------------------


class TestMCPError:
    """Tests for MCPError exception."""

    def test_construction(self):
        err = MCPError("something went wrong")
        assert str(err) == "something went wrong"

    def test_is_exception(self):
        assert issubclass(MCPError, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(MCPError, match="test error"):
            raise MCPError("test error")


# ---------------------------------------------------------------------------
# MCPStdioConnection
# ---------------------------------------------------------------------------


class TestMCPStdioConnection:
    """Tests for MCPStdioConnection with mocked subprocess."""

    def test_init_requires_command(self):
        with pytest.raises(ValueError, match="must have 'command'"):
            MCPStdioConnection(MCPServerConfig(url="http://example.com"))

    def test_init_with_valid_config(self):
        cfg = MCPServerConfig(command="python", args=["-m", "my_server"])
        conn = MCPStdioConnection(cfg)
        assert conn.config is cfg
        assert conn.process is None
        assert conn._started is False
        assert conn._request_id == 0

    @pytest.mark.asyncio
    async def test_start_spawns_process(self):
        cfg = MCPServerConfig(command="echo", args=["hello"])
        conn = MCPStdioConnection(cfg)

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.returncode = 0  # Process exited immediately

        # Mock _initialize to avoid actually sending JSON-RPC
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_process):
            with patch.object(conn, "_initialize", new_callable=AsyncMock, return_value={}):
                with patch.object(conn, "_read_responses", new_callable=AsyncMock):
                    await conn.start()

        assert conn._started is True
        assert conn.process is mock_process

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)
        conn._started = True  # Already started

        # Should return immediately without spawning
        await conn.start()
        assert conn.process is None  # No process created

    @pytest.mark.asyncio
    async def test_list_tools_parses_response(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn._started = True

        mock_result = {
            "tools": [
                {
                    "name": "calculate",
                    "description": "Do math",
                    "inputSchema": {"type": "object", "properties": {"expr": {"type": "string"}}},
                },
                {
                    "name": "search",
                    "description": "Search things",
                },
            ]
        }

        with patch.object(conn, "_send_request", new_callable=AsyncMock, return_value=mock_result):
            tools = await conn.list_tools()

        assert len(tools) == 2
        assert all(isinstance(t, ToolDefinition) for t in tools)
        assert tools[0].name == "calculate"
        assert tools[0].source == ToolSource.MCP_STDIO
        assert tools[0].mcp_server is cfg
        assert tools[1].name == "search"

    @pytest.mark.asyncio
    async def test_call_tool_single_content(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn._started = True

        mock_result = {
            "content": [{"type": "text", "text": "42"}]
        }

        with patch.object(conn, "_send_request", new_callable=AsyncMock, return_value=mock_result):
            result = await conn.call_tool("calculate", {"expr": "6*7"})

        assert result == "42"

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn._started = True

        mock_result = {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        }

        with patch.object(conn, "_send_request", new_callable=AsyncMock, return_value=mock_result):
            result = await conn.call_tool("multi", {})

        assert len(result) == 2
        assert result[0]["text"] == "first"

    @pytest.mark.asyncio
    async def test_close_terminates_process(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn._started = True

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()
        conn.process = mock_process

        # Create a real asyncio task that will be cancelled
        async def _noop():
            await asyncio.sleep(100)

        reader_task = asyncio.create_task(_noop())
        conn._reader_task = reader_task

        await conn.close()

        assert conn._started is False
        mock_process.terminate.assert_called_once()
        assert reader_task.cancelled()

    @pytest.mark.asyncio
    async def test_close_kills_on_timeout(self):
        """If process doesn't terminate within timeout, kill it."""
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn._started = True
        conn._reader_task = None

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        # After kill, the second wait() succeeds
        mock_process.wait = AsyncMock(return_value=None)
        conn.process = mock_process

        # Make wait_for raise TimeoutError (simulating slow shutdown)
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            await conn.close()

        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_request_raises_when_not_started(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn.process = None

        with pytest.raises(MCPError, match="Connection not started"):
            await conn._send_request("test", {})

    @pytest.mark.asyncio
    async def test_send_notification_raises_when_not_started(self):
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)
        conn.process = None

        with pytest.raises(MCPError, match="Connection not started"):
            await conn._send_notification("test", {})


# ---------------------------------------------------------------------------
# MCPHTTPConnection
# ---------------------------------------------------------------------------


class TestMCPHTTPConnection:
    """Tests for MCPHTTPConnection with mocked aiohttp."""

    def test_init_requires_url(self):
        with pytest.raises(ValueError, match="must have 'url'"):
            MCPHTTPConnection(MCPServerConfig(command="python"))

    def test_init_with_valid_config(self):
        cfg = MCPServerConfig(url="https://mcp.example.com", api_key="sk-test")
        conn = MCPHTTPConnection(cfg)
        assert conn.config is cfg
        assert conn._session is None
        assert conn._started is False

    def test_build_headers_basic(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        headers = conn._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_build_headers_with_api_key(self):
        cfg = MCPServerConfig(url="https://mcp.example.com", api_key="sk-test")
        conn = MCPHTTPConnection(cfg)
        headers = conn._build_headers()
        assert headers["Authorization"] == "Bearer sk-test"

    def test_build_headers_with_custom_headers(self):
        cfg = MCPServerConfig(
            url="https://mcp.example.com",
            headers={"X-Custom": "value", "X-Other": "data"},
        )
        conn = MCPHTTPConnection(cfg)
        headers = conn._build_headers()
        assert headers["X-Custom"] == "value"
        assert headers["X-Other"] == "data"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_start_creates_session(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)

        mock_session = MagicMock()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            await conn.start()

        assert conn._started is True
        assert conn._session is mock_session

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True
        existing_session = MagicMock()
        conn._session = existing_session

        await conn.start()
        assert conn._session is existing_session  # Not replaced

    @pytest.mark.asyncio
    async def test_start_raises_without_aiohttp(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)

        with patch.dict("sys.modules", {"aiohttp": None}):
            with patch("builtins.__import__", side_effect=ImportError("no aiohttp")):
                with pytest.raises(MCPError, match="aiohttp is required"):
                    await conn.start()

    @pytest.mark.asyncio
    async def test_list_tools_parses_response(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_response_data = {
            "tools": [
                {"name": "search", "description": "Search stuff"},
                {"name": "calc", "description": "Calculate"},
            ]
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_data)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        tools = await conn.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "search"
        assert tools[0].source == ToolSource.MCP_HTTP
        assert tools[1].name == "calc"

    @pytest.mark.asyncio
    async def test_list_tools_raises_on_error_status(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        with pytest.raises(MCPError, match="Failed to list tools: 500"):
            await conn.list_tools()

    @pytest.mark.asyncio
    async def test_list_tools_raises_when_not_started(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._session = None

        with pytest.raises(MCPError, match="Connection not started"):
            await conn.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_single_content(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "content": [{"type": "text", "text": "result"}]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        result = await conn.call_tool("search", {"query": "hello"})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ]
        })

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        result = await conn.call_tool("multi", {})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_call_tool_raises_on_error_status(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 422

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        with pytest.raises(MCPError, match="Tool call failed: 422"):
            await conn.call_tool("broken", {})

    @pytest.mark.asyncio
    async def test_call_tool_raises_when_not_started(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._session = None

        with pytest.raises(MCPError, match="Connection not started"):
            await conn.call_tool("test", {})

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_session = AsyncMock()
        conn._session = mock_session

        await conn.close()

        mock_session.close.assert_called_once()
        assert conn._session is None
        assert conn._started is False

    @pytest.mark.asyncio
    async def test_close_when_no_session(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = False
        conn._session = None

        # Should not raise
        await conn.close()
        assert conn._started is False


# ---------------------------------------------------------------------------
# MCPInProcessConnection
# ---------------------------------------------------------------------------


class TestMCPInProcessConnection:
    """Tests for MCPInProcessConnection with mocked InProcessMCPServer."""

    def _make_mock_server(self) -> MagicMock:
        server = MagicMock(spec=InProcessMCPServer)
        server.list_tools = AsyncMock(return_value=[])
        server.call_tool = AsyncMock(return_value="ok")
        server.get_prompt_hints = AsyncMock(return_value=[])
        server.get_client_renderers = AsyncMock(return_value=[])
        return server

    def test_init_requires_server_instance(self):
        cfg = MCPServerConfig(command="python")
        with pytest.raises(AssertionError, match="server_instance"):
            MCPInProcessConnection(cfg)

    def test_init_with_valid_config(self):
        server = self._make_mock_server()
        cfg = MCPServerConfig(server_instance=server, label="Test Server")
        conn = MCPInProcessConnection(cfg)
        assert conn.server is server
        assert conn.config is cfg

    @pytest.mark.asyncio
    async def test_start_is_noop(self):
        server = self._make_mock_server()
        cfg = MCPServerConfig(server_instance=server, label="Test")
        conn = MCPInProcessConnection(cfg)

        # Should not raise
        await conn.start()

    @pytest.mark.asyncio
    async def test_list_tools_delegates_to_server(self):
        server = self._make_mock_server()
        server.list_tools = AsyncMock(return_value=[
            {
                "name": "solve",
                "description": "Solve equations",
                "inputSchema": {"type": "object", "properties": {"eq": {"type": "string"}}},
            },
            {
                "name": "plot",
                "description": "Plot graph",
                "displayName": "Plot Graph",
                "icon": "chart",
                "displayDescription": "Plot a function",
            },
        ])
        cfg = MCPServerConfig(server_instance=server)
        conn = MCPInProcessConnection(cfg)

        tools = await conn.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "solve"
        assert tools[0].source == ToolSource.MCP_INPROCESS
        assert tools[0].ui is None  # No display metadata

        assert tools[1].name == "plot"
        assert tools[1].ui is not None
        assert tools[1].ui.display_name == "Plot Graph"
        assert tools[1].ui.icon == "chart"
        assert tools[1].ui.description == "Plot a function"

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        server = self._make_mock_server()
        server.list_tools = AsyncMock(return_value=[])
        cfg = MCPServerConfig(server_instance=server)
        conn = MCPInProcessConnection(cfg)

        tools = await conn.list_tools()
        assert tools == []

    @pytest.mark.asyncio
    async def test_call_tool_delegates_to_server(self):
        server = self._make_mock_server()
        server.call_tool = AsyncMock(return_value="x = 5")
        cfg = MCPServerConfig(server_instance=server)
        conn = MCPInProcessConnection(cfg)

        result = await conn.call_tool("solve", {"eq": "x + 3 = 8"})

        assert result == "x = 5"
        server.call_tool.assert_called_once_with("solve", {"eq": "x + 3 = 8"})

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        server = self._make_mock_server()
        cfg = MCPServerConfig(server_instance=server, label="Test")
        conn = MCPInProcessConnection(cfg)

        # Should not raise
        await conn.close()


# ---------------------------------------------------------------------------
# create_connection factory
# ---------------------------------------------------------------------------


class TestCreateConnection:
    """Tests for create_connection() factory function."""

    def test_stdio_config_returns_stdio_connection(self):
        cfg = MCPServerConfig(command="python", args=["-m", "server"])
        conn = create_connection(cfg)
        assert isinstance(conn, MCPStdioConnection)

    def test_http_config_returns_http_connection(self):
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = create_connection(cfg)
        assert isinstance(conn, MCPHTTPConnection)

    def test_inprocess_config_returns_inprocess_connection(self):
        server = MagicMock(spec=InProcessMCPServer)
        cfg = MCPServerConfig(server_instance=server)
        conn = create_connection(cfg)
        assert isinstance(conn, MCPInProcessConnection)

    def test_empty_config_raises(self):
        cfg = MCPServerConfig()  # No command, url, or server_instance
        with pytest.raises(ValueError, match="must have either"):
            create_connection(cfg)

    def test_dict_config_converted(self):
        """create_connection also accepts dict configs from model_dump."""
        cfg_dict = {"command": "node", "args": ["server.js"]}
        conn = create_connection(cfg_dict)
        assert isinstance(conn, MCPStdioConnection)

    def test_inprocess_takes_precedence(self):
        """When server_instance is set alongside command/url, inprocess wins."""
        server = MagicMock(spec=InProcessMCPServer)
        cfg = MCPServerConfig(command="python", url="http://x.com", server_instance=server)
        conn = create_connection(cfg)
        assert isinstance(conn, MCPInProcessConnection)


# ---------------------------------------------------------------------------
# InProcessMCPServer (abstract base)
# ---------------------------------------------------------------------------


class TestInProcessMCPServer:
    """Tests for InProcessMCPServer abstract base class."""

    @pytest.mark.asyncio
    async def test_get_prompt_hints_default_empty(self):
        """Default implementation returns empty list."""

        class MinimalServer(InProcessMCPServer):
            async def list_tools(self):
                return []

            async def call_tool(self, name, arguments):
                return ""

        server = MinimalServer()
        hints = await server.get_prompt_hints()
        assert hints == []

    @pytest.mark.asyncio
    async def test_get_client_renderers_default_empty(self):
        """Default implementation returns empty list."""

        class MinimalServer(InProcessMCPServer):
            async def list_tools(self):
                return []

            async def call_tool(self, name, arguments):
                return ""

        server = MinimalServer()
        renderers = await server.get_client_renderers()
        assert renderers == []


# ---------------------------------------------------------------------------
# Helper: async context manager mock for aiohttp
# ---------------------------------------------------------------------------


class AsyncContextManager:
    """Helper that wraps a mock response as an async context manager."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# MCPStdioConnection: _read_responses
# ---------------------------------------------------------------------------


class TestMCPStdioReadResponses:
    """Tests for _read_responses in MCPStdioConnection."""

    @pytest.mark.asyncio
    async def test_read_responses_resolves_futures(self):
        """_read_responses reads JSON from stdout and resolves pending futures."""
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)

        # Set up a mock process with async stdout
        response_bytes = b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n'

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[response_bytes, b""])

        mock_process = MagicMock()
        mock_process.returncode = None  # Process still running
        mock_process.stdout = mock_stdout
        conn.process = mock_process

        # Set up a pending future
        future = asyncio.get_running_loop().create_future()
        conn._pending_requests[1] = future

        await conn._read_responses()

        assert future.result() == {"ok": True}

    @pytest.mark.asyncio
    async def test_read_responses_error_sets_exception(self):
        """_read_responses sets exception on future for error responses."""
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)

        error_bytes = b'{"jsonrpc": "2.0", "id": 1, "error": {"message": "Not found"}}\n'

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[error_bytes, b""])

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdout = mock_stdout
        conn.process = mock_process

        future = asyncio.get_running_loop().create_future()
        conn._pending_requests[1] = future

        await conn._read_responses()

        with pytest.raises(MCPError, match="Not found"):
            future.result()

    @pytest.mark.asyncio
    async def test_read_responses_handles_bad_json(self):
        """_read_responses handles malformed JSON gracefully."""
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(side_effect=[b"this is not json\n", b""])

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdout = mock_stdout
        conn.process = mock_process

        # Should not raise
        await conn._read_responses()

    @pytest.mark.asyncio
    async def test_read_responses_stops_on_empty_line(self):
        """_read_responses exits when stdout returns empty bytes."""
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"")

        mock_process = MagicMock()
        mock_process.returncode = None  # Still "running"
        mock_process.stdout = mock_stdout
        conn.process = mock_process

        await conn._read_responses()

    @pytest.mark.asyncio
    async def test_read_responses_no_stdout(self):
        """_read_responses exits when stdout is None."""
        cfg = MCPServerConfig(command="echo")
        conn = MCPStdioConnection(cfg)

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdout = None
        conn.process = mock_process

        await conn._read_responses()


# ---------------------------------------------------------------------------
# MCPStdioConnection: _send_request timeout and _send_notification
# ---------------------------------------------------------------------------


class TestMCPStdioRequestFlow:
    """Tests for request/notification flow in MCPStdioConnection."""

    @pytest.mark.asyncio
    async def test_send_request_timeout(self):
        """_send_request raises MCPError on timeout."""
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        conn.process = mock_process

        # Create a future that will never resolve
        with pytest.raises(MCPError, match="timed out"):
            # Use a very short timeout
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                await conn._send_request("test", {})

    @pytest.mark.asyncio
    async def test_send_notification_success(self):
        """_send_notification writes JSON-RPC notification without id."""
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)

        mock_stdin = MagicMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        conn.process = mock_process

        await conn._send_notification("notifications/test", {"key": "val"})

        written = mock_stdin.write.call_args[0][0]
        parsed = json.loads(written.decode().strip())
        assert parsed["jsonrpc"] == "2.0"
        assert parsed["method"] == "notifications/test"
        assert "id" not in parsed  # Notifications don't have ids

    @pytest.mark.asyncio
    async def test_initialize_flow(self):
        """_initialize sends initialize request and initialized notification."""
        cfg = MCPServerConfig(command="python")
        conn = MCPStdioConnection(cfg)

        mock_send = AsyncMock(return_value={"capabilities": {}})
        mock_notify = AsyncMock()
        conn._send_request = mock_send
        conn._send_notification = mock_notify

        result = await conn._initialize()

        mock_send.assert_called_once_with("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "llming-lodge", "version": "1.0.0"},
        })
        mock_notify.assert_called_once_with("notifications/initialized", {})
        assert result == {"capabilities": {}}


# ---------------------------------------------------------------------------
# MCPHTTPConnection: extended flow tests
# ---------------------------------------------------------------------------


class TestMCPHTTPConnectionExtended:
    """Extended tests for MCPHTTPConnection covering more paths."""

    @pytest.mark.asyncio
    async def test_start_with_api_key_and_custom_headers(self):
        """HTTP connection builds correct headers with api_key and custom headers."""
        cfg = MCPServerConfig(
            url="https://mcp.example.com",
            api_key="sk-123",
            headers={"X-Session": "abc"},
        )
        conn = MCPHTTPConnection(cfg)

        mock_session = MagicMock()
        with patch("aiohttp.ClientSession", return_value=mock_session) as mock_cls:
            await conn.start()

        call_kwargs = mock_cls.call_args[1]
        headers = call_kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-123"
        assert headers["X-Session"] == "abc"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_call_tool_empty_content(self):
        """HTTP call_tool with empty content returns empty list."""
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"content": []})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        result = await conn.call_tool("empty", {})
        assert result == []

    @pytest.mark.asyncio
    async def test_list_tools_with_input_schema(self):
        """HTTP list_tools preserves inputSchema from server."""
        cfg = MCPServerConfig(url="https://mcp.example.com")
        conn = MCPHTTPConnection(cfg)
        conn._started = True

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "tools": [{
                "name": "tool_with_schema",
                "description": "Has schema",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }]
        })

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=AsyncContextManager(mock_resp))
        conn._session = mock_session

        tools = await conn.list_tools()
        assert tools[0].inputSchema["required"] == ["query"]
