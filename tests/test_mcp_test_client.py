"""Unit tests for MCPTestClient with mocked subprocess."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llming_models.tools.mcp.test_client import MCPTestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_process():
    """Create a mock asyncio subprocess that speaks JSON-RPC over stdio."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    proc.stderr = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    return proc


def _encode_response(data: dict) -> bytes:
    return json.dumps(data).encode() + b"\n"


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestMCPTestClientInit:
    """Tests for MCPTestClient.__init__."""

    def test_default_server_module(self):
        client = MCPTestClient()
        assert client.server_module == "llming_models.tools.mcp.sample_server"

    def test_custom_server_module(self):
        client = MCPTestClient(server_module="my.custom.server")
        assert client.server_module == "my.custom.server"

    def test_initial_state(self):
        client = MCPTestClient()
        assert client.process is None
        assert client.request_id == 0
        assert client._stderr_task is None


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------


class TestMCPTestClientStart:
    """Tests for MCPTestClient.start."""

    @pytest.mark.asyncio
    async def test_start_spawns_subprocess(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()

        # _initialize sends request id=1 (initialize) and a notification
        init_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=init_response)
        mock_proc.stderr.readline = AsyncMock(return_value=b"")

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            await client.start()

        assert client.process is mock_proc
        assert client.request_id == 1  # After initialize
        # stdin.write should have been called at least twice (init request + notification)
        assert mock_proc.stdin.write.call_count >= 2


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------


class TestMCPTestClientListTools:
    """Tests for MCPTestClient.list_tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        tools_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {"name": "tool_a", "description": "Tool A"},
                    {"name": "tool_b", "description": "Tool B"},
                ]
            },
        })
        mock_proc.stdout.readline = AsyncMock(return_value=tools_response)

        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "tool_a"
        assert tools[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_list_tools_error(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        error_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid request"},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=error_response)

        with pytest.raises(RuntimeError, match="Error listing tools"):
            await client.list_tools()


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


class TestMCPTestClientCallTool:
    """Tests for MCPTestClient.call_tool."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        tool_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "Found 2 products"}],
            },
        })
        mock_proc.stdout.readline = AsyncMock(return_value=tool_response)

        result = await client.call_tool("search_products", {"query": "cable"})
        assert "content" in result
        assert result["content"][0]["text"] == "Found 2 products"

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        error_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32601, "message": "Tool not found"},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=error_response)

        result = await client.call_tool("bad_tool", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_call_tool_no_arguments(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        tool_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"content": [{"type": "text", "text": "Categories list"}]},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=tool_response)

        result = await client.call_tool("list_categories")
        assert "content" in result

    @pytest.mark.asyncio
    async def test_call_tool_sends_correct_request(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 5

        tool_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 6,
            "result": {"content": []},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=tool_response)

        await client.call_tool("search_products", {"query": "test"})

        # Verify the request that was written to stdin
        written = mock_proc.stdin.write.call_args[0][0]
        request = json.loads(written.decode().strip())
        assert request["method"] == "tools/call"
        assert request["params"]["name"] == "search_products"
        assert request["params"]["arguments"] == {"query": "test"}
        assert request["id"] == 6


# ---------------------------------------------------------------------------
# call_tool_text
# ---------------------------------------------------------------------------


class TestMCPTestClientCallToolText:
    """Tests for MCPTestClient.call_tool_text."""

    @pytest.mark.asyncio
    async def test_returns_text_content(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        tool_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {"type": "text", "text": "Line 1"},
                    {"type": "text", "text": "Line 2"},
                ],
            },
        })
        mock_proc.stdout.readline = AsyncMock(return_value=tool_response)

        text = await client.call_tool_text("some_tool")
        assert text == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_error_raises_runtime_error(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        client.request_id = 1

        error_response = _encode_response({
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -1, "message": "fail"},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=error_response)

        with pytest.raises(RuntimeError, match="Tool error"):
            await client.call_tool_text("some_tool")


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestMCPTestClientStop:
    """Tests for MCPTestClient.stop."""

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc

        await client.stop()
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_stderr_task(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc

        # Create a real asyncio task that we can cancel
        cancelled = False

        async def _long_running():
            nonlocal cancelled
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                cancelled = True
                raise

        task = asyncio.create_task(_long_running())
        client._stderr_task = task

        await client.stop()
        assert task.cancelled() or cancelled

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        client.process = mock_proc

        await client.stop()
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_no_process(self):
        client = MCPTestClient()
        client.process = None
        # Should not raise
        await client.stop()


# ---------------------------------------------------------------------------
# _send_request
# ---------------------------------------------------------------------------


class TestSendRequest:
    """Tests for MCPTestClient._send_request."""

    @pytest.mark.asyncio
    async def test_raises_if_not_started(self):
        client = MCPTestClient()
        client.process = None
        with pytest.raises(RuntimeError, match="Server not started"):
            await client._send_request("test")

    @pytest.mark.asyncio
    async def test_raises_on_closed_connection(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc
        # Simulate closed stdout
        mock_proc.stdout.readline = AsyncMock(return_value=b"")

        with pytest.raises(RuntimeError, match="Server closed connection"):
            await client._send_request("test")

    @pytest.mark.asyncio
    async def test_increments_request_id(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc

        resp1 = _encode_response({"jsonrpc": "2.0", "id": 1, "result": {}})
        resp2 = _encode_response({"jsonrpc": "2.0", "id": 2, "result": {}})
        mock_proc.stdout.readline = AsyncMock(side_effect=[resp1, resp2])

        await client._send_request("method1")
        assert client.request_id == 1
        await client._send_request("method2")
        assert client.request_id == 2

    @pytest.mark.asyncio
    async def test_send_request_without_params(self):
        """_send_request omits params key when None."""
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc

        resp = _encode_response({"jsonrpc": "2.0", "id": 1, "result": {}})
        mock_proc.stdout.readline = AsyncMock(return_value=resp)

        await client._send_request("test_method")

        written = mock_proc.stdin.write.call_args[0][0]
        request = json.loads(written.decode().strip())
        assert "params" not in request


# ---------------------------------------------------------------------------
# _read_stderr
# ---------------------------------------------------------------------------


class TestReadStderr:
    """Tests for MCPTestClient._read_stderr."""

    @pytest.mark.asyncio
    async def test_read_stderr_exits_on_empty(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")
        client.process = mock_proc

        # Should terminate without error
        await client._read_stderr()

    @pytest.mark.asyncio
    async def test_read_stderr_handles_exception(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        mock_proc.stderr.readline = AsyncMock(side_effect=RuntimeError("broken"))
        client.process = mock_proc

        # Should not raise
        await client._read_stderr()


# ---------------------------------------------------------------------------
# _initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for MCPTestClient._initialize."""

    @pytest.mark.asyncio
    async def test_initialize_error_raises(self):
        client = MCPTestClient()
        mock_proc = _make_mock_process()
        client.process = mock_proc

        error_resp = _encode_response({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -1, "message": "Protocol mismatch"},
        })
        mock_proc.stdout.readline = AsyncMock(return_value=error_resp)

        with pytest.raises(RuntimeError, match="Initialize error"):
            await client._initialize()


# ---------------------------------------------------------------------------
# interactive_session
# ---------------------------------------------------------------------------


class TestInteractiveSession:
    """Tests for the interactive_session() function with mocked stdin."""

    @pytest.mark.asyncio
    async def test_interactive_tools_command(self):
        """The 'tools' command lists available tools."""
        from llming_models.tools.mcp.test_client import interactive_session

        # Prepare mock client that will be used inside interactive_session
        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch.object(mock_client, "list_tools", new_callable=AsyncMock,
                                      return_value=[{"name": "tool_a", "description": "Tool A"}]):
                        # Simulate user typing "tools" then "quit"
                        with patch("builtins.input", side_effect=["tools", "quit"]):
                            await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_call_command(self):
        """The 'call <tool>' command calls a tool."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock()
        mock_client.call_tool_text = AsyncMock(return_value="Result: 3")

        tool_def = {
            "name": "add",
            "description": "Add numbers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a"],
            }
        }
        mock_client.list_tools.return_value = [tool_def]

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            # call add, then provide args, then quit
            with patch("builtins.input", side_effect=["call add", "5", "7", "quit"]):
                await interactive_session()

        mock_client.call_tool_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_interactive_call_no_arg(self):
        """The 'call' command without a tool name prints usage."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch("builtins.input", side_effect=["call", "quit"]):
                        await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_call_unknown_tool(self):
        """Calling an unknown tool prints an error."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch.object(mock_client, "list_tools", new_callable=AsyncMock,
                                      return_value=[]):
                        with patch("builtins.input", side_effect=["call nonexistent", "quit"]):
                            await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_unknown_command(self):
        """Unknown commands print an error message."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch("builtins.input", side_effect=["foobar", "exit"]):
                        await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_empty_input(self):
        """Empty input is skipped."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch("builtins.input", side_effect=["", "quit"]):
                        await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_eof(self):
        """EOFError from input exits gracefully."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch("builtins.input", side_effect=EOFError()):
                        await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_call_required_arg_missing(self):
        """When a required arg is missing, it prints error and continues."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        tool_def = {
            "name": "greet",
            "description": "Greet someone",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            }
        }

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch.object(mock_client, "list_tools", new_callable=AsyncMock,
                                      return_value=[tool_def]):
                        # Empty required arg, then quit
                        with patch("builtins.input", side_effect=["call greet", "", "quit"]):
                            await interactive_session()

    @pytest.mark.asyncio
    async def test_interactive_call_json_arg(self):
        """Arguments that are valid JSON are parsed."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock()
        mock_client.call_tool_text = AsyncMock(return_value="Done")

        tool_def = {
            "name": "process",
            "description": "Process data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                },
                "required": ["data"],
            }
        }
        mock_client.list_tools.return_value = [tool_def]

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch("builtins.input", side_effect=["call process", '{"key": "value"}', "quit"]):
                await interactive_session()

        call_args = mock_client.call_tool_text.call_args[0]
        assert call_args[1]["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_interactive_call_tool_error(self):
        """Tool errors are caught and printed."""
        from llming_models.tools.mcp.test_client import interactive_session

        mock_client = MagicMock(spec=MCPTestClient)
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.call_tool_text = AsyncMock(return_value="")

        tool_def = {
            "name": "fail",
            "description": "Fails",
            "inputSchema": {"type": "object", "properties": {}},
        }

        with patch("llming_models.tools.mcp.test_client.MCPTestClient", return_value=mock_client):
            with patch.object(mock_client, "start", new_callable=AsyncMock):
                with patch.object(mock_client, "stop", new_callable=AsyncMock):
                    with patch.object(mock_client, "list_tools", new_callable=AsyncMock,
                                      return_value=[tool_def]):
                        with patch.object(mock_client, "call_tool_text", new_callable=AsyncMock,
                                          side_effect=RuntimeError("Tool crashed")):
                            with patch("builtins.input", side_effect=["call fail", "quit"]):
                                await interactive_session()
