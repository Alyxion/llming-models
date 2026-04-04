"""Unit tests for MCPBrowserConnection with mocked controller."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llming_models.tools.mcp.browser_connection import MCPBrowserConnection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(*, controller=None, pending=None, loop=None):
    """Build a session context dict for MCPBrowserConnection."""
    return {
        "controller": controller,
        "pending_requests": pending if pending is not None else {},
        "loop": loop,
    }


def _make_controller():
    """Create a mock controller with async _send."""
    ctrl = MagicMock()
    ctrl._send = AsyncMock()
    return ctrl


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestMCPBrowserConnectionInit:
    """Tests for MCPBrowserConnection.__init__."""

    def test_stores_nudge_uid(self):
        ctx = _make_ctx()
        conn = MCPBrowserConnection(nudge_uid="abc-123", session_ctx=ctx)
        assert conn.nudge_uid == "abc-123"

    def test_stores_context(self):
        ctx = _make_ctx()
        conn = MCPBrowserConnection(nudge_uid="abc", session_ctx=ctx)
        assert conn._ctx is ctx


# ---------------------------------------------------------------------------
# call_tool
# ---------------------------------------------------------------------------


class TestMCPBrowserConnectionCallTool:
    """Tests for MCPBrowserConnection.call_tool."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        ctrl = _make_controller()
        pending = {}
        ctx = _make_ctx(controller=ctrl, pending=pending)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        # Simulate the response arriving before the await completes
        async def fake_send(msg):
            # Find the request_id that was just registered
            req_id = msg["request_id"]
            entry = pending[req_id]
            entry["future"].set_result({"result": "tool output"})

        ctrl._send = fake_send

        result = await conn.call_tool("my_tool", {"arg": "val"})
        assert result == "tool output"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        ctrl = _make_controller()
        pending = {}
        ctx = _make_ctx(controller=ctrl, pending=pending)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        async def fake_send(msg):
            req_id = msg["request_id"]
            entry = pending[req_id]
            entry["future"].set_result({"error": "something went wrong"})

        ctrl._send = fake_send

        with pytest.raises(RuntimeError, match="Browser MCP tool error"):
            await conn.call_tool("bad_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_no_controller_raises(self):
        ctx = _make_ctx(controller=None)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        with pytest.raises(RuntimeError, match="No controller"):
            await conn.call_tool("tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self):
        ctrl = _make_controller()
        pending = {}
        ctx = _make_ctx(controller=ctrl, pending=pending)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        # Controller sends but nobody resolves the future
        async def fake_send(msg):
            pass  # Don't resolve the future

        ctrl._send = fake_send

        # Patch the timeout to be very short for testing
        import llming_models.tools.mcp.browser_connection as mod

        original_call = conn.call_tool

        async def short_timeout_call(name, arguments):
            """call_tool with a very short timeout for testing."""
            controller = conn._ctx.get("controller")
            if not controller:
                raise RuntimeError("No controller in browser MCP session context")

            from uuid import uuid4
            request_id = str(uuid4())
            loop = conn._ctx.get("loop") or asyncio.get_running_loop()
            future = loop.create_future()
            conn._ctx["pending_requests"][request_id] = {
                "future": future,
                "nudge_uid": conn.nudge_uid,
            }
            await controller._send({
                "type": "browser_mcp_call",
                "request_id": request_id,
                "nudge_uid": conn.nudge_uid,
                "tool_name": name,
                "arguments": arguments,
            })
            try:
                await asyncio.wait_for(future, timeout=0.01)
            except asyncio.TimeoutError:
                conn._ctx["pending_requests"].pop(request_id, None)
                raise RuntimeError(f"Browser MCP tool call '{name}' timed out (30s)")

        with pytest.raises(RuntimeError, match="timed out"):
            await short_timeout_call("slow_tool", {})

        # Pending request should be cleaned up
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_call_tool_sends_correct_message(self):
        ctrl = _make_controller()
        pending = {}
        ctx = _make_ctx(controller=ctrl, pending=pending)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        async def fake_send(msg):
            req_id = msg["request_id"]
            entry = pending[req_id]
            # Verify the message structure
            assert msg["type"] == "browser_mcp_call"
            assert msg["nudge_uid"] == "n1"
            assert msg["tool_name"] == "search"
            assert msg["arguments"] == {"query": "test"}
            entry["future"].set_result({"result": "ok"})

        ctrl._send = fake_send

        result = await conn.call_tool("search", {"query": "test"})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_call_tool_empty_result(self):
        ctrl = _make_controller()
        pending = {}
        ctx = _make_ctx(controller=ctrl, pending=pending)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        async def fake_send(msg):
            req_id = msg["request_id"]
            entry = pending[req_id]
            entry["future"].set_result({})  # No "result" key

        ctrl._send = fake_send

        result = await conn.call_tool("tool", {})
        assert result == ""  # Falls back to empty string


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestMCPBrowserConnectionClose:
    """Tests for MCPBrowserConnection.close."""

    @pytest.mark.asyncio
    async def test_close_sends_stop_message(self):
        ctrl = _make_controller()
        ctx = _make_ctx(controller=ctrl)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)

        await conn.close()

        ctrl._send.assert_called_once_with({
            "type": "stop_browser_mcp",
            "nudge_uid": "n1",
        })

    @pytest.mark.asyncio
    async def test_close_no_controller_is_noop(self):
        ctx = _make_ctx(controller=None)
        conn = MCPBrowserConnection(nudge_uid="n1", session_ctx=ctx)
        # Should not raise
        await conn.close()
