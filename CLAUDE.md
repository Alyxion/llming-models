# llming-models — assistant guide

Core LLM execution engine: multi-provider streaming, MCP tools, budget management. See `docs/` for architecture and public API. This file is the short list of non-obvious rules when editing code here.

## Hard rules

### MCP session isolation (CRITICAL — cross-user data leak risk)

**NEVER route MCP tool execution through `registry._mcp_connections` (the global registry's connection map) when running inside a multi-session server.**

`ToolRegistry._mcp_connections` is a **global dict keyed by tool name**. Every session that registers a tool name (e.g. `create_document`) overwrites the same entry. The last session to connect wins. Any subsequent tool call from any earlier session executes against the wrong MCP instance — which means the wrong document store, the wrong notify callback, and the wrong user's WebSocket.

This caused a confirmed cross-user document bleed bug: Session A's LLM called `create_document`, but the call executed against Session B's `DocumentCreatorMCP`, which emitted `doc_created` on Session B's socket. Session A's document appeared live in Session B's browser.

**The correct pattern:**
- Each `ChatSession` maintains its own `self._mcp_connections: dict[str, MCPConnection]`.
- `_build_toolboxes()` passes `mcp_connections=self._mcp_connections` to `get_toolboxes_for_config`.
- `_create_mcp_toolbox` captures `_connections = mcp_connections` (the session's dict) and resolves the connection from it at call time.
- Writing to `registry._mcp_connections` is still done for backward compat (standalone scripts, the realtime path), but **execution must never read from it inside a multi-session server**.

**When adding a new MCP execution path:**
- Always thread the session's `_mcp_connections` dict to the executor.
- Never call `registry.execute(tool_name, ...)` for user-facing tools in a multi-session context — it bypasses session isolation.
- If you see `get_default_registry().execute(...)` inside a per-session code path, treat it as a bug.

### No module-level shared state keyed by tool name

Never introduce a module-level dict, cache, or singleton that maps `tool_name → anything session-specific` (connections, stores, callbacks, WebSockets). Tool names are NOT unique across sessions. Session identity must be carried explicitly, not inferred from tool name.

### No `getattr` / `hasattr`

Access attributes directly. If something might not exist, give it a default on the class or use `isinstance` narrowing.

### Budget reservation before execution

`budget_manager.reserve_budget_async` must be called before any LLM API call, not after. Rolling back a reservation is safe; exceeding a limit silently is not.
