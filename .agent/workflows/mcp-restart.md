---
description: Restart PAS MCP server after code changes
---

# Restarting PAS MCP Server

When you make changes to PAS server code, the running MCP server needs to be restarted to pick up the changes.

## Option 1: IDE Restart ✅ (Reliable)

The most reliable approach for code changes:

1. Save all files
2. **Exit Antigravity IDE completely**
3. Restart Antigravity IDE

**Use when**: Any changes to `server.py` or helper modules in `src/pas/`.

> [!IMPORTANT]
> **Startup Time**: PAS takes 30-50 seconds to load due to SentenceTransformer model initialization.

## Option 2: mcpmon Hot-Reload (Experimental)

**Status: Partial Success**

The MCP config wraps PAS with `mcpmon` for file watching, but **Antigravity's MCP client doesn't auto-reconnect** when the server restarts.

**What works:**
- mcpmon detects file changes in `src/pas/`
- mcpmon restarts the Python server process
- The new code is loaded correctly

**What doesn't work:**
- Antigravity IDE doesn't automatically reconnect to the restarted server
- MCP tool calls fail with "Invalid request parameters" after hot-reload
- **No programmatic API exists** for agent to trigger refresh

### Workaround: User Clicks Refresh ✅

After mcpmon restarts the server (30-50 seconds after code change):

1. Open Agent Side Panel (`Ctrl+L`)
2. Click `...` menu → **Manage MCP Servers**
3. Click **Refresh** button

This restores the MCP connection without full IDE restart.

### Verification (2026-02-02)

Tested by adding a marker constant to `server.py`:
- File change detected by mcpmon ✅
- Python process restarted ✅
- New code loaded (verified via standalone test) ✅
- MCP tools accessible from agent ❌ (connection not refreshed)

## Configuration (For Reference)

The MCP config at `~/.gemini/antigravity/mcp_config.json`:

```json
{
  "pas-server": {
    "command": "/home/nocoma/.local/node_modules/.bin/mcpmon",
    "args": [
      "--watch", "/home/nocoma/Documents/MCP/PAS/src/pas",
      "--ext", "py",
      "--",
      "/home/nocoma/Documents/MCP/PAS/.venv312/bin/python",
      "-m", "pas.server"
    ]
  }
}
```

## Recommendation

**For now, use IDE restart** until Antigravity adds native MCP server reload support.

Batch your code changes to minimize restarts:
1. Make all related edits
2. Verify syntax: `python -c "from pas.server import ..."`
3. Restart IDE once

---

*Created: 2026-02-02*
*Updated: 2026-02-02 - Documented mcpmon limitation with Antigravity client*
