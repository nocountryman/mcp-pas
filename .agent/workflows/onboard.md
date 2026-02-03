---
description: Restore handoff context at session start
---

# Onboard Workflow

> 💡 **TIMING GUIDANCE**: Call at **session start** to restore context.
> Not recommended mid-session - context should be loaded upfront.

Use at the **start** of a session to restore context from previous work.

## When to Use

✅ **At session start** - First thing when continuing previous work
✅ **After `/handoff`** from previous session - To see what was handed off
✅ **To check for existing context** - See if there's pending work

## When NOT to Use

❌ **Mid-session** - Load context upfront, not during work
❌ **Repeatedly** - Once per session is enough
❌ **As a debugging tool** - Use database queries instead

## Default Behavior (Per-Project Singleton)

Gets THE active handoff for the current project and marks it as processed.

// turbo
```python
mcp_pas-server_onboard_session(project_id="mcp-pas", mark_processed=True)
```

## Response

```json
{
  "success": true,
  "mode": "singleton",
  "project_id": "mcp-pas",
  "handoff": {
    "handoff_id": "abc123...",
    "session_id": "90d87ad4...",
    "summary": "Phase 12 complete...",
    "next_task": "Test singleton enforcement",
    "linked_artifacts": ["handoff.py", "server.py"],
    "status": "processed"
  },
  "formatted_context": "## 🔄 Handoff: abc123..."
}
```

## Cross-Project Search (Explicit)

To search handoffs from other projects or by topic:

```python
# Search by topic across all projects
mcp_pas-server_onboard_session(topic="Phase 12 handoff")

# Search within specific project
mcp_pas-server_onboard_session(topic="LSP integration", project_id="other-project")

# Get specific handoff by ID
mcp_pas-server_onboard_session(handoff_id="abc123-def456-...")
```

## Lifecycle

```
┌─────────┐  /handoff   ┌────────┐  /onboard   ┌───────────┐
│ (none)  │ ──────────▶ │ ACTIVE │ ──────────▶ │ PROCESSED │
└─────────┘             └────────┘             └───────────┘
                             │
                             │ /handoff (new)
                             ▼
                        ┌──────────┐
                        │ ARCHIVED │
                        └──────────┘
```

## Best Practices

1. **Always onboard first** when continuing previous work
2. **Call once at session start** - not multiple times or mid-session
3. **Use `mark_processed=True`** to consume the handoff
4. **Cross-project is explicit** - must provide project_id or topic
