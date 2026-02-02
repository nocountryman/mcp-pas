---
description: Create a session handoff to preserve context for future agents
---

# Session Handoff Workflow

Use this at the end of a working session to preserve context for future agents.

## When to Use

- End of complex multi-step work
- Before switching to a different task
- When work is partially complete and needs continuation
- To document key decisions and context

## Commands

| Mode | Tool Call | Behavior |
|------|-----------|----------|
| **new** | `create_handoff(session_id, summary, ...)` | Create new handoff (archives previous ones for same session) |
| **list** | `onboard_session()` | List all active handoffs |
| **search** | `onboard_session(topic="...")` | Semantic search handoffs |
| **restore** | `onboard_session(handoff_id="...", mark_processed=true)` | Get specific handoff and consume it |

## Creating a Handoff

### Step 1: Summarize the Session

Prepare a concise summary of:
- What was accomplished
- Key decisions made
- Any blockers or unresolved issues

### Step 2: Create the Handoff

```python
mcp_pas-server_create_handoff(
    session_id="<active-pas-session-id>",
    summary="<what was done, key decisions, current state>",
    next_task="<suggested next step for future agent>",
    context={"key_decisions": [...], "blockers": [...]},
    linked_artifacts="file1.py,file2.sql"
)
```

**Note:** Creating a new handoff **automatically archives** any previous active handoffs for the same session.

### Step 3: Verify Creation

The tool returns a `handoff_id` and archived count:
```json
{
  "success": true,
  "handoff_id": "abc123...",
  "archived_previous": 1,
  "message": "Handoff created. Use onboard_session(...) to restore."
}
```

## Restoring Context (Onboarding)

When starting a new session to continue previous work:

### Option A: List Active Handoffs
```python
mcp_pas-server_onboard_session()  # All active
mcp_pas-server_onboard_session(project_id="mcp-pas")  # Filter by project
```

### Option B: Semantic Search
```python
mcp_pas-server_onboard_session(topic="Phase 12 handoff system")
```

### Option C: Restore Specific Handoff
```python
mcp_pas-server_onboard_session(
    handoff_id="<known-id>",
    mark_processed=True  # Consumes the handoff
)
```

## Handoff Lifecycle

```
┌─────────┐     create_handoff     ┌─────────┐
│  NEW    │ ────────────────────▶  │ ACTIVE  │
└─────────┘                        └────┬────┘
                                        │
                ┌───────────────────────┴────────────────────┐
                │                                            │
                ▼                                            ▼
        create_handoff                            onboard_session
        (same session)                         (different session)
                │                                            │
                ▼                                            ▼
         ┌──────────┐                               ┌───────────┐
         │ ARCHIVED │                               │ PROCESSED │
         └──────────┘                               └───────────┘
```

## Best Practices

1. **Be specific** in summaries - future agents have no context
2. **Include file paths** that were modified in `linked_artifacts`
3. **Document blockers** clearly so they're not rediscovered
4. **Use semantic search** to find related past work before starting new work
5. **One active handoff per session** - new handoffs archive previous ones

## Example Handoff

```python
mcp_pas-server_create_handoff(
    session_id="90d87ad4-82b5-4e71-9c07-c3495e1f3b9e",
    summary="Phase 12 Session Handoff/Onboard System complete. Created session_handoffs table with vector embeddings. Added helpers/handoff.py with 5 functions. Added create_handoff and onboard_session tools.",
    next_task="Test onboard_session semantic search. Consider auto-handoff in record_outcome.",
    context={"decisions": ["768-dim embeddings", "archive-on-new behavior"]},
    linked_artifacts="migrations/012_session_handoffs.sql,src/pas/helpers/handoff.py"
)
```
