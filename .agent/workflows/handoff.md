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

## Steps

### 1. Summarize the Session

Prepare a concise summary of:
- What was accomplished
- Key decisions made
- Any blockers or unresolved issues

### 2. Create the Handoff

// turbo
```bash
# If you have an active PAS session, use its ID
# Otherwise, create a minimal session first
```

Call the MCP tool:
```
mcp_pas-server_create_handoff(
    session_id="<active-pas-session-id>",
    summary="<what was done, key decisions, current state>",
    next_task="<suggested next step for future agent>",
    context='{"key_decisions": [...], "blockers": [...]}',
    linked_artifacts="<comma-separated artifact paths>"
)
```

### 3. Verify Creation

The tool returns a `handoff_id`. Future agents can retrieve this with:
```
mcp_pas-server_onboard_session(handoff_id="<id>")
```

## Onboarding (For New Sessions)

When starting a new session to continue previous work:

### Option A: List Active Handoffs
```
mcp_pas-server_onboard_session(project_id="mcp-pas")
```

### Option B: Semantic Search
```
mcp_pas-server_onboard_session(topic="Phase 12 handoff system")
```

### Option C: Specific Handoff
```
mcp_pas-server_onboard_session(handoff_id="<known-id>", mark_processed=true)
```

## Example Handoff

```python
mcp_pas-server_create_handoff(
    session_id="90d87ad4-82b5-4e71-9c07-c3495e1f3b9e",
    summary="Implemented Phase 12 Session Handoff/Onboard System. Created session_handoffs table with vector embeddings for semantic search. Added create_handoff and onboard_session tools to server.py.",
    next_task="Test onboard_session semantic search. Consider integrating with record_outcome for auto-handoff.",
    context='{"decisions": ["Used 768-dim embeddings (e5-base-v2)", "RealDictCursor requires dict key access"]}',
    linked_artifacts="migrations/012_session_handoffs.sql,src/pas/helpers/handoff.py"
)
```

## Best Practices

1. **Be specific** in summaries - future agents have no context
2. **Include file paths** that were modified
3. **Document blockers** clearly so they're not rediscovered
4. **Use semantic search** to find related past work before starting new work
