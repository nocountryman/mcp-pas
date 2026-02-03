---
description: Create a session handoff to preserve context for future agents
---

# Session Handoff Workflow

> ⚠️ **USER-INITIATED ONLY**: This workflow requires explicit user invocation.
> Never call `create_handoff` autonomously - the tool will block without `user_initiated=True`.

Use this at the **end** of a working session to preserve context for future agents.

## When to Use

- End of complex multi-step work
- Before switching to a different task
- When work is partially complete and needs continuation
- To document key decisions and context

## When NOT to Use

- ❌ Mid-session "just in case" bookmarks
- ❌ Before completing the current task
- ❌ Autonomously (without user explicitly calling /handoff)

## Default Behavior (Per-Project Singleton)

Each project has **at most one active handoff**. Creating a new handoff archives any previous active handoff for that project.

// turbo
```python
# Upsert handoff for current project
# NOTE: user_initiated=True is REQUIRED - tool blocks without it
mcp_pas-server_create_handoff(
    project_id="mcp-pas",
    summary="<what was done, key decisions, current state>",
    next_task="<suggested next step for future agent>",
    context={"key_decisions": [...], "blockers": [...]},
    linked_artifacts="file1.py,file2.sql",
    user_initiated=True  # REQUIRED - confirms user invoked /handoff
)
```

**Note:** `session_id` is auto-detected from the most recent active PAS session for the project. Only provide it explicitly if you want to link to a specific session.

## Response

```json
{
  "success": true,
  "handoff_id": "abc123...",
  "archived_previous": 1,
  "auto_detected_session": true,
  "message": "Handoff created for project 'mcp-pas'. Auto-linked to session abc123... Use onboard_session(project_id='mcp-pas') to restore."
}
```

## Best Practices

1. **Be specific** in summaries - future agents have no context
2. **Include file paths** that were modified in `linked_artifacts`
3. **Document blockers** clearly so they're not rediscovered
4. **One active handoff per project** - new handoffs archive previous ones
5. **Summary must be PAST TENSE** - describe what WAS done, not what WILL be done

## Example

```python
mcp_pas-server_create_handoff(
    project_id="mcp-pas",
    summary="Phase 12 Handoff System refined. Per-project singleton design. Auto-detect session. Simplified /onboard workflow.",
    next_task="Verify MCP restart picks up changes. Test singleton enforcement.",
    context={"decisions": ["per-project singleton", "session auto-detect"]},
    linked_artifacts="src/pas/helpers/handoff.py,src/pas/server.py",
    user_initiated=True
)
```
