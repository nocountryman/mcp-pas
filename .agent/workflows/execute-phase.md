---
description: Execute a roadmap phase with full PAS planning
---

# Execute Phase Workflow

Use this when you want to work on a specific phase from the PAS roadmap.

## Steps

### 1. Find the Phase

```python
# Get all phases for the project
phases = mcp_pas-server_get_roadmap_phases(project_id="mcp-pas")
# OR search by name
```

### 2. Get Phase Context

```python
context = mcp_pas-server_get_phase_context(phase_id="<phase_uuid>")
# Review: description, dependencies, success_criteria, can_activate
```

### 3. Check Dependencies

If `context.dependencies_met == False`:
- Review which dependencies are incomplete
- Either execute those phases first OR get user approval to proceed anyway

### 4. Activate the Phase

```python
result = mcp_pas-server_activate_phase(phase_id="<phase_uuid>")
# Status changes: planned → active
# Returns: context with next_step guidance
```

### 5. Start PAS Planning

Follow the returned `next_step` (typically):
```python
mcp_pas-server_start_reasoning_session(
    user_goal=f"Implement {result['phase_name']}",
    project_id="mcp-pas"
)
```

Then follow `/pas-planning` workflow.

### 6. Execute Implementation Plan

Work through the implementation plan steps, updating status:
```python
mcp_pas-server_update_step_status(step_id="<step_uuid>", status="done")
```

### 7. Update Success Criteria

Mark each criterion as checked when completed:
```python
mcp_pas-server_update_success_criterion(criterion_id="<id>", checked=True)
```

### 8. Complete the Phase

```python
result = mcp_pas-server_complete_phase(
    phase_id="<phase_uuid>",
    notes="Implementation verified with tests"
)
```

If unchecked criteria remain, the tool will error with the list of incomplete items.

---

## Quick Reference

| Tool | Purpose |
|------|---------|
| `get_phase_context` | See what's needed |
| `activate_phase` | Start work |
| `complete_phase` | Finish work |
| `update_step_status` | Track plan progress |
| `update_success_criterion` | Track criteria |
