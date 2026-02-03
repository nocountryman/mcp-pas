---
description: Execute a roadmap phase with full PAS planning
---

# Execute Phase Workflow

Use this when you want to work on a specific phase from the PAS roadmap.

## Steps

### 1. View Phase Status

```python
# Use Resource to see governance hierarchy
read_resource("pas-server", "pas://governance/mcp-pas")
# Shows: phases, active count, artifacts
```

### 2. Get Phase Context and Activate

```python
# Activate phase (changes planned → active)
result = mcp_pas-server_activate_phase(phase_id="<phase_uuid>")
# Returns: execution context with dependencies and guidance
```

### 3. Start PAS Planning

Follow the returned `next_step` (typically):
```python
mcp_pas-server_start_reasoning_session(
    user_goal=f"Implement {result['phase_name']}",
    project_id="mcp-pas"
)
```

Then follow `/pas-planning` workflow.

### 4. Execute Implementation Plan

Work through the implementation plan steps per the `/pas-planning` workflow.

### 5. Complete the Phase

```python
result = mcp_pas-server_complete_phase(
    phase_id="<phase_uuid>",
    notes="Implementation verified with tests"
)
```

---

## Quick Reference

| Action | Tool/Resource |
|--------|---------------|
| View status | `pas://governance/{project_id}` Resource |
| Activate | `activate_phase` |
| Complete | `complete_phase` |
| View artifacts | `pas://artifacts/{project_id}` Resource |
