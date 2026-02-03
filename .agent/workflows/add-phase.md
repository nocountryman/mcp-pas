---
description: Add a new phase to the roadmap database
---

# Add Phase to Roadmap

When user says "add to roadmap" or "new phase", use this workflow.

## Steps

### 1. Create Phase in Database

```python
mcp_pas-server_create_governance_phase(
    project_id="mcp-pas",
    phase_name="Phase X: <Name>",
    description="<Goal description>",
    status="planned"  # or "active" if starting immediately
)
```

### 2. Add Success Criteria (Optional)

```python
mcp_pas-server_add_success_criterion(
    phase_id="<phase_uuid>",
    criterion="Tests pass for new functionality"
)
```

### 3. Add Dependencies (If Needed)

```python
mcp_pas-server_add_phase_dependency(
    phase_id="<new_phase_uuid>",
    depends_on_phase_id="<prerequisite_phase_uuid>"
)
```

### 4. Confirm

```python
mcp_pas-server_export_roadmap_to_markdown(project_id="mcp-pas")
```

---

## ⚠️ DO NOT

- Edit roadmap markdown files directly
- Store phases only in artifacts

**Database is the Single Source of Truth (SSOT).**
