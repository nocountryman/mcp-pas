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

### 2. Verify and Export

```python
# Check governance via Resource
read_resource("pas-server", "pas://governance/mcp-pas")

# Export to markdown
mcp_pas-server_export_roadmap_to_markdown(project_id="mcp-pas")
```

### 3. View Status

```bash
# Or use /roadmap-status workflow
```

---

## ⚠️ DO NOT

- Edit roadmap markdown files directly
- Store phases only in artifacts

**Database is the Single Source of Truth (SSOT).**
