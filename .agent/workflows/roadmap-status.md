---
description: View current roadmap status and phase progress
---

# Roadmap Status

Quick view of all phases and their status.

## Steps

### 1. Export Roadmap

```python
mcp_pas-server_export_roadmap_to_markdown(project_id="mcp-pas")
```

### 2. For Detailed Phase Info

```python
# Get specific phase context
mcp_pas-server_get_phase_context(phase_id="<uuid>")
```

### 3. List All Phases Programmatically

```python
# Get raw phase list
phases = mcp_pas-server_get_project_governance(project_id="mcp-pas")
```

---

## Status Legend

| Status | Meaning |
|--------|---------|
| ✅ complete | Phase done |
| 🔄 active | In progress |
| 📋 planned | Not started |
| 🚫 blocked | Waiting on dependency |
