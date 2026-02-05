---
description: DB-first research workflow - store research to DB before files
---

# /research - Store Research Findings

## When to Use
After completing research, before moving to implementation.

## Steps

1. **Store research to DB** (required):
   ```
   store_research(
     project_id="mcp-pas",
     topic="<Research Topic>",
     content="<Full findings in markdown>",
     tags="<topic-tags>"
   )
   ```

2. **If file export needed** (optional):
   ```
   store_research(
     project_id="mcp-pas",
     topic="<Research Topic>",
     content="<Full findings>",
     export_path="/path/to/file.md"
   )
   ```

## Rules
- ❌ Do NOT use `write_to_file` for research outputs
- ✅ Always use `store_research` (DB-first)
- Database is source of truth, files are secondary copies
- Use `pas://artifacts/{project_id}` to search stored research
