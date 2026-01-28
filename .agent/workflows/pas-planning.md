---
description: Enforce PAS-driven implementation planning for non-trivial changes
---

# PAS Implementation Gate

## When to Use PAS for Planning

Before implementing ANY change, ask yourself:

**Is this a "flash/haiku" level change?**
- Single file edit
- < 10 lines changed
- No cross-file dependencies
- Clear, obvious fix (typo, missing field, etc.)

**If NO to any of these → USE PAS**

## PAS Planning Flow

1. **Start a reasoning session** with the implementation goal:
   ```
   mcp_pas-server_start_reasoning_session(user_goal="<your implementation goal>")
   ```

2. **Generate hypotheses** about the approach:
   ```
   mcp_pas-server_prepare_expansion(session_id="...")
   mcp_pas-server_store_expansion(h1_text=..., h1_confidence=..., h1_scope=...)
   ```

3. **Critique top hypothesis** to catch dependencies:
   ```
   mcp_pas-server_prepare_critique(node_id="...")
   mcp_pas-server_store_critique(counterargument=..., edge_cases=..., severity_score=...)
   ```

4. **Finalize** to get recommendation:
   ```
   mcp_pas-server_finalize_session(session_id="...")
   ```

5. **Record outcome** after implementation:
   ```
   mcp_pas-server_record_outcome(session_id="...", outcome="success"|"failure")
   ```

## Red Flags That Require PAS

- [ ] Modifying a function that other functions depend on
- [ ] Adding a new field to a data structure
- [ ] Changing API contracts or function signatures
- [ ] Cross-file refactoring
- [ ] Any change where you're unsure of scope
- [ ] Infrastructure changes (DB schema, config, etc.)

## Skip Criteria (OK to implement directly)

- Bug fix with clear root cause and single-file scope
- Documentation updates
- Adding comments
- Typo fixes
- Simple additions that don't change existing behavior
