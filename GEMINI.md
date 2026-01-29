# PAS Project Guidelines

## Rule 1: PAS-Driven Implementation Planning

**Before implementing ANY non-trivial change:**

1. Ask: "Could I miss a dependency or compatibility issue?"
2. If YES → Use PAS to plan first (see `/pas-planning` workflow)
3. If NO → Proceed with direct implementation

### Red Flags Requiring PAS

- Modifying shared data structures
- Changing function signatures
- Cross-file dependencies
- Adding required fields
- Any change you're uncertain about

### Skip Criteria (Direct Implementation OK)

- Single file, < 10 lines
- Clear root cause, obvious fix
- Documentation/comments
- Typo fixes

---

## Rule 2: Log ALL Failures Immediately 🚨

**When ANY of these occur, LOG FIRST, then fix:**

| Failure Type | Example | Log As |
|--------------|---------|--------|
| **Bug discovered** | Tests fail, runtime error | `failure` |
| **Planning gap** | Missed dependency, wrong assumption | `partial` |
| **Type mismatch** | SQL type error, API contract violation | `failure` |
| **Integration issue** | Library doesn't work as expected | `failure` |
| **Silent failure** | Code runs but wrong result | `failure` |

**Logging Flow:**
1. **STOP** - Don't fix it yet
2. **START SESSION** - Create PAS session for the bug
3. **STORE HYPOTHESIS** - Document what went wrong
4. **RECORD OUTCOME** - `failure` with semantic `failure_reason`
5. **NOW FIX** - Implement the fix

```python
# Quick logging pattern
mcp_pas-server_start_reasoning_session(user_goal="Bug: <description>")
mcp_pas-server_store_expansion(h1_text="<root cause>", h1_confidence=0.9)
mcp_pas-server_record_outcome(
    session_id="...",
    outcome="failure",
    failure_reason="<semantic description for future similarity matching>"
)
```

> **Why?** PAS learns from failures via semantic similarity (v17b). Unlogged failures = lost learning.

---

## Rule 3: Verify Before Completing

| Change Type | Verification Method |
|-------------|---------------------|
| Schema changes | `psql` query to verify |
| server.py changes | Restart MCP + test tool call |
| seed scripts | Run script + verify data |

**Never mark complete without empirical evidence.**
