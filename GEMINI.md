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

## Rule 2: Log Failures

Every bug discovered during implementation MUST be logged:

```python
mcp_pas-server_record_outcome(
    session_id="...",
    outcome="failure",
    failure_reason="<semantic description of what went wrong>"
)
```

This enables PAS self-learning via v17b semantic similarity.
