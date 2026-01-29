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

---

## Rule 4: Quality Gate for New Features 🚦

**For NEW FEATURES or LARGE WORK, do NOT proceed if decision quality is low.**

### Thresholds

| Work Type | Required Gap | Quality Level |
|-----------|--------------|---------------|
| New feature | ≥ 0.1 | High |
| Large refactor | ≥ 0.1 | High |
| Schema change | ≥ 0.1 | High |
| Bug fix | ≥ 0.05 | Medium |
| Small change | Any | Low OK |

### When Quality is LOW

If `finalize_session` returns `decision_quality: "low"` for a new feature:

1. **DO NOT PROCEED** with implementation
2. **Follow PAS suggestion**: e.g., "expand_alternatives"
3. **Generate more hypotheses** until gap ≥ 0.1
4. **Re-finalize** and verify quality is now HIGH

```python
# Check quality before proceeding
result = mcp_pas-server_finalize_session(session_id="...")
if result["decision_quality"] == "low" and is_new_feature:
    # DO NOT IMPLEMENT - need more hypotheses
    # Follow result["deepen_suggestions"]
```

### Why This Matters

- Low quality = hypotheses too similar or untested
- Proceeding anyway leads to preventable failures
- PAS learned this from the v22 duplication bug

> **Mantra**: "If quality is low, more thinking is needed."

---

## Rule 5: Quality Gate Enforcement 🚦 (v33)

**If `finalize_session` returns `[UNVERIFIED]` prefix in recommendation:**

1. **DO NOT** present as final answer
2. **MUST** call `prepare_expansion` and deepen
3. **Re-finalize** until `quality_gate.passed: true`

### When to Use `skip_quality_gate=True`

Only use this escape hatch if:
- User explicitly requests early/partial result
- Problem is inherently low-confidence (subjective decisions)
- You explain why in your response

### If Proceeding with [UNVERIFIED]

You **MUST** explain why you are proceeding with an unverified recommendation:
```
"Note: This recommendation has not passed the quality gate 
(score: X, gap: Y). Proceeding because [reason]."
```

> **v33 Change**: Quality gate is now enforced by default (opt-out, not opt-in).
