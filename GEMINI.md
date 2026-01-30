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

---

## Rule 6: Mandatory Sequential Gap Analysis 🔍 (v37)

**Before calling `finalize_session`, you MUST run constructive gap analysis:**

```python
# REQUIRED before finalize_session
mcp_pas-server_prepare_sequential_analysis(session_id="...", top_n=3)
# Process each prompt, then store results
mcp_pas-server_store_sequential_analysis(session_id="...", results="[...]")
# NOW you can finalize
mcp_pas-server_finalize_session(session_id="...")
```

### Why This Matters

| Approach | Mode | Question |
|----------|------|----------|
| **PAS Critique** | Adversarial | "What's wrong?" |
| **Sequential Analysis** | Constructive | "What's missing?" |

Both are needed. Critique finds flaws. Sequential analysis finds gaps.

### The 5-Layer Gap Check

Sequential analysis asks about each layer:
1. **CODE STRUCTURE**: What code changes are needed?
2. **DEPENDENCIES**: What packages/systems are assumed?
3. **DATA FLOW**: What data moves where?
4. **INTERFACES**: What APIs/contracts are affected?
5. **WORKFLOWS**: What user/system flows change?

### Skip Criteria

Only skip sequential analysis if:
- Trivial change (< 10 lines, single file)
- User explicitly says "just do it"
- You document why you're skipping

> **v37 Change**: Sequential gap analysis is now mandatory for PAS planning.

---

## Rule 7: Use Symbol Lookups for Scope Accuracy 🔍 (v38c)

**When `prepare_expansion` returns `suggested_lookups`, call `find_references` BEFORE generating hypotheses.**

### Why This Matters

`suggested_lookups` contains symbols extracted from your goal/parent text that exist in the synced project. Calling `find_references` on these symbols reveals:
- How many places use the symbol
- Which files would be affected by changes
- Accurate scope for your hypotheses

### Workflow

```python
# 1. Call prepare_expansion with project_id
result = mcp_pas-server_prepare_expansion(session_id="...", project_id="mcp-pas")

# 2. If suggested_lookups present, explore them
if result.get("suggested_lookups"):
    for lookup in result["suggested_lookups"]:
        refs = mcp_pas-server_find_references(
            project_id="mcp-pas", 
            symbol_name=lookup["symbol"]
        )
        # Now you know the impact scope

# 3. Generate hypotheses with informed scope
```

### Skip Criteria

- No `suggested_lookups` in response (no symbols found)
- `project_id` not provided to `prepare_expansion`
- Trivial change where scope is obvious

> **TODO**: Integrate into `/pas-planning` workflow for explicit enforcement.

---

## Rule 8: Preflight Enforcement 🛫 (v41)

**When `store_expansion` returns `preflight_warnings`, you MUST address them before proceeding.**

### Warning Types

| Warning | Meaning | Required Action |
|---------|---------|-----------------|
| `missing_schema_check` | SQL operations detected | Call `get_self_awareness()` |
| `missing_find_references` | Symbol lookups suggested | Call `find_references()` |
| `unacknowledged_warnings` | Past failures surfaced | Call `log_conversation()` |

### The `skip_preflight` Escape Hatch

**NEVER use `skip_preflight=True` without explicit user approval.**

This parameter exists for:
- Trivial bug-fix sessions (user-approved)
- Debugging preflight system itself
- Time-critical emergencies

When used, it is logged for outcome correlation - PAS will learn if bypasses correlate with failures.

> **v41 Change**: Preflight checks are now enforced at `store_expansion` time.

---

## Rule 9: Codebase Research Before Hypothesizing 🔍 (v42a)

**Before calling `store_expansion`, you MUST search for existing related functionality.**

### Mandatory Steps

```python
# 1. Call prepare_expansion (will auto-return related_modules now)
result = mcp_pas-server_prepare_expansion(session_id="...", project_id="mcp-pas")

# 2. Review related_modules returned (v42a automated search)
if result.get("related_modules"):
    for module in result["related_modules"]:
        # Study these before hypothesizing
        print(f"Existing: {module['file']} - {module['purpose']}")

# 3. Optionally do deeper search
mcp_pas-server_query_codebase(query="<goal keywords>", project_id="mcp-pas")

# 4. ONLY NOW generate hypotheses that build on existing infrastructure
```

### Why This Matters

Session `49ea0e60` showed that v42 Feature Tracker planning missed `purpose_helpers.py` because:
- Goal keywords didn't match existing code semantically
- Agent skipped `query_codebase` before hypothesizing

### Enforcement

| Layer | Mechanism |
|-------|-----------|
| **Soft** | This rule in GEMINI.md |
| **Hard** | Preflight check: `missing_codebase_research` warning |
| **Auto** | `prepare_expansion` returns `related_modules` from semantic search |

> **v42a Change**: Codebase research is now mandatory before hypothesis generation.
