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

---

## Rule 10: Roadmap vs Implementation Plan 📋 (v43)

**Multi-phase work = ROADMAP first, then per-phase implementation plans.**

### Decision Criteria

| Criteria | Document Type |
|----------|---------------|
| Single phase, actionable changes | Implementation Plan |
| Multiple phases, cross-cutting concerns | Roadmap → then per-phase plans |

### Roadmap Requirements

1. **PAS session with score ≥0.9** for overall design
2. **Diagrams** (mermaid) for architecture
3. **Reasoning context** - new LLM session must understand without conversation history
4. **Per-phase breakdown** - each phase gets own PAS session + implementation plan

### Implementation Plan Requirements

1. **Score ≥0.9** - HARD BLOCK if below
2. **Synthesized hypotheses MUST be critiqued** - not just created
3. **Exact code changes** - not descriptions
4. **Runnable verification** - copy-paste commands

### Templates

- Roadmap: `.agent/templates/roadmap_template.md`
- Implementation: `.agent/templates/implementation_plan_template.md`

> **v43 Change**: Roadmap/Implementation distinction is now enforced with templates.

---

## Rule 11: Mandatory Warning Acknowledgment 🛑 (v50)

**When `prepare_expansion` returns `past_failure_warnings`, you MUST acknowledge them.**

### The Problem (Phase 10 Bug)

In Phase 10, PAS warned about `SCOPE_BOUNDARY_CROSSING` but I ignored it and proceeded to `store_expansion`, resulting in 3 import errors that required debugging.

### Required Action

```python
# 1. Check for warnings in prepare_expansion result
result = mcp_pas-server_prepare_expansion(session_id="...", project_id="mcp-pas")

# 2. If past_failure_warnings present, MUST acknowledge
if result.get("past_failure_warnings"):
    for warning in result["past_failure_warnings"]:
        # Log acknowledgment before proceeding
        mcp_pas-server_log_conversation(
            session_id="...",
            log_type="context",
            raw_text=f"ACKNOWLEDGED WARNING: {warning['pattern']} - {warning['warning']}. Mitigation: [your plan here]"
        )

# 3. ONLY NOW can you call store_expansion
```

### Why This Matters

- Advisory warnings are useless if ignored
- Logging forces conscious decision-making
- PAS learns from acknowledged vs. ignored warnings

### Enforcement

| Level | Mechanism |
|-------|-----------|
| **Soft** | This rule in GEMINI.md |
| **Medium** | `preflight_warnings.unacknowledged_warnings` |
| **Hard** | Consider blocking `store_expansion` if `past_failure_warnings` not logged |

> **v50 Change**: Warning acknowledgment is now mandatory before hypothesis generation.

---

## Rule 12: Post-Synthesis Critique 🔄 (v50)

**When you call `synthesize_hypotheses`, you MUST critique the hybrid node before recording outcome.**

### The Problem (Phase 10 Bug)

I called `synthesize_hypotheses` which created a hybrid node (score 0.96), then immediately called `record_outcome` without validating the synthesis.

### Required Synthesis Flow

```mermaid
graph LR
    A[synthesize_hypotheses] --> B[prepare_critique<br/>hybrid_node]
    B --> C[store_critique]
    C --> D[prepare_sequential_analysis]
    D --> E[store_sequential_analysis]
    E --> F[finalize_session]
    F --> G[record_outcome]
```

```python
# Correct synthesis workflow
result = mcp_pas-server_synthesize_hypotheses(session_id="...", node_ids=[...])
hybrid_node_id = result["hybrid_node"]["node_id"]

# MUST critique the hybrid
mcp_pas-server_prepare_critique(node_id=hybrid_node_id)
mcp_pas-server_store_critique(node_id=hybrid_node_id, ...)

# MUST run gap analysis
mcp_pas-server_prepare_sequential_analysis(session_id="...")
mcp_pas-server_store_sequential_analysis(session_id="...", results="[...]")

# Re-finalize
mcp_pas-server_finalize_session(session_id="...")

# NOW record outcome
mcp_pas-server_record_outcome(session_id="...", outcome="success")
```

### Why This Matters

- Synthesis combines hypotheses but doesn't validate the combination
- Hybrid node inherits scores but may have new emergent flaws
- Uncritiqued synthesis = untested assumption

### Enforcement

The `synthesize_hypotheses` response already includes:
```json
"next_step": "Critique the hybrid hypothesis. Call prepare_critique(node_id='...')"
```

**You MUST follow this instruction.**

> **v50 Change**: Post-synthesis critique is now explicitly required. Skipping invalidates the session.

---

## Rule 13: Import Verification Before New Files 📁 (v50)

**Before creating a new helper file, verify import paths with `grep_search` for ALL functions you plan to import.**

### The Problem (Phase 10 Bug)

Created `helpers/critique.py` with incorrect imports:
- `from pas.helpers.embedding import get_embedding` ❌ (should be `pas.utils`)
- `from pas.db import get_db_connection` ❌ (should be `pas.utils`)
- `from pas.helpers.reasoning import _search_relevant_failures` ❌ (creates circular import)

### Required Verification

```python
# Before writing ANY import statement, verify the function location
mcp_pas-server_find_references(project_id="mcp-pas", symbol_name="get_embedding")
# OR
grep_search(Query="def get_embedding", SearchPath="/path/to/project")
```

### Checklist for New Helper Files

Before creating a new helper file:
- [ ] List all functions you plan to import
- [ ] For EACH function, run `grep_search` or `find_references` to verify location
- [ ] Check for circular import risks (don't import from server.py into helpers)
- [ ] Document verified imports in implementation plan

> **v50 Change**: Import verification is now mandatory before creating new files.

---

## Rule 14: Terminal Environment for Agent Commands 🖥️

**ALL `run_command` tool calls MUST include venv activation and environment loading.**

### The Problem

Agent `run_command` calls run in isolated subprocesses that:
- Do NOT inherit `~/.bashrc` settings
- Do NOT auto-activate virtualenvs
- Do NOT load `.env` files via direnv
- Do NOT have access to `DATABASE_URL` or other env vars

### Required Command Pattern

```bash
# ALWAYS prefix commands with activation + env loading
source .venv312/bin/activate && set -a && source .env && set +a && <your_command>

# Example: Running tests
source .venv312/bin/activate && set -a && source .env && set +a && pytest tests/

# Example: Database queries
source .venv312/bin/activate && set -a && source .env && set +a && psql "$DATABASE_URL" -c "SELECT 1"

# Example: Python scripts
source .venv312/bin/activate && set -a && source .env && set +a && python -m pas.server
```

### Quick Reference

| What | Value |
|------|-------|
| **Venv path** | `.venv312/bin/activate` |
| **Env file** | `.env` |
| **Python** | `.venv312/bin/python` |
| **DATABASE_URL** | `postgresql://mcp_admin:12345@localhost:5432/mcp_pas` |

### Alternative: Absolute Paths

When activation isn't needed:
```bash
.venv312/bin/python -m pytest tests/
.venv312/bin/python -c "from pas import server; print('ok')"
```

> **Platform Constraint**: This is a fundamental limitation of Antigravity's subprocess isolation. No IDE setting can change this.

