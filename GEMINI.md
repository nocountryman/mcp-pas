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

---

## Rule 15: Handoffs Are User-Initiated Only 📤 (v87)

**NEVER call `create_handoff` unless user explicitly invokes `/handoff`.**

### The Problem (Feb 2, 2026)

Agent called `create_handoff` mid-session to "capture context", which:
- Archived the previous valid handoff
- Summary described future work (wrong - should be past tense)
- Not an actual session end

### Enforcement

The tool has a **hard gate**:
```python
if not user_initiated:
    return {"error": "Handoffs must be user-initiated. Use /handoff workflow."}
```

### Correct Usage

Only via `/handoff` workflow, which sets `user_initiated=True`:
```python
mcp_pas-server_create_handoff(
    project_id="mcp-pas",
    summary="<PAST TENSE: what was done>",
    user_initiated=True  # Set by workflow, not agent
)
```

### Why This Matters

- Handoffs are for **session continuity to future agents**
- Not mid-session bookmarks
- Autonomous calls corrupt the handoff history

> **v87 Change**: `create_handoff` now requires `user_initiated=True` (set by /handoff workflow).

---

> Auto-generated from database. Last synced: 2026-02-02T00:31:17.284385

### Philosophy Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
| `allow_mvp` | `{'value': True, 'priority': 'nice_to_have', 'source_answer': 'A', 'source_session': '7154bf8e-30cc-4120-bea3-47e4e5ca5ccf', 'source_dimension': '197927c5-c9b9-4b6f-bd1e-e8ee92ee0a41'}` | warn |
| `balanced_quality` | `{'value': True, 'priority': 'nice_to_have', 'source_answer': 'C', 'source_session': '7154bf8e-30cc-4120-bea3-47e4e5ca5ccf', 'source_dimension': '197927c5-c9b9-4b6f-bd1e-e8ee92ee0a41'}` | warn |
| `codebase_research_required` | `True` | warn |
| `code_quality` | `production_grade` | warn |
| `dual_plan` | `True` | warn |
| `log_failures_immediately` | `True` | warn |
| `no_mvp` | `{'value': True, 'priority': 'nice_to_have', 'source_answer': 'B', 'source_session': '7154bf8e-30cc-4120-bea3-47e4e5ca5ccf', 'source_dimension': '197927c5-c9b9-4b6f-bd1e-e8ee92ee0a41'}` | warn |
| `pas_before_changes` | `True` | warn |
| `post_synthesis_critique_required` | `True` | warn |
| `preflight_enforcement` | `True` | warn |
| `quality_gate_required` | `True` | block |
| `quality_gate_threshold` | `0.9` | block |
| `roadmap_vs_plan_distinction` | `True` | warn |
| `sequential_analysis_required` | `True` | warn |
| `symbol_lookups_required` | `True` | warn |
| `warning_acknowledgment_required` | `True` | warn |

### Environment Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
| `terminal_env_activation` | `source .venv312/bin/activate && set -a && source .env && set +a` | block |

### Quality Constraints

| Key | Value | Enforcement |
|-----|-------|-------------|
| `db_first_research` | `True` | warn |
| `definition_of_done` | `{'value': 'full_dod'}` | warn |
| `import_verification_required` | `True` | warn |
| `verify_before_completing` | `True` | warn |

---

## Rule 16: DB-First Research Workflow 📊 (v92)

**Research outputs MUST be stored in DB first, then exported to files.**

### Correct Flow
```
Research → DB (scope_content / artifact) → Export to markdown if needed
```

### Incorrect Flow
```
Research → markdown file → copy to DB ❌
```

### Why This Matters
- DB is the Single Source of Truth (SSOT)
- Files can be corrupted or lost
- DB has backup, audit trail, versioning
- Export tools can regenerate files from DB

### Enforcement
- Phase `scope_content` stores research summary
- `store_governance_artifact` stores full content
- Export tools generate files from DB data

---

## Rule 17: Schema Verification Before DB Queries 🗄️ (v103)

**Before writing ANY SQL query, verify table/column names via `information_schema` or `pas://health`.**

### The Problem (Phase 18 Bug)

Implemented primitives with assumed column names that didn't exist:
- `user_goal` → actual: `goal`
- `status` → actual: `state`
- `constraint_value` → actual: `constraint_data`
- `enforcement` → actual: `enforcement_level`
- `effective_weight` → actual: `scientific_weight`
- `psychology_laws` → actual: `scientific_laws`

**Root cause**: Agent assumed schema from memory instead of querying.

### Required Verification

```python
# BEFORE writing any SQL query, verify the schema:

# Option 1: Read pas://health resource
read_resource(ServerName="pas-server", Uri="pas://health")

# Option 2: Use psql directly
psql "$DATABASE_URL" -c "\d table_name"

# Option 3: Query information_schema
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'your_table';
```

### Checklist for DB Query Code

Before writing code that queries a database:
- [ ] List all tables being queried
- [ ] For EACH table, verify columns via `\d table_name` or `information_schema`
- [ ] Check column types match expected usage
- [ ] Document verified schema in implementation plan

### pas://health Resource

Phase 18 includes a health check resource that validates all primitives:
```
pas://health → returns validation status for all DB queries
```

### Enforcement

| Level | Mechanism |
|-------|-----------|
| **Soft** | This rule in GEMINI.md |
| **Medium** | Preflight check: `missing_schema_check` warning |
| **Hard** | `pas://health` resource for runtime validation |

> **v103 Change**: Schema verification is now mandatory before writing DB query code.

---

## Rule 18: Enable Auto-Sync for New Projects 🔄 (Phase 11)

**When onboarding a new project, ALWAYS enable auto_sync for 100% data availability.**

### The Problem

`find_references` and other codebase tools return empty/stale data when:
- Project not synced (`sync_project` never called)
- Files changed while MCP server was off
- Files changed but watcher not running

### The Solution: Auto-Sync

```python
# One-time setup after sync_project
mcp_pas-server_enable_auto_sync(project_id="your-project", enabled=True)
```

This persists in the database. On every server restart:
1. **Delta sync** - Catches changes made while server was off (mtime comparison)
2. **Start watcher** - inotify for real-time updates

### Current Configuration

| Project | auto_sync |
|---------|-----------|
| `mcp-pas` | ✅ Enabled |

### Enforcement

- Add to `/project-onboard` workflow as final step
- Agent should prompt for enable_auto_sync after first sync_project

### Graph Data Available

The `symbol_references` table stores caller→callee relationships:

| Column | Purpose |
|--------|---------|
| `source_symbol` | The calling symbol |
| `target_symbol` | The called symbol |
| `relation_type` | `call`, `reference`, `import` |
| `source_file` / `target_file` | File paths |

**Note**: Phase 25 (RepoGraph Integration) is planned for full graph visualization.

> **Phase 11 Change**: Auto-sync now runs automatically on server startup for enabled projects.

---

## Rule 19: Primitive-Workflow Sync ↔️ (Phase 12)

**When modifying MCP primitives OR workflows, check for cross-updates.**

### The Problem

Changes to primitives and workflows are interdependent:
- New `@mcp.tool()` may need a workflow update (e.g., add to `/pas-planning`)
- Workflow changes may assume primitives that don't exist
- Neither triggers automatic check for the other

### Required Cross-Check

Before finalizing any change that touches:

| You Change... | Also Check... |
|---------------|---------------|
| `@mcp.tool()` / `@mcp.resource()` | Related workflows in `.agent/workflows/` |
| `.agent/workflows/*.md` | Referenced primitives exist |
| Either | GEMINI.md rule updates needed |

### Checklist

```markdown
- [ ] If adding MCP tool: Is it documented in relevant workflow?
- [ ] If modifying workflow: Do referenced tools exist?
- [ ] Does this change require a new GEMINI.md rule?
- [ ] Query `pas://primitives/{project_id}` to verify
```

### Enforcement

| Level | Mechanism |
|-------|-----------|
| **Soft** | This rule in GEMINI.md |
| **Hard** | Preflight check: `primitive_workflow_sync` warning |

> **Phase 12 Change**: Primitives are now indexed in `mcp_primitives` table for cross-reference.

