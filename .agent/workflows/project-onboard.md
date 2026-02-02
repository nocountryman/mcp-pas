---
description: Onboard a new project by discovering constraints through interview
---

# Project Onboarding via Constraint Discovery

## When to Use

- Setting up PAS for a **new project**
- Project has no `GEMINI.md` or minimal constraints defined
- Want to capture user's coding philosophy, environment, and quality preferences

---

## Workflow

### 1. Start Constraint Discovery Session

```python
mcp_pas-server_start_reasoning_session(
    user_goal="Discover project constraints for [project-name]",
    session_mode="constraint_discovery",
    project_id="my-project",
    project_path="/path/to/project",
    raw_input="<user's request>"
)
```

> **Required for GEMINI.md auto-export**: `project_id` and `project_path` must be provided.

---

### 2. Sync Project (First Time)

```python
mcp_pas-server_sync_project(
    project_path="/path/to/project",
    project_id="my-project"
)
```

> This indexes the codebase and enables LSP lookups.

---

### 3. Run Interview Loop

```python
# Generate constraint discovery questions
mcp_pas-server_identify_gaps(session_id="...")

# Get and answer questions one at a time
while True:
    q = mcp_pas-server_get_next_question(session_id="...")
    if q.get("is_complete"):
        break
    
    # Present question to user, get answer
    answer = "A"  # User's choice
    
    mcp_pas-server_submit_answer(
        session_id="...",
        question_id=q["question"]["id"],
        answer=answer
    )
```

**Sample Questions:**
- MVP philosophy (ship fast vs polished)
- Quality gate strictness (block vs warn vs flexible)
- Environment setup (venv activation pattern)

---

### 4. Complete Interview → Constraints Created + Auto-Exported

```python
result = mcp_pas-server_check_interview_complete(session_id="...")

# Returns:
# {
#   "is_complete": true,
#   "constraints_created": 3,  # v76: Auto-mapped from answers
#   "gemini_exported": true,   # v76: Auto-written to GEMINI.md
#   "latent_traits": [{"trait": "RISK_AVERSE", ...}]
# }
```

> **Automatic**: If `project_path` was provided in step 1, constraints are auto-exported to `GEMINI.md`.

---

### 5. (Optional) Manual GEMINI.md Export

Only needed if auto-export failed or you want to re-export:

```python
# Get export content
result = mcp_pas-server_sync_gemini_constraints(
    project_id="my-project",
    project_path="/path/to/project",
    direction="db_to_file"
)

# Write to file
mcp_pas-server_write_gemini_export(
    project_path="/path/to/project",
    content=result["export_content"]
)
```

---

### 6. Record Outcome

```python
mcp_pas-server_record_outcome(
    session_id="...",
    outcome="success",
    notes="Created N constraints for project"
)
```

---

## Constraint Categories

| Category | Examples |
|----------|----------|
| **philosophy** | `no_mvp`, `quality_gate_required`, `pas_before_changes` |
| **environment** | `terminal_env_activation`, `venv_path` |
| **quality** | `verify_before_completing`, `import_verification_required` |

---

## Enforcement Levels

| Level | Behavior |
|-------|----------|
| `block` | Hard stop - cannot proceed without compliance |
| `warn` | Advisory - surface in preflight, allow override |

---

## What Gets Created

After interview completion:
1. **`project_constraints`** table entries with:
   - `constraint_type`: philosophy/environment/quality
   - `key`: e.g., `no_mvp`
   - `value`: e.g., `true`
   - `enforcement`: block/warn
   - `meta.priority`: must_have/nice_to_have
   - `meta.source_session`: PAS session ID

2. **Detected traits** (e.g., `RISK_AVERSE`, `QUALITY_FOCUSED`) stored in session context

3. **GEMINI.md** section (if exported) with all constraints documented

---

## Conflict Resolution (DB ↔ GEMINI.md)

When constraints exist in both DB and GEMINI.md with different values:

### 1. Detect Drift

```python
# Extract constraints from GEMINI.md first
sync_result = mcp_pas-server_sync_gemini_constraints(
    project_id="my-project",
    project_path="/path/to/project",
    direction="detect_only"
)
# Process extraction_prompt with LLM to get file_constraints

# Then check for drift
drift_result = mcp_pas-server_detect_constraint_drift(
    project_id="my-project",
    constraints_json='[{"key": "no_mvp", "value": true, ...}]'
)
```

### 2. Review Drifts

If `drift_result["drift_detected"]` is true:
```json
{
  "drifts": [
    {"key": "no_mvp", "drift_score": 0.23, 
     "db_value": false, "file_value": true}
  ],
  "resolution_prompt": "Which source should be authoritative?..."
}
```

### 3. Resolve Conflict

```python
# Option A: GEMINI.md wins (file → DB)
mcp_pas-server_resolve_constraint_drift(
    project_id="my-project",
    project_path="/path/to/project",
    resolution="file_to_db",
    file_constraints_json='[...]'  # from extraction
)

# Option B: DB wins (DB → file)
mcp_pas-server_resolve_constraint_drift(
    project_id="my-project",
    project_path="/path/to/project",
    resolution="db_to_file"
)

# Option C: Skip (manual resolution)
mcp_pas-server_resolve_constraint_drift(
    project_id="my-project",
    project_path="/path/to/project",
    resolution="skip"
)
```

---

## Quick Start

```python
# One-liner to start onboarding
mcp_pas-server_start_reasoning_session(
    user_goal="Onboard project: my-app",
    session_mode="constraint_discovery",
    project_id="my-app",
    project_path="/path/to/my-app"
)
```

Then follow the interview prompts! Constraints auto-export to `GEMINI.md` on completion.
