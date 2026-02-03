---
description: Enforce PAS-driven implementation planning for non-trivial changes
---

# PAS Implementation Gate

## Quick Decision: PAS Required?

**Is this "flash/haiku" level?**
- Single file, < 10 lines
- No cross-file dependencies
- Clear, obvious fix

**If NO to ANY → USE PAS**

---

## Workflow: Single-Phase Work

```
1. start_reasoning_session(user_goal="...")
2. sync_project(project_path="...")  // Ensure DB is current (v53 delta sync)
3. prepare_expansion(session_id="...", project_id="...")
   → Check suggested_lookups, call find_references for each
   → Review related_modules returned
   → **v50: If past_failure_warnings present → MUST log_conversation to acknowledge**
4. store_expansion(h1_text, h1_confidence, h1_scope, h2_text, ...)
5. prepare_critique(node_id="<top hypothesis>")
6. store_critique(counterargument, severity_score, major_flaws, minor_flaws)
7. prepare_critique(session_id="...", mode="sequential")  # Gap analysis
8. store_critique(... sequential gap results ...)
9. finalize_session(session_id="...")
   → HARD BLOCK: score < 0.9 = DO NOT PROCEED
   → If quality_gate.passed = false, deepen or expand more hypotheses
   
9a. **v59: Review Dual Recommendation**
   → If response includes `aspirational` different from `balanced`:
      → Document both in plan's "Alternatives Considered" table
      → If user said "effort doesn't matter", prefer aspirational
   → Include pros/cons for each option

## 🔍 LSP Impact Analysis (v52 Phase 1)
9b. Before creating plan, gather LSP data:
   → Call find_references for key symbols in scope
   → Document affected files in plan's "LSP Impact Analysis" section
   → If callers discovered outside scope → expand scope or document why excluded

## ⛔ HARD BLOCK: Implementation Plan Required
10. Create implementation_plan.md using template BEFORE any code edits
   → Template: `.agent/templates/implementation_plan_template.md`
   → Request user review via notify_user(PathsToReview=[...], BlockedOnUser=True)
   → DO NOT write code until user approves or auto-proceeds
   → The plan is YOUR structured checklist - without it you skip verification steps

10a. **v73 Phase 3: Validate Plan Before Implementation**
   → Call `validate_plan(session_id, plan_text, lsp_impact=<lsp_data_if_available>)`
   → Check response for:
      - `lsp_section_check.has_lsp_section` = true if lsp_impact was provided
      - `missing` list is empty (or only LOW severity)
      - `scope_warnings` addressed or documented
   → If validation fails, update plan before proceeding

## 🔍 Related Files Check (v85)
10b. Before creating ANY new file (code, template, config):
   → Search for related files: `find_by_name(Pattern="*<keyword>*", SearchDirectory="...")`
   → Example: Before creating `research_template.md`, check for `*research*` or `*template*`
   → If related files found:
      - Review their content to understand overlap
      - Decide: consolidate, extend existing, or create new with clear distinction
      - Document decision in implementation plan
   → **Why**: Prevents duplication (e.g., creating `research_template.md` when `research_plan_template.md` exists)

11. AFTER user approval: Execute code changes following plan
12. record_outcome(session_id="...", outcome="...")
```

> **Enforcement Rationale**: The implementation plan is not just user documentation—it's the agent's structured execution checklist. Skipping it leads to missed verification steps, scope drift, and undocumented decisions.


### If Hypotheses Synthesized
```
After synthesize_hypotheses() creates hybrid node:
→ MUST call prepare_critique on hybrid node
→ MUST call store_critique on hybrid node
→ MUST call prepare_critique(mode='sequential') for gap analysis
→ MUST call store_critique with gap analysis results
→ THEN finalize_session
```

> **Why**: Sequential analysis on the final hybrid catches gaps that critique alone misses (e.g., missing __init__.py exports, test import strategy).


---

## Workflow: Multi-Phase Work (ROADMAP)

```
1-8. Same as above but treat as HIGH-LEVEL design
9. If finalize_session shows multi-phase scope:
   → Create ROADMAP using roadmap_template.md
   → Each phase gets SEPARATE PAS session + implementation_plan
10. record_outcome on roadmap session
11. Start new PAS sessions for each phase
```

---

## Workflow: Research Mode

```
1. start_reasoning_session(session_mode="research", user_goal="Research: ...")
2. Explore using query_codebase, search_web, view_file
3. Document findings in research_plan_template.md
4. If research leads to implementation:
   → Create new implementation session (session_mode="implementation")
   → Link research findings in implementation plan
5. record_outcome(session_id="...", outcome="success|partial")
```

---

## Quality Gate Enforcement

| Metric | Threshold | Action if Below |
|--------|-----------|-----------------|
| Score | ≥0.9 | Expand deeper, add hypotheses |
| Gap | ≥0.08 | Explore more diverse alternatives |
| Synthesized? | Must critique | Run full critique on hybrid |

**NEVER** use `skip_quality_gate=True` without user explicit approval.

---

## Templates (v82 Enhanced)

| Template | When to Use | Key Features |
|----------|-------------|---------------|
| [implementation_plan_template.md](file:///.agent/templates/implementation_plan_template.md) | Single-phase work | Nudges, dual-plan section, constraint footer |
| [roadmap_template.md](file:///.agent/templates/roadmap_template.md) | Multi-phase work | Per-phase guidance, mermaid examples |
| [research_template.md](file:///.agent/templates/research_template.md) | Research mode sessions | Architecture, gaps, findings, priority matrix |

### Template Features
- **Agent Context Header**: Role, constraints, mode reminder
- **Section Nudges**: `> [!IMPORTANT]` checks after each section
- **Dual Recommendation**: Balanced vs Aspirational table
- **Constraint Footer**: Known issues and active constraints

---

## Red Flags Requiring PAS

- [ ] Modifying shared data structures
- [ ] Changing function signatures
- [ ] Cross-file dependencies
- [ ] Adding required fields
- [ ] Schema changes
- [ ] Any uncertainty about scope
- [ ] **v50: Creating new helper files (verify imports first!)**
- [ ] PAS session score ≥ 0.9
- [ ] All major critiques from PAS addressed in plan
- [ ] Synthesized hypotheses critiqued (if applicable)
- [ ] **LSP Impact Analysis completed** (find_references on key symbols)
- [ ] N/A sections explicitly marked (not left blank)
- [ ] Verification commands tested/runnable
- [ ] Exact code shown (not descriptions)
- [ ] Sequential gap analysis completed

---

## v50: Pre-Implementation Checks

### Warning Acknowledgment
When `prepare_expansion` returns `past_failure_warnings`:
```python
for warning in result.get("past_failure_warnings", []):
    mcp_pas-server_log_conversation(
        session_id="...", log_type="context",
        raw_text=f"ACKNOWLEDGED: {warning['pattern']} - Mitigation: [plan]"
    )
# THEN call store_expansion
```

### Import Verification (New Helper Files)
Before creating ANY new helper file, verify ALL imports:
```bash
# For each function you plan to import:
grep -rn "def function_name" src/pas/
# OR use find_references
mcp_pas-server_find_references(project_id="...", symbol_name="function_name")
```

**Checklist for new helpers:**
- [ ] List all functions to import
- [ ] Verify location of each with grep_search
- [ ] Check for circular import risks (no imports FROM server.py into helpers)
- [ ] Document verified imports in implementation plan

---

## Workflow Refinements

### Best Practices
- **Log failures immediately**: Call `record_outcome(outcome="failure", failure_reason="...")` BEFORE fixing bugs
- **Restart after config changes**: Server must restart to pick up `config.yaml` changes
- **Update workflows if relevant**: Check if changes affect `.agent/workflows/` or `.agent/skills/` files
- **Update CHANGELOG**: Add version entry after significant changes

### Code Quality (Python) 🐍
**Apply these thresholds to new/modified code:**

| Metric | Target | Fail At |
|--------|--------|---------|
| Cyclomatic Complexity | ≤10 | >15 |
| Function Length | ≤50 lines | >80 |
| File Length | ≤500 lines | >800 |

**Verification**: `radon cc [file.py] -s -a`

**Principles** (nudge LLM toward clean code):
- **SRP**: Each function does ONE thing - if you need "and", split it
- **DRY**: 3+ similar blocks → extract to helper
- **Naming**: Verb for functions (`get_user`), noun for variables (`user_id`)

### After Implementation
1. Run tests to verify changes
   ```bash
   # REQUIRED for AMD/ROCm: Include HSA workaround
   HSA_OVERRIDE_GFX_VERSION=10.3.0 pytest tests/test_server.py -v
   ```
2. Check if any skills/slashcommands need updates
3. Call `record_outcome` with final result
4. Commit with meaningful message referencing PAS session ID

---

## Rule 11: Raw Input Logging 📝 (v44)

**When user provides a prompt that initiates a session:**

1. Pass `raw_input` to `start_reasoning_session` with VERBATIM text
2. Store summarized goal separately in `user_goal`
3. If blocked by preflight, either:
   - Add `raw_input` parameter with user's exact words
   - Use `skip_raw_input_check=True` for LLM-initiated sessions

```python
# User-initiated session (REQUIRED: raw_input)
mcp_pas-server_start_reasoning_session(
    user_goal="Implement feature X for user",
    raw_input="can you implement feature X? it should do Y and Z"
)

# LLM-initiated session (skip check)
mcp_pas-server_start_reasoning_session(
    user_goal="Refactor internal module",
    skip_raw_input_check=True
)
```

**Hard enforced**: Sessions with user-initiated keywords ('build', 'implement', 'design', etc.) will FAIL without `raw_input`.

