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
2. prepare_expansion(session_id="...", project_id="...")
   → Check suggested_lookups, call find_references for each
   → Review related_modules returned
3. store_expansion(h1_text, h1_confidence, h1_scope, h2_text, ...)
4. prepare_critique(node_id="<top hypothesis>")
5. store_critique(counterargument, severity_score, major_flaws, minor_flaws)
6. prepare_sequential_analysis(session_id="...")
7. store_sequential_analysis(session_id="...", results="[...]")
8. finalize_session(session_id="...")
   → HARD BLOCK: score < 0.9 = DO NOT PROCEED
   → If quality_gate.passed = false, deepen or expand more hypotheses
9. Create implementation_plan.md using template
10. record_outcome(session_id="...", outcome="...")
```

### If Hypotheses Synthesized
```
After synthesize_hypotheses() creates hybrid node:
→ MUST call prepare_critique on hybrid node
→ MUST call store_critique on hybrid node
→ THEN finalize_session
```

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

## Quality Gate Enforcement

| Metric | Threshold | Action if Below |
|--------|-----------|-----------------|
| Score | ≥0.9 | Expand deeper, add hypotheses |
| Gap | ≥0.08 | Explore more diverse alternatives |
| Synthesized? | Must critique | Run full critique on hybrid |

**NEVER** use `skip_quality_gate=True` without user explicit approval.

---

## Templates

- **Single-phase**: `.agent/templates/implementation_plan_template.md`
- **Multi-phase**: `.agent/templates/roadmap_template.md`

---

## Red Flags Requiring PAS

- [ ] Modifying shared data structures
- [ ] Changing function signatures
- [ ] Cross-file dependencies
- [ ] Adding required fields
- [ ] Schema changes
- [ ] Any uncertainty about scope

---

## Workflow Refinements

### Best Practices
- **Log failures immediately**: Call `record_outcome(outcome="failure", failure_reason="...")` BEFORE fixing bugs
- **Restart after config changes**: Server must restart to pick up `config.yaml` changes
- **Update workflows if relevant**: Check if changes affect `.agent/workflows/` or `.agent/skills/` files
- **Update CHANGELOG**: Add version entry after significant changes

### After Implementation
1. Run tests to verify changes
2. Check if any skills/slashcommands need updates
3. Call `record_outcome` with final result
4. Commit with meaningful message referencing PAS session ID
