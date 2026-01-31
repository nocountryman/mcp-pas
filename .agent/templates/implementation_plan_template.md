# Implementation Plan Template (REQUIRED for Single-Phase Work)

> Use this template for single-phase, actionable implementation.
> MUST be generated via complete PAS workflow with score ≥0.9.

---

## REQUIRED SECTIONS

### 1. PAS Session Evidence (MANDATORY)
```markdown
## PAS Reasoning Summary

**Session ID**: `<uuid>`
**Goal**: [The user_goal passed to start_reasoning_session]

### Hypotheses Evaluated
| ID | Content (truncated) | Score | Critiqued? |
|----|---------------------|-------|-----------|
| h1 | [...] | 0.XX | YES/NO |
| h2 | [...] | 0.XX | YES/NO |

### Winning Hypothesis
**Node ID**: `<uuid>`
**Final Score**: ≥0.9 (REQUIRED - do NOT proceed if below)
**Decision Quality**: HIGH (REQUIRED)
**Gap**: ≥0.08 (REQUIRED)

### Key Critiques & How Addressed
- Critique 1: [Issue] → [How addressed in plan]
- Critique 2: [Issue] → [How addressed in plan]

### Sequential Gap Analysis Results
- Gaps identified: [list]
- Addressed in plan: [yes/no for each]
```

### 2. Scope Declaration
```markdown
## Scope

**Files Modified**:
- `[layer] path/to/file.py` - [what changes]

**Files Created**:
- `[layer] path/to/new.py` - [purpose]
> N/A if no new files created.

**Files Deleted**:
- `path/to/old.py` - [why]
> N/A if no files deleted.
```

### 3. Detailed Changes
```markdown
## Changes

### [Component/File Name]

#### [Function/Section]

**Before** (if modifying):
```language
[exact current code]
```

**After**:
```language
[exact new code with diff markers if helpful]
```

**Rationale**: [Why this change - link to PAS critique if relevant]
```

### 5. Verification Plan (adapt for your stack)
```markdown
## Verification

### Automated Tests
```bash
# Python
pytest tests/

# JavaScript
npm test

# Or your project's test command
[exact command to run]
```

Expected output: [what success looks like]

### Manual Verification
1. [Step 1]
2. [Step 2]
```

### 6. Project Structure (if creating/moving files)
```markdown
## Project Structure

> **For NEW projects**: Use best practices (Python: `src/` layout)
> **For EXISTING projects**: Align with current layout, or document transition if restructuring

[Target structure if applicable]
```

### 7. Environment Context (if pip/python commands)
```markdown
## Environment

> **IMPORTANT**: Specify the venv for ALL pip/python commands.
> Bare `pip` or `python` may use system interpreter, not project venv.

| Item | Value |
|------|-------|
| **Venv Path** | `[FILL: /path/to/.venv/]` |
| **pip command** | `[VENV]/bin/pip install ...` |
| **python command** | `[VENV]/bin/python -m ...` |
```

### 8. Code Quality Requirements (Python)
```markdown
## Code Quality

> Apply these thresholds to new/modified code. Document exceptions.

### Thresholds
| Metric | Target | Warn | Fail |
|--------|--------|------|------|
| Cyclomatic Complexity | ≤10 | 11-15 | >15 |
| Function Length | ≤50 lines | 51-80 | >80 |
| File Length | ≤500 lines | 501-800 | >800 |

### Principles
- **SRP**: Each function does ONE thing
- **DRY**: Extract repeated patterns (3+ occurrences)
- **Naming**: snake_case for functions/variables, CamelCase for classes

### Verification Command
```bash
# Run before committing
radon cc [file.py] -s -a
```

### Exceptions
> Document any justified exceptions (state machines, parsers, etc.):
- [ ] [Function name]: [Reason for exception]
```

### 7. Workflow/Skills Updates (if applicable)
```markdown
## Workflow Updates

### Affected Workflows
- [ ] `/workflow-name` - [describe change needed]

### Affected Skills  
- [ ] `.agent/skills/skill-name/SKILL.md` - [describe change needed]

### Slashcommand Updates
- [ ] Add new: `/new-command`
- [ ] Update: `/existing-command` - [why]

> **N/A**: Mark as "N/A - no workflow/skill changes" if this section doesn't apply.
```

---

## ENFORCEMENT RULES

1. **Quality Gate HARD BLOCK**: Score <0.9 = DO NOT CREATE PLAN
2. **Synthesized Hypotheses**: If hypotheses were synthesized, the hybrid MUST be critiqued
3. **All Critiques Addressed**: Every major flaw from store_critique must have resolution
4. **Exact Code**: Changes must show exact code, not descriptions
5. **Runnable Verification**: Test commands must be copy-paste runnable
6. **Workflow Updates**: If changes affect workflows/skills, document in Section 5

---

## PRE-SUBMISSION CHECKLIST

> ✅ Complete ALL items before finalizing this plan.

- [ ] PAS session score ≥ 0.9
- [ ] All major critiques from PAS addressed in plan
- [ ] Synthesized hypotheses critiqued (if applicable)
- [ ] N/A sections explicitly marked (not left blank)
- [ ] Verification commands tested/runnable
- [ ] Exact code shown (not descriptions)
- [ ] Sequential gap analysis completed
