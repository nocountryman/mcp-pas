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

**Files Deleted**:
- `path/to/old.py` - [why]
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

### 4. Verification Plan
```markdown
## Verification

### Automated Tests
```bash
[exact command to run]
```

Expected output: [what success looks like]

### Manual Verification
1. [Step 1]
2. [Step 2]
```

---

## ENFORCEMENT RULES

1. **Quality Gate HARD BLOCK**: Score <0.9 = DO NOT CREATE PLAN
2. **Synthesized Hypotheses**: If hypotheses were synthesized, the hybrid MUST be critiqued
3. **All Critiques Addressed**: Every major flaw from store_critique must have resolution
4. **Exact Code**: Changes must show exact code, not descriptions
5. **Runnable Verification**: Test commands must be copy-paste runnable
