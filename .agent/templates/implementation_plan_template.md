# Implementation Plan Template

<!--
=== AGENT CONTEXT ===
Role: Implementation Planning Agent
Project: [Fill from PAS session project_id]
Mode: implementation
-->

> **🎯 Your Role**: Create an actionable plan that another agent can execute without conversation context.
> 
> **📋 Active Constraints** (verify via `start_reasoning_session`):
> - 🚫 `quality_gate_threshold`: 0.9 (BLOCKING)
> - 🚫 `terminal_env_activation`: required (BLOCKING)  
> - ⚠️ `verify_before_completing`: true
> - ⚠️ `sequential_analysis_required`: true

---

## PAS Reasoning Summary

> [!IMPORTANT] **Quality Gate Check**
> - Score must be ≥0.9 (do NOT proceed if below)
> - If hybrid/synthesized: MUST be critiqued before finalization
> - Gap must be ≥0.08 between top candidates

**Session ID**: `[uuid]`
**Goal**: [user_goal from start_reasoning_session]

### Hypotheses Evaluated

| ID | Content | Score | Critiqued? |
|----|---------|-------|------------|
| h1 | [summary] | 0.XX | YES/NO |
| h2 | [summary] | 0.XX | YES/NO |
| h3 | [summary] | 0.XX | YES/NO |

### Winning Hypothesis

**Node ID**: `[uuid]`
**Final Score**: [must be ≥0.9]
**Decision Quality**: HIGH/MEDIUM/LOW
**Gap**: [must be ≥0.08]

### Key Critiques & How Addressed

| Critique | Severity | Resolution in Plan |
|----------|----------|-------------------|
| [Issue 1] | 0.X | [How addressed] |
| [Issue 2] | 0.X | [How addressed] |

### Sequential Gap Analysis

| Gap Identified | Layer | Addressed? |
|----------------|-------|------------|
| [Gap 1] | CODE_STRUCTURE | ✅/❌ |
| [Gap 2] | DEPENDENCIES | ✅/❌ |

### Dual Recommendation (v82)

> [!TIP] **Why Two Options?**
> PAS provides Balanced (best ROI) and Aspirational (highest value).
> Document both so stakeholders can make informed tradeoffs.

| Aspect | Balanced (Chosen) | Aspirational |
|--------|-------------------|--------------|
| **Approach** | [Description] | [Description] |
| **Effort** | [1-3] | [1-3] |
| **Benefit** | [1-3] | [1-3] |
| **Why Not Chosen** | N/A | [Tradeoff reason] |
| **When to Reconsider** | N/A | [Conditions] |

---

## Scope

> [!TIP] **Scope Verification**
> - Run `find_references` for each modified symbol
> - Check for circular import risks in new files
> - Verify import paths with `grep_search` before creating helpers

**Files Modified**:
- `[layer] path/to/file.py` - [what changes]

**Files Created**:
- `[layer] path/to/new.py` - [purpose]

> Mark "N/A" if no files created.

**Files Deleted**:
- `path/to/old.py` - [why]

> Mark "N/A" if no files deleted.

---

## LSP Impact Analysis

> [!IMPORTANT] **Blast Radius Check**
> Run `find_references` on key symbols to discover all affected files.

**Symbols in scope**:

| File | Key Symbols |
|------|-------------|
| `path/to/file.py` | `func1`, `Class1` |

**Affected files** (from find_references):

| Symbol | Used By | Count |
|--------|---------|-------|
| `func1` | `other.py`, `test.py` | 5 |

**Scope completeness**: [Are all callers included above?]

---

## Changes

> [!IMPORTANT] **Code Quality Requirements**
> - Show EXACT code, not descriptions
> - Use diff format for modifications
> - Include rationale linking to PAS critiques
> - Cyclomatic complexity ≤15

### [Component/File Name]

#### [Function/Section]

**Before** (if modifying):
```python
[exact current code]
```

**After**:
```python
[exact new code]
```

**Rationale**: [Link to PAS critique that drove this change]

---

## Verification

> [!CAUTION] **No Empirical Evidence = Not Done**
> - Terminal output for CLI/backend changes
> - Screenshots for UI changes
> - Test results for logic changes
> - "Looks correct" is NOT verification

### Automated Tests

```bash
# Activate environment first (REQUIRED for PAS project)
source .venv312/bin/activate && set -a && source .env && set +a

# Run tests
pytest tests/ -v
```

**Expected output**: [What success looks like]

### Manual Verification

1. [Step 1]
2. [Step 2]

---

## Environment

> [!WARNING] **Terminal Commands**
> All `run_command` calls MUST include venv activation.
> Bare `pip`/`python` uses system interpreter, NOT project venv.

| Item | Value |
|------|-------|
| **Venv Path** | `.venv312/bin/activate` |
| **Activation** | `source .venv312/bin/activate && set -a && source .env && set +a` |
| **pip** | `.venv312/bin/pip` |
| **python** | `.venv312/bin/python` |

---

## Pre-Submission Checklist

> **⚠️ Known Issues to Watch For**:
> - Import errors when creating helpers (verify with `grep_search`)
> - SQL transaction aborts (use savepoints for multi-step)
> - Scope boundary crossings (check `find_references`)
> - HTML comments invisible in markdown (use visible alerts)

- [ ] PAS session score ≥ 0.9
- [ ] All major critiques addressed in plan
- [ ] Synthesized hypotheses critiqued (if applicable)
- [ ] Sequential gap analysis completed
- [ ] Exact code shown (not descriptions)
- [ ] Verification commands tested/runnable
- [ ] N/A sections explicitly marked

---

> **📋 Constraint Reminder**: This project enforces `verify_before_completing`. 
> Do not mark complete without empirical evidence.
