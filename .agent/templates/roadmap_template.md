# Roadmap Template (REQUIRED for Multi-Phase Work)

> Use this template when work spans multiple phases or components.
> Each phase becomes its own implementation plan via separate PAS session.

---

## REQUIRED SECTIONS

### 1. Problem Statement & Reasoning
```markdown
## Problem Statement

**What problem does this solve?**
[Describe the pain point or gap]

**Why is this important?**
[Business/technical justification]

**PAS Session Evidence:**
- Session ID: `<uuid>`
- Decision Quality: HIGH/MEDIUM/LOW
- Final Score: ≥0.9 (REQUIRED)
- Gap: ≥0.08 (REQUIRED)
```

### 2. Architectural Overview
```markdown
## Architecture

### System Context Diagram
[Mermaid diagram showing how this fits in the system]

### Component Diagram  
[Mermaid diagram showing internal components]

### Data Flow
[Mermaid sequence or flowchart]
```

### 3. Phase Breakdown
```markdown
## Phases

### Phase 1: [Name]
- **Scope**: [What's included]
- **Dependencies**: [What must exist first]
- **PAS Session Required**: YES (link to implementation_plan.md)
- **Estimated Effort**: [low/medium/high]

### Phase 2: [Name]
[...]
```

### 4. Cross-Phase Decisions
```markdown
## Design Decisions

| Decision | Options Considered | Chosen | Rationale (from PAS) |
|----------|-------------------|--------|---------------------|
| [What] | [A, B, C] | [B] | [Why - link to PAS critique] |
```

### 5. Success Criteria
```markdown
## Success Criteria

- [ ] Criterion 1 (verifiable)
- [ ] Criterion 2 (verifiable)
```

### 6. Project Structure
```markdown
## Project Structure

> **For NEW projects**: Use language best practices
> - Python: `src/` layout with `pyproject.toml`
> - Node/TS: `src/` with `package.json`
>
> **For EXISTING projects**: 
> - If maintaining current layout → align with it
> - If restructuring → document BOTH current AND target structure
>   (Treat restructuring as a separate PAS session)

[Target structure tree if creating new files or restructuring]
```

### 7. Environment Context
```markdown
## Environment

> **Specify venv** for pip/python commands. Bare `pip`/`python` = system interpreter.

| Project Venv | Command Pattern |
|--------------|-----------------|
| `[FILL: /path/to/.venv/]` | `[VENV]/bin/pip`, `[VENV]/bin/python` |
```

### 8. Code Quality Standards (Python)
```markdown
## Code Quality Standards

> All new/modified code MUST meet these thresholds.

| Metric | Target | Max Allowed |
|--------|--------|-------------|
| Cyclomatic Complexity | ≤10 | 15 (with justification) |
| Function Length | ≤50 lines | 80 lines |
| File Length | ≤500 lines | 800 lines |

**Principles**: SRP (single responsibility), DRY (don't repeat), KISS (keep simple)

**Verification**: `radon cc [file.py] -s -a`
```

---

## ENFORCEMENT RULES

1. **Quality Gate**: Overall roadmap PAS session MUST score ≥0.9
2. **Per-Phase Plans**: Each phase gets separate PAS session + implementation_plan
3. **Self-Sufficient**: New LLM session must understand roadmap without conversation context
4. **Diagrams Required**: At least one mermaid diagram for architecture
5. **Decisions Linked**: Major decisions must reference PAS reasoning

---

## PRE-SUBMISSION CHECKLIST

> ✅ Complete ALL items before presenting this roadmap.

- [ ] PAS session score ≥ 0.9
- [ ] At least one mermaid diagram included
- [ ] Each phase has clear scope and dependencies
- [ ] Success criteria are verifiable (not vague)
- [ ] Design decisions link to PAS reasoning
- [ ] Roadmap is self-sufficient (understandable without conversation context)
