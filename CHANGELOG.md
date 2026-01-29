# Changelog

All notable changes to PAS (Scientific Reasoning MCP) are documented here.

## [v32] - 2026-01-29

### Fixed
- **MCP Sampling Compatibility** - All `create_message` calls replaced with prompt-return pattern
  - `prepare_critique` → returns `llm_prompt` for agent
  - `run_sequential_analysis` → split into `prepare_sequential_analysis` + `store_sequential_analysis`
  - `identify_gaps` → returns `llm_question_prompt` for agent
  - `finalize_session` → returns `exhaustive_prompt` for agent

### Changed
- Agent-Assisted Sampling architecture (v32h) - server as "dumb pipe" template generator

---

## [v31] - 2026-01-28

### Added
- **Negative Space Critique** - `critique_mode="negative_space"` to find what's missing
- **Exhaustive Checkout** - Layer-by-layer gap analysis on recommendations
- **Quality Gate** - Score ≥0.9, Gap ≥0.1 thresholds with improvement suggestions
- **Assumption Surfacing** - Hidden assumptions extraction via prompts

---

## [v27] - 2026-01-27

### Added
- **Session Tagging** - `tag_session` tool for organization
- **Past Failures Surfacing** - Show similar failures during critique

---

## [v26] - 2026-01-27

### Fixed
- Past failures surfacing in critique flow

---

## [v20] - 2026-01-26

### Added
- **Adaptive Depth Quality Metrics** - Self-aware quality assessment
- Likelihood-based critique detection

---

## [v19] - 2026-01-25

### Added
- **Domain Detection System** - Auto-detect problem domain (backend, frontend, etc.)
- **Dimension-based Questions** - Domain-specific interview questions

---

## [v18] - 2026-01-25

### Added
- **Smart LLM Gating** - Skip LLM question generation for specific goals
- **Progressive Disclosure** - One question at a time interview flow

---

## [v17] - 2026-01-24

### Added
- **RLVR Auto-Recording** (v17b) - Auto-detect success/failure from terminal output
- **Terminal Output Parsing** (v17a) - Domain-agnostic regex patterns
- **Semantic Failure Learning** (v17b.2) - Failure reasons for similarity matching

---

## [v16] - 2026-01-23

### Added
- **Historical Context Questions** - Learn from past Q&A patterns
- **Calibration Warnings** - Confidence nudges for extreme values
- **Hidden Context System** - Psychology-based question design

---

## [v15] - 2026-01-22

### Added
- **Domain-Agnostic Scope Guidance** - Self-learning scope recommendations
- **Evidence Tracking** - Store supporting evidence for hypotheses

---

## [v1-v14] - Earlier

Initial implementation:
- Bayesian Tree of Thoughts
- Scientific Law Grounding (15+ laws)
- Smart Interview Pattern
- Session Lifecycle Management
- Self-Learning via Outcome Recording
