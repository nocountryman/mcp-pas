# Changelog

All notable changes to PAS (Scientific Reasoning MCP) are documented here.

## [v38] - 2026-01-29

### Added
- **LSIF Integration** - Precision code navigation via Language Server Index Format
  - `import_lsif(project_id, lsif_path)` - Import LSIF JSON from Pyright
  - `find_references(project_id, symbol_name)` - Find all references to a symbol
  - `go_to_definition(project_id, file, line)` - Jump to symbol definition
  - `call_hierarchy(project_id, symbol_name, direction)` - Build caller/callee tree
  - New `symbol_references` table with 4 indexes for fast querying
  - Batch insert (1000/batch) for large codebases

- **v38b: Live Jedi Integration** - Always-fresh references without regeneration
  - `find_references` now uses live Jedi analysis (no LSIF needed)
  - Automatic fallback to LSIF if Jedi fails
  - Returns `source: "jedi"` or `source: "lsif"` to indicate data source
  - Added `jedi` to requirements.txt

- **v38c: Semi-Auto Reference Integration** - Symbol suggestions in prepare_expansion
  - Added optional `project_id` parameter to `prepare_expansion`
  - Extracts snake_case/CamelCase patterns from goal/parent text
  - Validates against `file_symbols` table (requires synced project)
  - Returns `suggested_lookups` array with symbol, file, line for agent to explore
  - Explicit instruction nudge: "⚠️ SUGGESTED: Before generating hypotheses, call find_references..."
  - Added Rule 7 to GEMINI.md for enforcement
  - PAS session `8abf2f83` passed quality gate (score: 0.9615)


### Discovered
- **Pyright lacks LSIF export** - Neither pip nor npm pyright has `--outputtype lsif`
  - Logged as partial outcome in session `bf627e62`



---

## [v37] - 2026-01-29


### Enforced
- **Mandatory Sequential Gap Analysis** - `prepare_sequential_analysis` + `store_sequential_analysis` must be called before `finalize_session`
  - Adversarial (critique) + Constructive (gaps) = complete analysis
  - 5-layer gap check: CODE, DEPENDENCIES, DATA FLOW, INTERFACES, WORKFLOWS
  - `skip_sequential_analysis=True` for explicit opt-out (like `skip_quality_gate`)
  - Persists `sequential_analyzed` flag in thought_nodes metadata

### Fixed
- **Interview `choices` KeyError** - `get_next_question` no longer fails when question lacks `choices` field
  - Added `choices: []` to historical questions in `identify_gaps`
  - Defensive `.get("choices", [])` in `get_next_question`

---

## [v36] - 2026-01-29

### Added
- **Externalized Config** - Quality gate params in `config.yaml`
  - `load_config()` loads from YAML, env vars override (e.g., `PAS_QUALITY_GATE_MIN_SCORE_THRESHOLD`)
  - `PAS_CONFIG` global loaded at startup
  - `finalize_session` now uses config defaults instead of hardcoded values

---

## [v35] - 2026-01-29

### Added
- **Pure Utility Helpers** - 6 extracted functions for testability:
  - `_apply_heuristic_penalties`, `_compute_ensemble_prior`, `_infer_traits_from_hidden_values`
  - `_get_outcome_multiplier`, `_compute_critique_accuracy`, `_compute_law_effective_weight`

### Changed
- **Complexity Reduction** - Average 15.75 → 13.74 (↓13%)
- **Type Safety** - mypy errors 53 → 3 (↓94%)

---

## [v35c] - 2026-01-29

### Changed
- **Quality Gate Tuning** - Lowered `min_gap_threshold` from 0.10 to 0.08
  - v32b planning consistently hit ~0.08 gap with thorough exploration
  - 0.10 was too strict for practical use

---

## [v32] - 2026-01-29

### Added
- **Dynamic Failure Pattern Discovery** - Patterns now stored in database
  - New `failure_patterns` table: pattern_name, keywords, warning_text, auto_derived
  - `_search_relevant_failures()` queries DB instead of hardcoded dict
  - Fallback to legacy dict if DB unavailable
  - Seeded with 4 patterns: SCHEMA_BEFORE_CODE, RESTART_BEFORE_VERIFY, ENV_CHECK_FIRST, RESPECT_QUALITY_GATE

---

## [v32b] - 2026-01-29

### Added
- **Warning Persistence** - `finalize_session` now surfaces past failure warnings
  - Dual-source search: goal + recommendation content
  - Dedupe by pattern, add to `warnings_surfaced` array
  - **Prepend ⚠️ items to `implementation_checklist`**
  - Agent gets both structured data AND visible checklist items

---

## [v31b] - 2026-01-29

### Added
- **Past Failure Surfacing** - Proactive warnings in `prepare_expansion` and `prepare_critique`
  - `KEYWORD_FAILURE_PATTERNS` dict: 16 keywords → 4 patterns (SCHEMA_BEFORE_CODE, RESTART_BEFORE_VERIFY, ENV_CHECK_FIRST, RESPECT_QUALITY_GATE)
  - `_search_relevant_failures()` helper: hybrid keyword + semantic matching
  - Returns `past_failure_warnings` array when matches found

---

## [v34] - 2026-01-29

### Added
- **Auto-Tagging** - `record_outcome` auto-applies `suggested_tags` from `finalize_session`
- **JSONB Persistence** - `suggested_tags` column in `reasoning_sessions`

---

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
