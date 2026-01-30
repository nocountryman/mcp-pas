# Changelog

All notable changes to PAS (Scientific Reasoning MCP) are documented here.

## [v44] - 2026-01-30

### Added
- **Raw Input Logging Enforcement** - Captures verbatim user prompts for psychological analysis
  - `raw_input` parameter on `start_reasoning_session` - stores exact user words
  - `skip_raw_input_check` escape hatch for LLM-initiated sessions
  - `check_raw_input_required()` preflight in `preflight_helpers.py`
  - **HARD ENFORCED**: Sessions with user-initiated keywords fail without `raw_input`
  - Auto-logs raw input with `log_type='verbatim'` in `conversation_log`

- **Rule 11** added to `pas-planning.md` workflow

### Technical
- PAS Session: `f6a25538-d320-442a-9431-407e2186e9c7` (score 0.953)
- Keywords detected: 'build', 'implement', 'design', 'create', 'user wants', etc.

---

## [v43] - 2026-01-30

### Added
- **Project Purpose Awareness** - Teleological understanding at project level
  - `project_registry` table with purpose_hierarchy, detected_domain, domain_confidence
  - HNSW index on purpose_embedding for semantic search
  - **New Tools**:
    - `infer_project_purpose` - Returns cached purpose or LLM inference prompt
    - `store_project_purpose` - Stores mission, user_needs, must_have_modules
    - `analyze_completeness` - Compares must_have_modules against actual modules
    - `get_purpose_chain` - Traces file → module → project purpose hierarchy
  - **Helper Functions** (`purpose_helpers.py`):
    - `PROJECT_PURPOSE_PROMPT_TEMPLATE` - 5-field project purpose inference
    - `build_project_purpose_prompt` - Builds inference prompt from project context
    - `parse_project_purpose_response` - Parses LLM response to structured data

- **PAS Workflow Templates** - Standardized planning artifacts
  - `.agent/templates/roadmap_template.md` - Multi-phase project roadmaps
  - `.agent/templates/implementation_plan_template.md` - Single-phase implementation plans
  - Templates enforce PAS evidence, scope declarations, verification plans

### Changed
- **`sync_project`** - Auto-upserts project_registry entry
- **`prepare_expansion`** - Includes project grounding when project_id has purpose
  - Adds `🎯 PROJECT MISSION` to instructions for hypothesis alignment
  - Returns `project_grounding` field with mission, user_needs, detected_domain
- **`config.yaml`** - Added `purpose_inference.completeness_similarity_threshold: 0.7`
- **`pas-planning.md` Workflow** - Major overhaul for quality gate enforcement
  - Score ≥0.9 HARD BLOCK before proceeding
  - Gap ≥0.08 required for decision confidence
  - Synthesized hypotheses MUST be critiqued

### Technical
- PAS sessions: `74b5612a` (roadmap, score 1.0), `e17bf57c` (Phase 1 plan, score 0.933)
- 3-phase implementation: Schema & Core Tools → Purpose Helpers → Integration

---


## [v42-tests] - 2026-01-30


### Added
- **Self-Aware Test Suite** - 4-layer test architecture for PAS
  - **Layer 1**: Static domain tests in `test_tools/*.py` (40+ tests for 38 tools)
  - **Layer 2**: YAML scenario runner (`test_scenarios.py`) with declarative workflow tests
  - **Layer 3**: Self-aware coverage reporter using `get_self_awareness()` to auto-detect missing tests
  - **Layer 4**: Pytest hooks in `conftest.py` that log test failures back to PAS for self-learning
  - Failure categorization: LOGIC, DB, CODE, WORKFLOW, INFRA
  - 58 tests discovered and passing
- **Dogfooding Law** - New scientific law for self-referential systems
  - "Internal tools should use their own capabilities"
  - Surfaced during planning gap analysis for test suite design

### Technical
- PAS sessions: `52a28c18` (planning), `6c57a67b` (gap logging)
- New directory structure: `tests/test_tools/`, `tests/scenarios/`

---

## [v42b] - 2026-01-30

### Added
- **Knowledge Surfacing Strategy** - Scope-based failure matching
  - `store_expansion` now returns `scope_failure_warnings` based on declared scope
  - `prepare_critique` returns `past_critiques` from similar hypotheses
  - `surfaced_warning_ids` column for deduplication
  - Context-aware thresholds in `_search_relevant_failures()`

---

## [v41] - 2026-01-30

### Added
- **Preflight Enforcement System** - Structural guardrails at `store_expansion`
  - `preflight_helpers.py` - Preflight check logic (~180 lines)
  - Warning types: `missing_schema_check`, `missing_find_references`, `unacknowledged_warnings`
  - `skip_preflight=True` escape hatch (logged for outcome correlation)
  - Rule 8 added to GEMINI.md for enforcement

---

## [v40] - 2026-01-30

### Added
- **Modularization Phase 7-12** - Further server.py decomposition
  - `metacognitive_helpers.py` - 5-stage metacognitive prompting (~200 lines)
  - `purpose_helpers.py` - Hierarchical purpose inference (~150 lines)
  - `hybrid_helpers.py` - Hypothesis synthesis (~120 lines)
  - `calibration_helpers.py` - CSR calibration logic (~180 lines)
  - `self_awareness_helpers.py` - Schema introspection (~250 lines)
- **Self-Awareness Tool** (`get_self_awareness`) - PAS can introspect its own:
  - Database schema (29 tables)
  - Tool catalog (38 tools with descriptions)
  - Architecture (4 workflows: reasoning, learning, codebase, metacognitive)
  - Metacognitive stages (5 stages per arXiv:2308.05342v4)

### Technical
- Major architectural milestone: PAS can now understand itself for meta-planning

---

## [v39] - 2026-01-30


### Added
- **Modularization Phase 0-6** - Major server.py decomposition
  - `errors.py` - Standardized exception hierarchy (~200 lines)
    - `PASError` base class with `to_dict()` for MCP responses
    - Session, Node, Database, Validation, Quality Gate, Codebase, Sampling errors
  - `utils.py` - Shared utilities (~185 lines)
    - `get_db_connection()`, `safe_close_connection()`
    - `get_embedding()` with lazy model loading
    - Validation helpers: `validate_uuid()`, `validate_confidence()`, `validate_outcome()`
    - `detect_negation()` for NLI
  - `reasoning_helpers.py` - Pure reasoning functions (~310 lines)
    - `apply_heuristic_penalties()`, `compute_ensemble_prior()`
    - `compute_decision_quality()`, `apply_uct_tiebreaking()`
    - `build_processed_candidate()`, `generate_suggested_tags()`
    - Constants: `HEURISTIC_PENALTIES`, `ROLLOUT_WEIGHT`, `UCT_THRESHOLD`
  - `learning_helpers.py` - RLVR/outcome functions (~240 lines)
    - `parse_terminal_signals()`, `extract_failure_reason()`
    - `signal_to_outcome()`, `compute_trait_reinforcement()`
    - Constants: `SUCCESS_PATTERNS`, `FAILURE_PATTERNS`, `FAILURE_REASON_PATTERNS`
  - `interview_helpers.py` - Interview flow helpers (~220 lines)
    - `get_interview_context()`, `extract_domain_from_goal()`
    - `format_question_for_display()`, `compute_interview_progress()`
    - Constants: `DEFAULT_INTERVIEW_CONFIG`, `DEFAULT_QUALITY_THRESHOLDS`
  - `codebase_helpers.py` - Code navigation utilities (~260 lines)
    - `extract_symbols()` - tree-sitter symbol extraction
    - `get_language_from_path()`, `should_skip_file()`
    - `extract_symbol_patterns_from_text()`, `build_reference_summary()`
    - Constants: `LANGUAGE_MAP`, `SKIP_EXTENSIONS`, `SKIP_DIRS`
  - `sessions_helpers.py` - Session lifecycle utilities (~270 lines)
    - `derive_user_id_from_goal()`, `compute_decayed_trait_score()`
    - `build_trait_entry()`, `merge_traits_into_context()`
    - `summarize_session_for_response()`, `build_continuation_context()`
    - Constants: `TRAIT_HALF_LIFE_DAYS`, `VALID_SESSION_STATES`

### Changed
- **server.py reduced from 5773 to 5389 lines (-384, 6.7%)**
- Pure helper functions imported with `_` prefix aliases for compatibility
- All patterns and constants now centralized in domain-specific modules

### Technical
- PAS self-analysis session `8fbb42ab` (score: 0.913, quality gate passed)
- "Logic-Only Module Extraction" pattern: MCP decorators stay in server.py as thin wrappers
- 7 new modules created, ~1,700 lines of reusable code

---

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
