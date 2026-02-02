#!/usr/bin/env python3
"""
Scientific Reasoning MCP Server
Bayesian Tree of Thoughts with Semantic Search

This server provides tools for structured scientific reasoning,
backed by PostgreSQL with pgvector for semantic search.
"""

import os
import asyncio
import json
import re
import uuid
import logging
from typing import Any, Literal, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    TextContent,
    SamplingMessage,
    CreateMessageRequestParams,
    CreateMessageResult,
)

# v39: Modularized imports
from pas.helpers.reasoning import (
    apply_heuristic_penalties as _apply_heuristic_penalties,
    get_outcome_multiplier as _get_outcome_multiplier,
    compute_critique_accuracy as _compute_critique_accuracy,
    compute_law_effective_weight as _compute_law_effective_weight,
    compute_ensemble_prior as _compute_ensemble_prior,
    generate_suggested_tags as _generate_suggested_tags,
    compute_decision_quality as _compute_decision_quality,
    apply_uct_tiebreaking as _apply_uct_tiebreaking,
    build_processed_candidate as _build_processed_candidate,
    infer_traits_from_hidden_values as _infer_traits_from_hidden_values,
    search_relevant_failures as _search_relevant_failures,
    # Constants
    HEURISTIC_PENALTIES,
    KEYWORD_FAILURE_PATTERNS,
    DEPTH_BONUS_PER_LEVEL,
    ROLLOUT_WEIGHT,
    UCT_THRESHOLD,
    UCT_EXPLORATION_C,
    DOMAIN_PATTERNS,
)

from pas.helpers.learning import (
    parse_terminal_signals,
    extract_failure_reason,
    signal_to_outcome,
    compute_trait_reinforcement,
    compute_law_success_rate,
    # Patterns (constants)
    SUCCESS_PATTERNS,
    FAILURE_PATTERNS,
    FAILURE_REASON_PATTERNS,
)

from pas.helpers.interview import (
    get_interview_context,
    extract_domain_from_goal,
    process_interview_answer,
    format_question_for_display,
    compute_interview_progress,
    extract_hidden_context_from_interview,
    # Config constants
    DEFAULT_INTERVIEW_CONFIG,
    DEFAULT_QUALITY_THRESHOLDS,
    DOMAIN_KEYWORDS,
    # Phase 5: check_interview_complete helpers
    TRAIT_RULES,
    infer_traits_from_hidden_values,
    build_context_summary,
    aggregate_hidden_values,
    run_semantic_trait_matching,
    collect_unmatched_descriptions,
    run_trait_inference,
    persist_interview_context,
    # Phase 7: submit_answer helpers
    find_and_validate_question,
    record_answer_with_choice,
    process_answer_side_effects,
    # Phase 8: identify_gaps helpers
    query_historical_failures,
    detect_domains,
    load_dimension_questions,
    build_goal_question_prompt,
    prioritize_questions,
    # v73: Phase 7b Confirmation Loop
    generate_confirmation_questions,
)

from pas.helpers.critique import (
    fetch_node_with_laws,
    build_critique_prompt,
    search_past_failures,
    search_past_critiques,
    build_assumption_extraction_prompt,
)

from pas.helpers.codebase import (
    extract_symbols as _extract_symbols,
    extract_symbols_lsp as _extract_symbols_lsp,
    get_language_from_path,
    should_skip_file,
    compute_file_hash,
    derive_project_id,
    extract_symbol_patterns_from_text,
    build_reference_summary,
    # Phase 6: find_references helpers (v53: DB-first, Jedi removed)
    resolve_project_root,
    deduplicate_references,
    fetch_project_root,
    prefilter_files,
    HAS_RIPGREP,
    # Constants
    LANGUAGE_MAP,
    SKIP_EXTENSIONS,

    SKIP_DIRS,
)

from pas.helpers.sessions import (
    derive_user_id_from_goal,
    compute_decayed_trait_score,
    should_include_trait,
    build_trait_entry,
    build_initial_context,
    merge_traits_into_context,
    validate_session_state,
    compute_session_duration,
    summarize_session_for_response,
    build_continuation_context,
    # Constants
    TRAIT_HALF_LIFE_DAYS,
    TRAIT_INCLUSION_THRESHOLD,
    MAX_PERSISTENT_TRAITS,
    VALID_SESSION_STATES,
)

# v40: Hybrid synthesis helpers (complementarity detection)
from pas.helpers.hybrid import (
    detect_complementarity,
    synthesize_hypothesis_text,
    extract_addressed_goals,
    merge_scopes,
)

# v40 Phase 1: Purpose inference helpers
from pas.helpers.purpose import (
    build_purpose_prompt,
    parse_purpose_response,
    validate_purpose_cache,
    build_purpose_context_entry,
    summarize_purposes_for_prompt,
    # v40 Phase 1.5: Module aggregation
    build_module_aggregation_prompt,
    parse_module_response,
)

# v40 Phase 2: Metacognitive 5-stage prompting
from pas.helpers.metacognitive import (
    METACOGNITIVE_STAGES,
    get_stage_info,
    get_stage_prompt,
    validate_stage_progression,
    get_calibration_guidance,
    format_stage_status,
)

# v40 Phase 3: CSR Calibration
from pas.helpers.calibration import (
    OUTCOME_MAPPING,
    map_outcome_to_numeric,
    compute_calibration_stats,
    format_calibration_for_response,
)

# v40 Phase 4: Self-Awareness
from pas.helpers.self_awareness import (
    ARCHITECTURE_MAP,
    get_schema_info,
    get_tool_registry,
    get_session_statistics,
)

# Phase 1 Refactor: Finalize session helpers
from pas.helpers.finalize import (
    check_sequential_gate,
    check_critique_gate,
    compute_quality_gate,
    build_score_improvement_suggestions,
    apply_unverified_prefix,
    surface_warnings_and_tags,
    identify_pending_critiques,
    build_implementation_checklist,
    query_past_failures,
    query_calibration_warning,
    build_deep_critique_requests,
    check_complementarity,
    apply_uct_and_compute_decision,
    build_next_step_guidance,
    build_scope_guidance,
    build_exhaustive_prompt,
    get_context_summary,
    fetch_sibling_data,
    process_candidates,
    apply_config_defaults,
    fetch_session,
    fetch_candidates,
    # v51: ROI analysis
    build_roi_analysis,
    # v82: Dual recommendation (constraint-aware)
    compute_dual_recommendation,
)

# Phase 2 Refactor: Store expansion helpers
from pas.helpers.expansion import (
    build_hypotheses_list,
    verify_active_session,
    resolve_parent_path,
    match_laws_and_compute_prior,
    log_conversation_source,
    compute_workflow_nudges,
    query_scope_failures,
    run_preflight_checks,
    surface_scope_failures,
    fetch_calibration_context,  # v60: Calibration decay
)

# Phase 3 Refactor: Prepare expansion helpers
from pas.helpers.expansion import (
    validate_session_and_get_parent,
    fetch_and_boost_laws,
    build_trait_instructions,
    surface_past_failures,
    extract_symbol_suggestions,
    search_related_modules,
    fetch_project_grounding,
    fetch_historical_patterns,
    # v73: Phase 7a Psychological Pre-Processing
    perform_psychological_preprocessing,
    cache_psychological_extraction,
)

# Phase 4 Refactor: Record outcome helpers
from pas.helpers.outcomes import (
    compute_outcome_embeddings,
    insert_and_attribute_outcome,
    log_training_data,
    record_critique_accuracy,
    log_calibration_record,
    persist_user_traits,
    trigger_auto_refresh,
    apply_auto_tags,
    check_synthesis_critique_warning,
)

# Phase 6: Project Governance helpers
from pas.helpers.governance import (
    get_or_create_project_vision,
    get_roadmap_phases,
    create_roadmap_phase,
    store_artifact,
    get_artifact_versions,
    get_latest_artifact,
    search_artifacts_by_tag,
    search_artifacts_semantic,
    get_governance_hierarchy,
)

# Shared utilities (singleton embedding model)
from pas.utils import get_embedding


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pas-server")

# =============================================================================
# Phase 8: Unified Reasoning Modes Configuration
# =============================================================================

MODE_CONFIG = {
    "implementation": {
        "finalize_label": "recommendation",
        "outcomes": ["success", "partial", "failure"],
        "expansion_suffix": "Generate implementation hypotheses with concrete code changes.",
        "enforce_quality_gate": True,
    },
    "research": {
        "finalize_label": "synthesis",
        "outcomes": ["success", "partial", "failure", "knowledge_gained"],
        "expansion_suffix": "Generate exploratory hypotheses for design options and tradeoffs.",
        "enforce_quality_gate": False,  # Show warnings, don't block
    },
    "constraint_discovery": {
        "finalize_label": "constraints",
        "outcomes": ["success", "partial", "failure"],
        "expansion_suffix": "Discover project constraints through interview.",
        "enforce_quality_gate": False,  # Interview-driven, not hypothesis-driven
    },
}

# =============================================================================
# v36: Configuration Management
# =============================================================================


import yaml
from pathlib import Path

def load_config() -> dict:
    """Load configuration from config.yaml, with env var overrides."""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    
    # Default config (fallback if file doesn't exist)
    config = {
        "quality_gate": {
            "min_score_threshold": 0.9,
            "min_gap_threshold": 0.08,
        },
        "failure_surfacing": {
            "semantic_threshold": 0.55,
            "max_warnings": 3,
        },
        "session": {
            "max_depth": 10,
            "default_top_n": 3,
        },
        "calibration": {  # v56-v60: Calibration decay
            "decay_rate": 0.5,
            "bias_threshold": 0.1,
            "min_samples_for_decay": 20,
            "min_samples_for_stats": 10,
            "enable_domain_stratification": True,
            "fallback_to_global": True,
        }
    }
    
    # Load from YAML file if exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Deep merge
                    for section, values in file_config.items():
                        if section in config and isinstance(values, dict):
                            config[section].update(values)
                        else:
                            config[section] = values
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}, using defaults")
    
    # Env var overrides (e.g., PAS_QUALITY_GATE_MIN_SCORE_THRESHOLD=0.85)
    for section, values in config.items():
        if isinstance(values, dict):
            for key, default in values.items():
                env_key = f"PAS_{section.upper()}_{key.upper()}"
                env_val = os.getenv(env_key)
                if env_val:
                    try:
                        # Type coercion based on default type
                        if isinstance(default, float):
                            config[section][key] = float(env_val)
                        elif isinstance(default, int):
                            config[section][key] = int(env_val)
                        else:
                            config[section][key] = env_val
                        logger.info(f"Config override: {env_key}={env_val}")
                    except ValueError:
                        pass
    
    return config

# Global config - loaded once at startup
PAS_CONFIG = load_config()

# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    """Create a new database connection."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
    register_vector(conn)
    return conn


def safe_close_connection(conn):
    """
    Safely close a database connection with rollback.
    
    Always rolls back before closing to prevent stuck transactions
    caused by uncommitted errors. Commit must be called explicitly.
    """
    if conn:
        try:
            conn.rollback()  # Safe to call even after commit
        except Exception:
            pass  # Ignore rollback errors
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Ignore close errors


# =============================================================================
# MCP Server Setup
# =============================================================================

mcp = FastMCP(
    name="pas-server",
)


# =============================================================================
# v9b: Constitutional Principles for Critic
# =============================================================================

CONSTITUTIONAL_PRINCIPLES = [
    # Code-focused principles
    {
        "principle": "Identify unstated assumptions",
        "focus": "Implicit requirements, hidden dependencies, unverified preconditions",
        "question": "What must be true for this to work that isn't explicitly stated?",
        "domain": "code"
    },
    {
        "principle": "Check boundary conditions",
        "focus": "Empty inputs, null values, extreme sizes, negative numbers",
        "question": "What happens at the edges of valid input ranges?",
        "domain": "code"
    },
    {
        "principle": "Verify causation vs correlation",
        "focus": "Logical reasoning, cause-effect chains, spurious relationships",
        "question": "Does X actually cause Y, or are they merely correlated?",
        "domain": "code"
    },
    {
        "principle": "Question unstated dependencies",
        "focus": "Libraries, services, configurations, external state",
        "question": "What external systems/state does this rely on?",
        "domain": "code"
    },
    {
        "principle": "Consider failure modes",
        "focus": "Error handling, timeouts, partial failures, rollback",
        "question": "What happens when this fails? Is recovery possible?",
        "domain": "code"
    },
    # v9b.1: UI/UX design principles
    {
        "principle": "Apply Hick's Law (choice complexity)",
        "focus": "Decision paralysis, option overload, cognitive burden",
        "question": "Are there too many choices? Can we reduce or progressively disclose options?",
        "domain": "ui_ux"
    },
    {
        "principle": "Verify visual hierarchy",
        "focus": "Primary/secondary actions, information prominence, scan patterns",
        "question": "Is the most important element the most visually prominent? Does flow match user priority?",
        "domain": "ui_ux"
    },
    {
        "principle": "Check accessibility (WCAG)",
        "focus": "Color contrast, keyboard nav, screen readers, touch targets",
        "question": "Can users with disabilities perceive and operate this? Is contrast ≥4.5:1?",
        "domain": "ui_ux"
    },
    {
        "principle": "Test responsive breakpoints",
        "focus": "Mobile-first, tablet, desktop, large screens",
        "question": "Does this work on all screen sizes? Are touch targets ≥44px on mobile?",
        "domain": "ui_ux"
    },
    {
        "principle": "Validate feedback & affordances",
        "focus": "Loading states, hover/active states, error messages, success confirmations",
        "question": "Does the UI communicate what's happening? Are interactive elements obviously clickable?",
        "domain": "ui_ux"
    }
]


# =============================================================================
# v31: Past Failure Surfacing - Keyword Patterns
# v32: Legacy fallback - kept for offline mode only
# Primary source is now failure_patterns table
KEYWORD_FAILURE_PATTERNS: dict[str, tuple[str, str]] = {
    'schema': ('SCHEMA_BEFORE_CODE', 'Run schema migration before deploying code changes'),
    'table': ('SCHEMA_BEFORE_CODE', 'Verify table exists in database before querying'),
    'restart': ('RESTART_BEFORE_VERIFY', 'Restart MCP server after code changes'),
    'venv': ('ENV_CHECK_FIRST', 'Verify .venv exists and is activated'),
    'quality': ('RESPECT_QUALITY_GATE', 'Do not skip quality gate checks'),
}


def _build_plan_template_prompt(
    quality_gate: dict,
    goal: str,
    dual_recommendation: dict | None = None
) -> dict | None:
    """
    v49: Detect whether roadmap or implementation plan template is appropriate.
    v69: Add dual-plan output section when aspirational differs from balanced.
    
    Keywords like 'roadmap', 'multi-phase', 'architecture', 'phases' suggest
    multi-phase work requiring a roadmap. Otherwise, implementation plan.
    
    Args:
        quality_gate: Quality gate result from finalize
        goal: Session goal text
        dual_recommendation: v69 - Dual recommendation with balanced/aspirational
    
    Returns template prompt dict or None if gate not passed.
    """
    if not quality_gate.get("passed"):
        return None
    
    goal_lower = goal.lower()
    roadmap_keywords = ["roadmap", "multi-phase", "phases", "architecture", "design system", "restructure"]
    is_roadmap = any(kw in goal_lower for kw in roadmap_keywords)
    
    # v69: Check if dual-plan output is needed
    has_aspirational = (
        dual_recommendation is not None
        and dual_recommendation.get("aspirational") is not None
    )
    
    if is_roadmap:
        return {
            "template_type": "roadmap",
            "template_path": ".agent/templates/roadmap_template.md",
            "required_sections": [
                "Problem Statement & PAS Evidence",
                "Architectural Overview (mermaid diagrams)",
                "Phase Breakdown (each phase = separate PAS session)",
                "Cross-Phase Design Decisions",
                "Success Criteria (verifiable)",
                "Environment (venv path)"
            ],
            "enforcement": "SOFT - use roadmap for multi-phase work"
        }
    else:
        # Base required sections for implementation plan
        sections = [
            "PAS Reasoning Summary (session ID, hypotheses table, scores)",
            "Key Critiques & How Addressed",
            "Sequential Gap Analysis Results",
            "Scope Declaration (files modified/created/deleted)",
            "Detailed Changes (exact before/after code)",
            "Verification Plan (copy-paste runnable commands)",
            "Environment (venv path)",
            "Pre-submission Checklist"
        ]
        
        result = {
            "template_type": "implementation_plan",
            "template_path": ".agent/templates/implementation_plan_template.md",
            "required_sections": sections,
            "enforcement": "SOFT - template shown, agent should follow"
        }
        
        # v69: Add dual-plan section when aspirational differs
        if has_aspirational:
            balanced = dual_recommendation.get("balanced", {})
            aspirational = dual_recommendation.get("aspirational", {})
            result["dual_plan_output"] = {
                "required": True,
                "instruction": (
                    "Generate TWO complete implementation plans:\n"
                    f"1. **Balanced Plan** (node {balanced.get('node_id', 'N/A')[:8]}...): "
                    f"Best ROI - {balanced.get('content', '')[:100]}...\n"
                    f"2. **Aspirational Plan** (node {aspirational.get('node_id', 'N/A')[:8]}...): "
                    f"Highest benefit - {aspirational.get('content', '')[:100]}...\n\n"
                    "Include a **Comparison Section** highlighting:\n"
                    "- Scope differences between plans\n"
                    "- Effort/benefit tradeoffs\n"
                    "- Recommendation for which to use based on constraints"
                ),
                "note": dual_recommendation.get("aspirational_note", "Consider aspirational if effort is unconstrained")
            }
        
        return result



# =============================================================================
# v10b: Critic Ensemble Personas
# =============================================================================

CRITIC_PERSONAS = [
    {
        "name": "Strict Skeptic",
        "instruction": "Find any reason to reject this hypothesis. Assume it's flawed until proven otherwise. Be harsh but fair.",
        "focus_areas": ["logical fallacies", "unsupported claims", "wishful thinking"]
    },
    {
        "name": "Pragmatic Engineer",
        "instruction": "Will this work in production? Consider scale, maintenance, edge cases, and operational complexity.",
        "focus_areas": ["scalability", "maintainability", "error handling", "performance"]
    },
    {
        "name": "Domain Expert",
        "instruction": "Does this align with established patterns and scientific principles? Check against known laws and best practices.",
        "focus_areas": ["pattern adherence", "principle alignment", "industry standards"]
    },
    # v59: Security Analyst (domain-conditional activation)
    {
        "name": "Security Analyst",
        "instruction": "What can be exploited? Check for injection (SQL, command, XSS), auth bypass, data leaks, privilege escalation, insecure defaults, sensitive data exposure.",
        "focus_areas": ["injection", "authentication", "authorization", "data protection", "input validation"],
        "domain_triggers": ["auth", "api", "password", "token", "session", "permission", "database", "query", "input", "user", "login", "credential", "secret", "key"]
    },
    # v59: Meta-Critic (always-on self-questioning)
    {
        "name": "Meta-Critic",
        "instruction": "Am I asking the right questions? Is the problem framed correctly? What assumptions are being made? What's NOT being considered? Is this solving symptoms or root cause?",
        "focus_areas": ["problem framing", "hidden assumptions", "root cause", "scope completeness"],
        "always_active": True
    }
]

AGGREGATION_GUIDANCE = "Run critique with each persona. For final severity: use MAX severity if any persona finds a critical flaw, otherwise AVERAGE across personas."


# =============================================================================
# v10a: Negation Detection Helper
# =============================================================================

NEGATION_PATTERNS = {'not', "n't", 'never', 'prohibit', 'avoid', 'without', 'disable', 'prevent', 'forbid', 'cannot', "won't", "shouldn't", "mustn't"}


def detect_negation(text: str) -> set:
    """Detect negation patterns in text for v10a NLI MVP."""
    text_lower = text.lower()
    return {p for p in NEGATION_PATTERNS if p in text_lower}


# =============================================================================
# Sampling Helper - Communicates with the Host LLM (Antigravity IDE)
# =============================================================================

class SamplingError(Exception):
    """Raised when sampling request fails or is denied."""
    pass


class SamplingDeniedError(SamplingError):
    """Raised when user denies the sampling request."""
    pass


async def sample_agent(
    prompt: str,
    context: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system_prompt: str | None = None
) -> str:
    """
    Request text generation from the host LLM via MCP sampling.
    
    This function constructs a sampling/createMessage request and sends it
    to the connected MCP client (Antigravity IDE), which routes it to the LLM.
    
    Args:
        prompt: The user prompt to send to the LLM
        context: Optional context to prepend to the prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-1.0)
        system_prompt: Optional system prompt override
        
    Returns:
        The generated text from the LLM
        
    Raises:
        SamplingDeniedError: If user denies the sampling request
        SamplingError: If sampling fails for any other reason
    """
    # Build the message content
    content = prompt
    if context:
        content = f"Context:\n{context}\n\n{prompt}"
    
    # Construct the sampling message
    messages = [
        SamplingMessage(
            role="user",
            content=TextContent(type="text", text=content)
        )
    ]
    
    # Build request parameters
    params = CreateMessageRequestParams(
        messages=messages,
        maxTokens=max_tokens,
        temperature=temperature,
        systemPrompt=system_prompt or "You are a scientific reasoning assistant. Provide clear, logical analysis."
    )
    
    try:
        # Get the current request context to access sampling capability
        from mcp.server.fastmcp import Context
        ctx = Context.current()  # type: ignore[attr-defined]
        
        if not ctx or not ctx.session:
            raise SamplingError("No active MCP session - cannot perform sampling")
        
        # Check if client supports sampling
        client_caps = ctx.session.client_capabilities
        if not client_caps or not client_caps.sampling:
            raise SamplingError("Client does not support sampling capability")
        
        # Send the sampling request via JSON-RPC
        logger.info(f"Requesting sample from host LLM: {prompt[:100]}...")
        
        result: CreateMessageResult = await ctx.session.create_message(
            messages=params.messages,
            max_tokens=params.maxTokens,
            temperature=params.temperature,
            system_prompt=params.systemPrompt
        )
        
        # Extract the response text
        if result.content and result.content.type == "text":
            logger.info(f"Received response: {result.content.text[:100]}...")
            return result.content.text
        else:
            raise SamplingError(f"Unexpected response content type: {result.content}")
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for denial patterns
        if "denied" in error_msg or "rejected" in error_msg or "cancelled" in error_msg:
            logger.warning(f"Sampling request denied by user: {e}")
            raise SamplingDeniedError(f"Sampling request was denied: {e}")
        
        # Check for capability errors
        if "not supported" in error_msg or "capability" in error_msg:
            logger.error(f"Sampling not supported: {e}")
            raise SamplingError(f"Sampling not supported by client: {e}")
        
        # Re-raise other errors
        logger.error(f"Sampling failed: {e}")
        raise SamplingError(f"Sampling request failed: {e}")


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def start_reasoning_session(
    user_goal: str,
    raw_input: Optional[str] = None,
    skip_raw_input_check: bool = False,
    session_mode: str = "implementation",
    unconstrained: bool = False,
    project_id: Optional[str] = None,
    project_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Start a new reasoning session for the given goal.
    
    Creates a new session in the database and returns the session ID.
    Use this to begin a structured reasoning process.
    
    Args:
        user_goal: The high-level goal or question to reason about
        raw_input: v44 - Verbatim user prompt (auto-logged with log_type='verbatim')
        skip_raw_input_check: v44 - Bypass raw_input enforcement for LLM-initiated sessions
        session_mode: Phase 8 - 'implementation' (code changes) or 'research' (exploration)
        unconstrained: Phase 8 - If True in research mode, skip project constraint surfacing
        project_id: v76 - Project identifier (required for constraint_discovery mode)
        project_path: v76 - Project path (required for constraint_discovery GEMINI.md export)
        
    Returns:
        Dictionary with session_id and status
    """
    if not user_goal or not user_goal.strip():
        return {
            "success": False,
            "error": "User goal cannot be empty"
        }
    
    # Phase 8: Validate session_mode
    if session_mode not in MODE_CONFIG:
        return {
            "success": False,
            "error": f"Invalid session_mode. Must be one of: {list(MODE_CONFIG.keys())}"
        }
    
    # v44: Hard enforcement of raw_input for user-initiated sessions
    if not skip_raw_input_check:
        from pas.helpers.preflight import check_raw_input_required
        raw_input_warning = check_raw_input_required(user_goal, raw_input)
        if raw_input_warning:
            return {
                "success": False,
                "error": "raw_input required for user-initiated sessions",
                "warning": raw_input_warning,
                "hint": "Pass raw_input='<verbatim user prompt>' or skip_raw_input_check=True if LLM-initiated"
            }
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # v19: Generate goal embedding for domain detection
        goal_embedding = None
        try:
            goal_embedding = get_embedding(user_goal.strip())
        except Exception as e:
            logger.warning(f"Failed to generate goal embedding: {e}")
        
        # v76: Build initial context with project info for constraint_discovery
        initial_context = {"source": "mcp_tool"}
        if project_id:
            initial_context["project_id"] = project_id
        if project_path:
            initial_context["project_path"] = project_path
        
        # Insert the new session (Phase 8: with session_mode and unconstrained)
        cur.execute(
            """
            INSERT INTO reasoning_sessions (id, goal, goal_embedding, state, session_mode, unconstrained, context)
            VALUES (%s, %s, %s, 'active', %s, %s, %s)
            RETURNING id, goal, state, session_mode, unconstrained, created_at
            """,
            (session_id, user_goal.strip(), goal_embedding, session_mode, unconstrained, json.dumps(initial_context))
        )
        
        row = cur.fetchone()
        
        # =====================================================================
        # v22 Feature 1: Load Persistent Trait Profiles
        # Query user_trait_profiles for this project and pre-populate context
        # =====================================================================
        persistent_traits = []
        user_id = None
        
        try:
            # Use goal hash as a proxy for project identity (could be enhanced)
            import hashlib
            # Generate user_id from goal pattern (simplified - in production use project path)
            user_id = hashlib.sha256(user_goal.strip()[:100].encode()).hexdigest()[:32]
            
            cur.execute("""
                SELECT trait_name, cumulative_score, last_reinforced_at
                FROM user_trait_profiles
                WHERE user_id = %s AND cumulative_score > 0
                ORDER BY cumulative_score DESC
                LIMIT 5
            """, (user_id,))
            
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            half_life_days = 30
            
            for trait_row in cur.fetchall():
                days_elapsed = (now - trait_row["last_reinforced_at"]).days
                decayed_score = trait_row["cumulative_score"] * (0.5 ** (days_elapsed / half_life_days))
                
                if decayed_score >= 0.3:  # Threshold for inclusion
                    persistent_traits.append({
                        "trait": trait_row["trait_name"],
                        "confidence": min(decayed_score, 1.0),
                        "source": "persistent",
                        "days_since_reinforced": days_elapsed
                    })
            
            if persistent_traits:
                # Update session context with persistent traits AND user_id
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = context || %s::jsonb
                    WHERE id = %s
                """, (json.dumps({"persistent_traits": persistent_traits, "user_id": user_id}), session_id))
                conn.commit()
                logger.info(f"v22: Loaded {len(persistent_traits)} persistent traits for session")
            else:
                # v22: Still store user_id for trait persistence on record_outcome
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = context || %s::jsonb
                    WHERE id = %s
                """, (json.dumps({"user_id": user_id}), session_id))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"v22: Failed to load persistent traits: {e}")
        
        conn.commit()
        
        logger.info(f"Created reasoning session: {session_id}")
        
        response = {
            "success": True,
            "session_id": str(row["id"]),
            "goal": row["goal"],
            "state": row["state"],
            "created_at": row["created_at"].isoformat(),
            "message": f"Reasoning session started. Use this session_id for subsequent reasoning operations.",
            # v40 Phase 2: Metacognitive 5-stage
            "metacognitive_stage": 0,
            "first_stage_prompt": get_stage_prompt(1)
        }
        
        if persistent_traits:
            response["persistent_traits"] = persistent_traits
            response["message"] += f" Loaded {len(persistent_traits)} persistent trait(s) from history."
        
        # v79: Surface project constraints at session start when project_id provided
        if project_id and not unconstrained:
            try:
                cur.execute("""
                    SELECT constraint_type, constraint_key, constraint_data, enforcement_level
                    FROM project_constraints pc
                    JOIN project_registry pr ON pc.project_id = pr.id
                    WHERE pr.project_id = %s AND pc.valid_to IS NULL
                    ORDER BY 
                        CASE enforcement_level WHEN 'block' THEN 1 WHEN 'warn' THEN 2 ELSE 3 END,
                        constraint_type
                """, (project_id,))
                
                constraints = []
                for row in cur.fetchall():
                    icon = "🚫" if row["enforcement_level"] == "block" else "⚠️"
                    constraints.append({
                        "icon": icon,
                        "type": row["constraint_type"],
                        "key": row["constraint_key"],
                        "enforcement": row["enforcement_level"]
                    })
                
                if constraints:
                    response["project_constraints"] = constraints
                    response["constraint_summary"] = f"{len(constraints)} active constraints ({sum(1 for c in constraints if c['enforcement'] == 'block')} blocking)"
                    logger.info(f"v79: Surfaced {len(constraints)} constraints for project {project_id}")
            except Exception as e:
                logger.warning(f"v79: Failed to surface constraints: {e}")
        
        # v44: Auto-log raw_input with log_type='verbatim' if provided
        if raw_input:
            from pas.helpers.sessions import log_verbatim_input
            verbatim_log_id = log_verbatim_input(
                cur, conn, session_id, raw_input, user_id, get_embedding
            )
            if verbatim_log_id:
                response["verbatim_log_id"] = verbatim_log_id
                response["message"] += " Raw input logged (verbatim)."
        
        return response

        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_session_status(session_id: str) -> dict[str, Any]:
    """
    Get the current status of a reasoning session.
    
    Args:
        session_id: The UUID of the session to check
        
    Returns:
        Dictionary with session details and thought count
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session details
        cur.execute(
            """
            SELECT id, goal, state, context, prior_beliefs, created_at, updated_at
            FROM reasoning_sessions
            WHERE id = %s
            """,
            (session_id,)
        )
        
        session = cur.fetchone()
        if not session:
            return {
                "success": False,
                "error": f"Session {session_id} not found"
            }
        
        # Count thoughts in this session
        cur.execute(
            """
            SELECT COUNT(*) as thought_count,
                   MAX(depth) as max_depth
            FROM thought_nodes
            WHERE session_id = %s
            """,
            (session_id,)
        )
        
        stats = cur.fetchone()
        
        return {
            "success": True,
            "session_id": str(session["id"]),
            "goal": session["goal"],
            "state": session["state"],
            "thought_count": stats["thought_count"],
            "max_depth": stats["max_depth"] or 0,
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def search_relevant_laws(query: str, limit: int = 5) -> dict[str, Any]:
    """
    Search for relevant scientific laws using semantic similarity.
    
    Uses pgvector to find laws that are semantically similar to the query.
    
    Args:
        query: The text to search for relevant laws
        limit: Maximum number of results (default: 5)
        
    Returns:
        List of relevant laws with similarity scores
    """
    try:
        # Use singleton embedding from utils
        query_embedding = get_embedding(query)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Search for similar laws
        cur.execute(
            """
            SELECT 
                id,
                law_name,
                definition,
                scientific_weight,
                1 - (embedding <=> %s::vector) as similarity
            FROM scientific_laws
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, limit)
        )
        
        laws = []
        for row in cur.fetchall():
            laws.append({
                "id": row["id"],
                "law_name": row["law_name"],
                "definition": row["definition"],
                "scientific_weight": float(row["scientific_weight"]),
                "similarity": round(float(row["similarity"]), 4)
            })
        
        return {
            "success": True,
            "query": query,
            "results": laws,
            "count": len(laws)
        }
        
    except Exception as e:
        logger.error(f"Failed to search laws: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v44b: Requirement Extraction Tools
# =============================================================================

@mcp.tool()
async def prepare_prompt_analysis(session_id: str) -> dict[str, Any]:
    """
    Returns analysis prompt with laws pulled from DB.
    
    Gets the verbatim log for the session and searches for relevant
    psychology laws to include in the prompt template.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Prompt context with verbatim text and available psychology laws
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get verbatim text from conversation_log
        cur.execute(
            """
            SELECT raw_text FROM conversation_log
            WHERE session_id = %s AND log_type = 'verbatim'
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id,)
        )
        row = cur.fetchone()
        verbatim_text = row["raw_text"] if row else None
        
        if not verbatim_text:
            safe_close_connection(conn)
            return {
                "success": False,
                "error": "No verbatim log found for session"
            }
        
        # Search for psychology-related laws
        query = "requirement elicitation psychology hedging speech act user intent"
        query_embedding = get_embedding(query)
        
        cur.execute(
            """
            SELECT 
                id,
                law_name,
                definition,
                scientific_weight,
                law_domain,
                1 - (embedding <=> %s::vector) as similarity
            FROM scientific_laws
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 10
            """,
            (query_embedding, query_embedding)
        )
        
        laws = []
        for row in cur.fetchall():
            laws.append({
                "id": row["id"],
                "law_name": row["law_name"],
                "definition": row["definition"],
                "scientific_weight": float(row["scientific_weight"]),
                "law_domain": row["law_domain"]
            })
        
        # Format laws for prompt
        laws_text = "\n".join([
            f"- **{law['law_name']}** (id: {law['id']}): {law['definition']}"
            for law in laws
        ])
        
        analysis_prompt = f"""## Requirement & Pattern Extraction Analysis

### User Prompt
```
{verbatim_text}
```

### Available Techniques (cite which you use)
{laws_text}

### Instructions
1. Extract explicit requirements (direct requests)
2. Extract implicit requirements (inferred from context)
3. Identify uncertainty markers (hedging language)
4. **Detect speech patterns:**
   - **Hesitation**: Ellipsis (...), filler words (like, basically), mid-sentence rephrasing, heavy hedging
   - **Enthusiasm**: Exclamation marks, superlatives (ideal, perfect), intensive adverbs (really, very)
   - **Concern**: Repetition of constraints (must, only), risk/failure words, security focus
5. For each extraction, cite which law/technique you applied

### Output Format
Return ONLY valid JSON:
```json
{{
  "explicit_requirements": [{{"text": "...", "confidence": 0.9}}],
  "implicit_requirements": [{{"text": "...", "inferred_from": "..."}}],
  "uncertainty_markers": [{{"phrase": "...", "type": "hedging"}}],
  "speech_patterns": [
    {{"type": "hesitation"|"enthusiasm"|"concern", "markers": ["..."], "confidence": 0.8, "source_phrase": "..."}}
  ],
  "laws_applied": [
    {{"law_id": 16, "law_name": "Laddering", "applied_to": "requirement 1", "confidence": 0.9}}
  ]
}}
```"""

        
        safe_close_connection(conn)
        
        return {
            "success": True,
            "session_id": session_id,
            "verbatim_text": verbatim_text,
            "available_laws": [{"id": l["id"], "name": l["law_name"]} for l in laws],
            "analysis_prompt": analysis_prompt
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare prompt analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_prompt_analysis(
    session_id: str,
    analysis_json: str
) -> dict[str, Any]:
    """
    Store LLM analysis with law citations.
    
    Stores extracted requirements and creates junction records
    linking requirements to the laws used to extract them.
    
    Args:
        session_id: The reasoning session UUID
        analysis_json: JSON string with extraction results
        
    Returns:
        Confirmation with counts of stored items
    """
    try:
        analysis = json.loads(analysis_json)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        stored_count = 0
        citation_count = 0
        
        # Store explicit requirements
        for req in analysis.get("explicit_requirements", []):
            embedding = get_embedding(req["text"])
            
            cur.execute(
                """
                INSERT INTO extracted_requirements 
                    (session_id, requirement_text, requirement_type, confidence, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                RETURNING id
                """,
                (session_id, req["text"], "explicit", req.get("confidence", 0.8), embedding)
            )
            req_id = cur.fetchone()["id"]
            stored_count += 1
            
            # Link cited laws
            for law in analysis.get("laws_applied", []):
                if law.get("applied_to") and law["applied_to"].lower() in req["text"].lower():
                    cur.execute(
                        """
                        INSERT INTO requirement_law_citations 
                            (requirement_id, law_id, confidence, application_context)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (req_id, law["law_id"], law.get("confidence", 0.8), law.get("law_name"))
                    )
                    citation_count += 1
        
        # Store implicit requirements
        for req in analysis.get("implicit_requirements", []):
            embedding = get_embedding(req["text"])
            
            cur.execute(
                """
                INSERT INTO extracted_requirements 
                    (session_id, requirement_text, requirement_type, inferred_from, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                RETURNING id
                """,
                (session_id, req["text"], "implicit", req.get("inferred_from"), embedding)
            )
            stored_count += 1
        
        # Store speech patterns (v44c)
        patterns_stored = 0
        for pattern in analysis.get("speech_patterns", []):
            pattern_type = pattern.get("type", "unknown")
            markers = pattern.get("markers", [])
            source_phrase = pattern.get("source_phrase", "")
            confidence = pattern.get("confidence", 0.8)
            
            # Generate embedding for v44d synthesis
            pattern_text = f"{pattern_type}: {source_phrase} ({', '.join(markers)})"
            embedding = get_embedding(pattern_text)
            
            # Get conversation_log_id for FK (most recent verbatim log)
            cur.execute(
                """
                SELECT id FROM conversation_log
                WHERE session_id = %s AND log_type = 'verbatim'
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,)
            )
            log_row = cur.fetchone()
            conversation_log_id = log_row["id"] if log_row else None
            
            cur.execute(
                """
                INSERT INTO detected_patterns 
                    (session_id, conversation_log_id, pattern_type, markers, confidence, source_phrase, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (session_id, conversation_log_id, pattern_type, markers, confidence, source_phrase, embedding)
            )
            patterns_stored += 1
        
        conn.commit()
        
        # Get laws cited for response
        laws_cited = [l["law_name"] for l in analysis.get("laws_applied", [])]
        
        return {
            "success": True,
            "session_id": session_id,
            "stored_count": stored_count,
            "citations_created": citation_count,
            "patterns_stored": patterns_stored,  # v44c
            "laws_cited": laws_cited,
            "uncertainty_markers_count": len(analysis.get("uncertainty_markers", []))
        }

        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in analysis: {e}")
        return {
            "success": False,
            "error": f"Invalid JSON: {e}"
        }
    except Exception as e:
        logger.error(f"Failed to store prompt analysis: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Reasoning Tools - Expansion & Critique (Phase 1: Prompt-Driven)
# =============================================================================

# NOTE: get_embedding is imported from utils (singleton, loads model once)


@mcp.tool()
async def store_psychological_extraction(
    session_id: str,
    extraction_json: str
) -> dict[str, Any]:
    """
    v73: Cache psychological extraction in session context.
    
    Called after agent processes the psychological_extraction_prompt
    from prepare_expansion. Caches the result so subsequent expansions
    can use the psychological context.
    
    Args:
        session_id: The reasoning session UUID
        extraction_json: JSON string with hedging_detected, enthusiasm_level, etc.
        
    Returns:
        Confirmation with cached fields
    """
    try:
        extraction = json.loads(extraction_json)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        success = cache_psychological_extraction(cur, conn, session_id, extraction)
        
        safe_close_connection(conn)
        
        if success:
            return {
                "success": True,
                "session_id": session_id,
                "cached_fields": list(extraction.keys()),
                "hedging_detected": extraction.get("hedging_detected", False),
                "inferred_priority": extraction.get("inferred_priority", "unknown"),
                "message": "Psychological extraction cached in session context"
            }
        else:
            return {
                "success": False,
                "error": "Failed to cache extraction"
            }
            
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"store_psychological_extraction failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def prepare_expansion(
    session_id: str,
    parent_node_id: str | None = None,
    project_id: Optional[str] = None  # v38c: For symbol suggestion lookups
) -> dict[str, Any]:
    """
    Prepare context for thought expansion. Returns parent content, session goal, 
    and relevant scientific laws so the LLM can generate hypotheses.
    
    After calling this, generate 3 hypotheses and pass them to store_expansion().
    
    Args:
        session_id: The reasoning session UUID
        parent_node_id: Parent thought UUID (None to expand from root/goal)
        project_id: Optional project ID for symbol suggestions (v38c)
        
    Returns:
        Context dict with parent_content, goal, relevant_laws for hypothesis generation
    """

    # v2026-02-02: Debug - add source check to response
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # =====================================================================
        # Helper 1: Validate session and get parent
        # =====================================================================
        session, parent_content, parent_path, parent_node_id, error = validate_session_and_get_parent(
            cur, session_id, parent_node_id
        )
        if error:
            return error
        
        # =====================================================================
        # Helper 2: Fetch and boost laws (v22)
        # =====================================================================
        laws, boosted_laws = fetch_and_boost_laws(cur, parent_content, session_id, get_embedding)
        
        # =====================================================================
        # Helper 3: Build trait instructions (v21)
        # =====================================================================
        instructions, latent_traits = build_trait_instructions(cur, session_id)
        
        # =====================================================================
        # Helper 4: Surface past failures (v31)
        # =====================================================================
        past_failure_warnings = surface_past_failures(_search_relevant_failures, parent_content)
        
        # =====================================================================
        # v73: Phase 7a - Psychological Pre-Processing (root expansion only)
        # =====================================================================
        is_root_expansion = parent_node_id is None
        psych_extraction, psych_prompt = perform_psychological_preprocessing(
            cur, conn, session_id, is_root_expansion, get_embedding
        )
        
        # Build initial response
        response = {
            "success": True,
            "session_id": session_id,
            "parent_node_id": parent_node_id,
            "parent_path": parent_path,
            "parent_content": parent_content,
            "goal": session["goal"],
            "relevant_laws": laws,
            "instructions": instructions,
            "latent_traits": latent_traits if latent_traits else None
        }
        
        # Add past failure warnings if any
        if past_failure_warnings:
            response["past_failure_warnings"] = past_failure_warnings
        
        # v73: Add psychological extraction data or prompt
        if psych_extraction:
            response["psychological_extraction"] = psych_extraction
            response["instructions"] += "\n\n📊 PSYCHOLOGICAL CONTEXT: User shows " + \
                ("hedging (reduce confidence)" if psych_extraction.get("hedging_detected") else "confidence") + \
                f". Priority: {psych_extraction.get('inferred_priority', 'unknown')}."
        elif psych_prompt:
            response["psychological_extraction_prompt"] = psych_prompt
            response["instructions"] += "\n\n⚠️ PSYCHOLOGICAL ANALYSIS RECOMMENDED: Process the extraction prompt to understand user intent before generating hypotheses."
        
        # =====================================================================
        # Helper 5: Extract symbol suggestions (v38c)
        # =====================================================================
        suggested_lookups, instruction_add = extract_symbol_suggestions(
            cur, project_id, session["goal"], parent_content
        )
        if suggested_lookups:
            response["suggested_lookups"] = suggested_lookups
            response["instructions"] += instruction_add
        
        # =====================================================================
        # v41: Preflight Enforcement - SQL Detection
        # =====================================================================
        try:
            from pas.helpers.preflight import detect_sql_operations, log_tool_call
            
            combined_text = f"{session['goal']} {parent_content}"
            if detect_sql_operations(combined_text):
                response["schema_check_required"] = True
                response["instructions"] += "\n\n⚠️ SQL OPERATIONS DETECTED: Call get_self_awareness() to verify schema before writing SQL queries."
                logger.info("v41: SQL operations detected, schema_check_required=True")
            
            # Log this prepare_expansion call
            log_tool_call(cur, session_id, "prepare_expansion", {
                "has_suggested_lookups": "suggested_lookups" in response,
                "has_failure_warnings": "past_failure_warnings" in response,
                "schema_check_required": response.get("schema_check_required", False),
                "has_project_id": project_id is not None
            })
            conn.commit()
        except Exception as e:
            logger.warning(f"v41: Preflight detection failed (non-fatal): {e}")
        
        # =====================================================================
        # Helper 6: Search related modules (v42a)
        # =====================================================================
        related_modules, instruction_add = search_related_modules(
            cur, project_id, session["goal"], get_embedding
        )
        if related_modules:
            response["related_modules"] = related_modules
            response["instructions"] += instruction_add
        
        # =====================================================================
        # Helper 7: Fetch project grounding (v43)
        # =====================================================================
        project_grounding, instruction_add = fetch_project_grounding(cur, project_id)
        if project_grounding:
            response["project_grounding"] = project_grounding
            response["instructions"] += instruction_add
        
        # =====================================================================
        # Helper 8: Fetch historical patterns (v44d)
        # =====================================================================
        historical_patterns = fetch_historical_patterns(
            cur, session_id, session["goal"], get_embedding
        )
        if historical_patterns:
            response["historical_patterns"] = historical_patterns
        
        # =====================================================================
        # v60: Surface calibration warning if overconfidence detected
        # =====================================================================
        calibration_ctx = fetch_calibration_context(cur, project_id)
        if calibration_ctx["should_apply_decay"]:
            stats = calibration_ctx["stats"]
            response["calibration_warning"] = {
                "message": f"Overconfidence bias detected (+{stats.get('overconfidence_bias', 0):.3f}). Stated confidences will be reduced by decay factor.",
                "bias": stats.get("overconfidence_bias"),
                "decay_will_apply": True,
                "domain": calibration_ctx["domain"],
                "samples": stats.get("sample_count")
            }
        
        # =====================================================================
        # Phase 7d: Active Law Application - Law Analysis Prompts
        # =====================================================================
        response["phase_7d_section_reached"] = True  # DEBUG BEFORE TRY
        try:
            from pas.helpers.expansion import build_law_analysis_prompt
            
            logger.debug(f"Phase 7d: laws count = {len(laws) if laws else 0}, has session = {session is not None}")
            response["phase_7d_reached"] = True  # DEBUG: confirm this code block is reached
            if laws:
                # Build law-specific analysis prompts
                law_analysis_prompts = [
                    build_law_analysis_prompt(law, session["goal"])
                    for law in laws[:3]  # Top 3 matched laws
                ]
                
                # Store matched law IDs in session context for gating at store_expansion
                matched_law_ids = [law["id"] for law in laws[:3]]
                cur.execute("""
                    UPDATE reasoning_sessions 
                    SET context = COALESCE(context, '{}'::jsonb) || jsonb_build_object('matched_laws', %s::jsonb)
                    WHERE id = %s
                """, (json.dumps(matched_law_ids), session_id))
                conn.commit()
                
                response["law_analysis_prompts"] = law_analysis_prompts
                response["law_analysis_required"] = True
                response["instructions"] += "\n\n🎯 LAW ANALYSIS REQUIRED: Before calling store_expansion, you MUST answer the questions in law_analysis_prompts to ensure laws actively inform your hypotheses."
                
                logger.info(f"Phase 7d: Generated {len(law_analysis_prompts)} law analysis prompt(s)")
        except Exception as e:
            logger.warning(f"Phase 7d: Law analysis prompt generation failed (non-fatal): {e}")
            response["phase_7d_error"] = str(e)  # DEBUG: Add to response for visibility
        
        # =====================================================================
        # Phase 7c: Surface Active Constraints
        # =====================================================================
        try:
            from pas.helpers.constraint_validation import get_constraint_summary
            if project_id:
                constraint_summary = get_constraint_summary(project_id)
                if constraint_summary["constraint_count"] > 0:
                    response["active_constraints"] = constraint_summary["constraints"]
                    response["instructions"] += f"\n\n{constraint_summary['constraint_prompt']}"
                    if constraint_summary["blocking_count"] > 0:
                        response["instructions"] += f"\n⚠️ {constraint_summary['blocking_count']} BLOCKING constraint(s) active - violations will be rejected."
                    logger.info(f"Phase 7c: Surfaced {constraint_summary['constraint_count']} active constraint(s)")
        except Exception as e:
            logger.warning(f"Phase 7c: Constraint surfacing failed (non-fatal): {e}")
        
        return response



        
    except Exception as e:
        logger.error(f"prepare_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_law_analysis(
    session_id: str,
    law_analyses: str,  # JSON array: [{"law_id": 1, "answers": ["answer1", "answer2"]}]
) -> dict[str, Any]:
    """
    Phase 7d: Store agent's law analysis answers. Required before store_expansion when laws matched.
    
    After calling prepare_expansion, if law_analysis_prompts is returned, you MUST:
    1. Process each law's questions
    2. Call this tool with your answers
    3. THEN call store_expansion
    
    Args:
        session_id: The reasoning session UUID
        law_analyses: JSON array of analysis results per law
        
    Returns:
        Confirmation with next step
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Parse the analyses
        try:
            analyses = json.loads(law_analyses)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON for law_analyses: {e}"}
        
        if not analyses:
            return {"success": False, "error": "law_analyses cannot be empty"}
        
        # Store in session context
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = COALESCE(context, '{}'::jsonb) || jsonb_build_object('law_analyses', %s::jsonb)
            WHERE id = %s
            RETURNING id
        """, (json.dumps(analyses), session_id))
        
        if cur.rowcount == 0:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        conn.commit()
        
        logger.info(f"Phase 7d: Stored {len(analyses)} law analysis/analyses for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "analyses_stored": len(analyses),
            "message": "Law analyses stored. You may now call store_expansion.",
            "next_step": "Call store_expansion with your hypotheses"
        }
        
    except Exception as e:
        logger.error(f"store_law_analysis failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_expansion(

    session_id: str,
    parent_node_id: str | None,
    # Hypothesis 1 (required)
    h1_text: str,
    h1_confidence: float,
    h1_scope: Optional[str] = None,
    # Hypothesis 2 (optional)
    h2_text: Optional[str] = None,
    h2_confidence: Optional[float] = None,
    h2_scope: Optional[str] = None,
    # Hypothesis 3 (optional)
    h3_text: Optional[str] = None,
    h3_confidence: Optional[float] = None,
    h3_scope: Optional[str] = None,
    # v51: Effort-Benefit scoring (1=low, 2=medium, 3=high)
    h1_effort: Optional[int] = None,
    h1_benefit: Optional[int] = None,
    h2_effort: Optional[int] = None,
    h2_benefit: Optional[int] = None,
    h3_effort: Optional[int] = None,
    h3_benefit: Optional[int] = None,
    # v7b: Revision tracking (borrowed from sequential thinking)
    is_revision: bool = False,
    revises_node_id: Optional[str] = None,
    # v25: Conversation logging
    source_text: Optional[str] = None,
    user_id: Optional[str] = None,
    # v41: Preflight enforcement
    skip_preflight: bool = False,
    # Phase 7c: Constraint validation
    project_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Store generated hypotheses with Bayesian scoring.
    
    Uses flattened parameters for universal LLM compatibility.
    At least one hypothesis (h1) is required, h2 and h3 are optional.
    
    Args:
        session_id: The reasoning session UUID
        parent_node_id: Parent node UUID (None if expanding from goal)
        h1_text: First hypothesis text (required)
        h1_confidence: First hypothesis confidence 0.0-1.0 (required)
        h1_scope: First hypothesis affected scope (comma-separated files/modules)
        h2_text: Second hypothesis text (optional)
        h2_confidence: Second hypothesis confidence 0.0-1.0 (optional)
        h2_scope: Second hypothesis affected scope
        h3_text: Third hypothesis text (optional)
        h3_confidence: Third hypothesis confidence 0.0-1.0 (optional)
        h3_scope: Third hypothesis affected scope
        h1_effort-h3_effort: Optional effort estimates (1=low, 2=medium, 3=high)
        h1_benefit-h3_benefit: Optional benefit estimates (1=low, 2=medium, 3=high)
        is_revision: True if these hypotheses revise previous thinking (v7b)
        revises_node_id: Node ID being revised, if is_revision is True
        source_text: v25 - Raw user input that inspired this hypothesis (logged for semantic search)
        user_id: v25 - Optional user identifier for multi-user scenarios
        
    Returns:
        Created nodes with Bayesian posterior scores, declared scopes, and revision info
    """
    try:
        # Build hypotheses list from flattened params - Phase 2 Refactor
        hypotheses = build_hypotheses_list(
            h1_text, h1_confidence, h1_scope,
            h2_text, h2_confidence, h2_scope,
            h3_text, h3_confidence, h3_scope,
            h1_effort, h1_benefit,
            h2_effort, h2_benefit,
            h3_effort, h3_benefit
        )
        
        if not hypotheses:
            return {"success": False, "error": "At least one hypothesis (h1_text) is required"}
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify session - Phase 2 Refactor
        session_error = verify_active_session(cur, session_id)
        if session_error:
            return session_error
        
        # =====================================================================
        # Phase 7d: Law Analysis Gate
        # =====================================================================
        cur.execute("""
            SELECT 
                context->'matched_laws' as matched_laws,
                context->'law_analyses' as law_analyses
            FROM reasoning_sessions WHERE id = %s
        """, (session_id,))
        ctx_row = cur.fetchone()
        
        if ctx_row:
            matched_laws = ctx_row["matched_laws"] if ctx_row["matched_laws"] else []
            law_analyses = ctx_row["law_analyses"] if ctx_row["law_analyses"] else []
            
            if matched_laws and not law_analyses:
                # Laws were matched but no analysis provided - return soft warning
                return {
                    "success": False,
                    "error": "law_analysis_required",
                    "message": f"{len(matched_laws)} law(s) matched but no analysis provided",
                    "matched_law_ids": matched_laws[:5],
                    "required_action": "Call store_law_analysis with your law analysis answers first",
                    "suggestion": "Review law_analysis_prompts from prepare_expansion and answer each law's questions"
                }
        
        # =====================================================================
        # Phase 7c: Constraint Validation
        # =====================================================================

        constraint_validation = {"checked": False, "violations": [], "warnings": []}
        if project_id:
            try:
                from pas.helpers.constraint_validation import validate_hypothesis
                all_violations = []
                all_warnings = []
                
                for hyp in hypotheses:
                    h_text = hyp.get("hypothesis", "")
                    if h_text:
                        validation = validate_hypothesis(h_text, project_id)
                        all_violations.extend(validation.get("violations", []))
                        all_warnings.extend(validation.get("warnings", []))
                
                constraint_validation = {
                    "checked": True,
                    "violations": all_violations,
                    "warnings": all_warnings,
                    "blocked": len(all_violations) > 0
                }
                
                # Reject if any blocking violations
                if constraint_validation["blocked"]:
                    logger.warning(f"Phase 7c: Hypothesis rejected - constraint violation: {all_violations[0]}")
                    return {
                        "success": False,
                        "error": "Hypothesis violates blocking constraint",
                        "constraint_validation": constraint_validation,
                        "suggestion": "Rephrase hypotheses to avoid: " + ", ".join(
                            v.get("pattern_matched", "unknown") for v in all_violations
                        )
                    }
                
                if all_warnings:
                    logger.info(f"Phase 7c: {len(all_warnings)} constraint warning(s) for hypotheses")
            except Exception as e:
                logger.warning(f"Phase 7c: Constraint validation failed (non-fatal): {e}")
        
        # Determine parent path - Phase 2 Refactor
        parent_path, parent_node_id, parent_error = resolve_parent_path(
            cur, session_id, parent_node_id, get_embedding
        )
        if parent_error:
            return parent_error
        
        # v60: Fetch calibration context for decay application (once per call)
        from pas.helpers.calibration import compute_confidence_decay
        calibration_ctx = fetch_calibration_context(cur, project_id=project_id)
        decay_applied_info = None
        
        created_nodes = []
        for i, hyp in enumerate(hypotheses[:3]):
            hypothesis_text = str(hyp.get("hypothesis", ""))
            _conf = hyp.get("confidence")
            llm_confidence = float(_conf) if _conf is not None else 0.5  # type: ignore[arg-type]
            declared_scope = hyp.get("scope")
            
            if not hypothesis_text:
                continue
            
            # v60: Apply calibration decay if overconfidence detected
            if calibration_ctx["should_apply_decay"]:
                llm_confidence, decay_info = compute_confidence_decay(
                    llm_confidence,
                    calibration_ctx["stats"].get("overconfidence_bias", 0)
                )
                if decay_info.get("applied"):
                    decay_applied_info = decay_info  # Track for response
            
            # Generate embedding and find similar law - Phase 2 Refactor
            hyp_emb = get_embedding(hypothesis_text)
            prior, supporting_law, law_name = match_laws_and_compute_prior(
                cur, hypothesis_text, hyp_emb, _compute_ensemble_prior
            )
            
            likelihood = llm_confidence
            
            # v60: Build metadata with ROI if provided
            roi_data = hyp.get("roi")
            metadata_json = json.dumps({"roi": roi_data}) if roi_data else None
            
            # Insert node
            node_id = str(uuid.uuid4())
            new_path = f"{parent_path}.h{i+1}"
            
            cur.execute(
                """
                INSERT INTO thought_nodes (id, session_id, path, content, node_type, prior_score, likelihood, embedding, supporting_laws, declared_scope, metadata)
                VALUES (%s, %s, %s, %s, 'hypothesis', %s, %s, %s, %s, %s, %s)
                RETURNING id, path, prior_score, likelihood, posterior_score
                """,
                (node_id, session_id, new_path, hypothesis_text, prior, likelihood, hyp_emb, supporting_law, declared_scope, metadata_json)
            )
            node = cur.fetchone()
            
            created_nodes.append({
                "node_id": str(node["id"]),
                "path": node["path"],
                "content": hypothesis_text[:150] + "..." if len(hypothesis_text) > 150 else hypothesis_text,
                "prior_score": float(node["prior_score"]),
                "likelihood": float(node["likelihood"]),
                "posterior_score": float(node["posterior_score"]) if node["posterior_score"] else None,
                "supporting_law": law_name,
                "declared_scope": declared_scope,
                "roi": roi_data  # v51: Include ROI in response if provided
            })
        
        # =====================================================================
        # v25: Conversation Logging - Phase 2 Refactor
        # =====================================================================
        conversation_log_id = None
        if source_text and created_nodes:
            conversation_log_id = log_conversation_source(
                cur, session_id, source_text, user_id, 
                created_nodes[0]["node_id"], get_embedding
            )
        
        conn.commit()
        
        # Workflow nudges - Phase 2 Refactor
        nudges = compute_workflow_nudges(created_nodes, is_revision, revises_node_id)
        next_step = nudges["next_step"]
        confidence_nudge = nudges["confidence_nudge"]
        revision_info = nudges["revision_info"]
        revision_nudge = nudges["revision_nudge"]
        
        # v21: Scope-Based Failure Matching - Phase 2 Refactor
        scope_warnings = query_scope_failures(cur, conn, created_nodes)
        
        # v41: Preflight Enforcement Check - Phase 2 Refactor
        preflight_warnings, preflight_bypassed = run_preflight_checks(
            cur, conn, session_id, skip_preflight, created_nodes
        )
        
        # v42b: Scope-Based Failure Surfacing - Phase 2 Refactor
        scope_failure_warnings = surface_scope_failures(
            cur, conn, session_id, created_nodes, _search_relevant_failures
        )
        
        return {
            "success": True, 
            "session_id": session_id, 
            "created_nodes": created_nodes, 
            "count": len(created_nodes),
            "next_step": next_step,
            "confidence_nudge": confidence_nudge,
            "revision_info": revision_info,
            "revision_nudge": revision_nudge,
            "scope_warnings": scope_warnings if scope_warnings else None,  # v21: Historical failure warnings
            "scope_failure_warnings": scope_failure_warnings if scope_failure_warnings else None,  # v42b: Scope-based surfacing
            "preflight_warnings": preflight_warnings if preflight_warnings else None,  # v41: Preflight enforcement
            "preflight_bypassed": preflight_bypassed if preflight_bypassed else None,  # v41: Track bypasses
            "decay_applied": decay_applied_info,  # v60: Calibration decay info
            "constraint_validation": constraint_validation if constraint_validation.get("checked") else None  # Phase 7c
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_expansion failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def prepare_critique(
    node_id: str,
    critique_mode: str = "standard"  # v31a: 'standard' or 'negative_space'
) -> dict[str, Any]:
    """
    Prepare context for critiquing a thought node. Returns node content and 
    supporting laws so the LLM can generate counterarguments.
    
    After calling this, generate a critique and pass it to store_critique().
    
    Args:
        node_id: The thought node UUID to critique
        critique_mode: v31a - 'standard' (what's wrong?) or 'negative_space' (what's missing?)
        
    Returns:
        Context dict with node_content, supporting_laws for critique generation
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Delegate to helpers
        node, laws_text = fetch_node_with_laws(cur, node_id)
        if not node:
            return {"success": False, "error": f"Node {node_id} not found"}
        
        # Build critique prompt
        prompt, system, expected_format = build_critique_prompt(
            node["content"], node["goal"], laws_text, critique_mode
        )
        llm_prompt = {"prompt": prompt, "system": system, "expected_format": expected_format}
        logger.info(f"v32: Returning prompt for agent to process (mode: {critique_mode})")
        
        # Search past failures and critiques
        past_failures = search_past_failures(cur, conn, node["content"], node_id)
        past_critiques = search_past_critiques(cur, node["content"], node_id)
        
        # Build assumption extraction prompt
        assumption_extraction_prompt = build_assumption_extraction_prompt(node["content"])
        
        return {
            "success": True,
            "node_id": node_id,
            "path": node["path"],
            "node_content": node["content"],
            "session_goal": node["goal"],
            "critique_mode": critique_mode,
            "current_scores": {
                "prior": float(node["prior_score"]),
                "likelihood": float(node["likelihood"]),
                "posterior": float(node["posterior_score"]) if node["posterior_score"] else None
            },
            "supporting_laws": laws_text,
            "llm_prompt": llm_prompt,
            "suggested_critique": None,
            "negative_space_gaps": None,
            "past_failures": past_failures,
            "assumption_extraction_prompt": assumption_extraction_prompt,
            "instructions": f"Process the llm_prompt to generate critique. Mode: {critique_mode}. After generating, call store_critique(node_id='{node_id}', counterargument=..., severity_score=..., logical_flaws='...', edge_cases='...').",
            "constitutional_principles": CONSTITUTIONAL_PRINCIPLES,
            "critic_personas": CRITIC_PERSONAS,
            "aggregation_guidance": AGGREGATION_GUIDANCE,
            "past_critiques": past_critiques  # v42b
        }
        
    except Exception as e:
        logger.error(f"prepare_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if conn:
            safe_close_connection(conn)


@mcp.tool()

async def store_critique(
    node_id: str,
    counterargument: str,
    severity_score: float,
    logical_flaws: str = "",
    edge_cases: str = "",
    major_flaws: str = "",  # v9a: Major flaws that invalidate hypothesis
    minor_flaws: str = ""   # v9a: Minor concerns worth noting
) -> dict[str, Any]:
    """
    Store critique results and update node likelihood.
    
    Uses flattened parameters for universal LLM compatibility.
    
    Args:
        node_id: The thought node UUID
        counterargument: Main counter-argument text
        severity_score: Impact score 0.0-1.0 (higher = more severe critique)
        logical_flaws: Comma or newline separated list of flaws (optional)
        edge_cases: Comma or newline separated list of edge cases (optional)
        major_flaws: v9a - Major issues that would invalidate the hypothesis (comma-separated)
        minor_flaws: v9a - Minor concerns worth noting but not blocking (comma-separated)
        
    Returns:
        Updated node scores
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT likelihood FROM thought_nodes WHERE id = %s", (node_id,))
        node = cur.fetchone()
        if not node:
            return {"success": False, "error": "Node not found"}
        
        severity = float(severity_score)
        old_likelihood = float(node["likelihood"])
        
        # v9a: Tiered penalty calculation
        MAJOR_PENALTY = 0.15  # Each major flaw has significant impact
        MINOR_PENALTY = 0.03  # Each minor flaw has small impact
        
        major_count = len([f for f in major_flaws.replace('\n', ',').split(',') if f.strip()])
        minor_count = len([f for f in minor_flaws.replace('\n', ',').split(',') if f.strip()])
        
        if major_count > 0 or minor_count > 0:
            # v9a: Use tiered penalty calculation
            total_penalty = (major_count * MAJOR_PENALTY) + (minor_count * MINOR_PENALTY)
            new_likelihood = max(0.1, old_likelihood * (1 - min(total_penalty, 0.8)))
            tier_breakdown = {
                "major_count": major_count,
                "minor_count": minor_count,
                "total_penalty": round(total_penalty, 3),
                "method": "tiered"
            }
        else:
            # Legacy: fallback to severity_score
            new_likelihood = max(0.1, old_likelihood * (1 - severity * 0.5))
            tier_breakdown = {
                "major_count": 0,
                "minor_count": 0,
                "total_penalty": round(severity * 0.5, 3),
                "method": "legacy_severity"
            }
        
        # v52: Persist critique data to metadata for checklist validation
        critique_data = {
            "counterargument": counterargument[:500],
            "major_flaws": [f.strip() for f in major_flaws.replace('\n', ',').split(',') if f.strip()],
            "minor_flaws": [f.strip() for f in minor_flaws.replace('\n', ',').split(',') if f.strip()],
            "edge_cases": [c.strip() for c in edge_cases.replace('\n', ',').split(',') if c.strip()],
            "severity_score": severity
        }
        cur.execute(
            """
            UPDATE thought_nodes 
            SET likelihood = %s, 
                updated_at = NOW(),
                metadata = metadata || %s::jsonb
            WHERE id = %s 
            RETURNING prior_score, likelihood, posterior_score
            """,
            (new_likelihood, json.dumps({"critique": critique_data}), node_id)
        )
        updated = cur.fetchone()
        conn.commit()
        
        # Count flaws/cases for summary (simple split)
        flaw_count = len([f for f in logical_flaws.replace('\n', ',').split(',') if f.strip()])
        case_count = len([c for c in edge_cases.replace('\n', ',').split(',') if c.strip()])
        
        return {
            "success": True,
            "node_id": node_id,
            "critique_summary": {
                "counterargument": counterargument[:200],
                "flaw_count": flaw_count,
                "edge_case_count": case_count,
                "severity_score": severity
            },
            "score_update": {
                "old_likelihood": old_likelihood,
                "new_likelihood": float(updated["likelihood"]),
                "posterior_score": float(updated["posterior_score"]) if updated["posterior_score"] else None
            },
            # v9a: Tier breakdown for transparency
            "tier_breakdown": tier_breakdown
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_critique failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def prepare_sequential_analysis(
    session_id: str,
    top_n: int = 5
) -> dict[str, Any]:
    """
    v32: Prepare adversarial gap analysis prompts for top candidates.
    
    Returns prompts for agent to process. Agent should analyze each
    candidate and call store_sequential_analysis with results.
    
    Args:
        session_id: The reasoning session UUID
        top_n: Number of top candidates to analyze (max 5)
        
    Returns:
        Prompts for each candidate for agent to process
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session goal
        cur.execute(
            "SELECT goal FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Get top candidates
        cur.execute(
            """
            SELECT id, content, posterior_score
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            ORDER BY posterior_score DESC
            LIMIT %s
            """,
            (session_id, min(top_n, 5))
        )
        candidates = cur.fetchall()
        
        if not candidates:
            return {"success": False, "error": "No candidates to analyze"}
        
        # v32 FIX: Return prompts for agent to process (no MCP sampling)
        candidate_prompts = []
        for candidate in candidates:
            seq_prompt = f"""What gaps exist between this hypothesis and the goal?

GOAL: {session["goal"]}
HYPOTHESIS: {candidate["content"]}

Return ONLY valid JSON:
{{"gaps": ["gap1", "gap2"], "revisions_needed": ["revision1"]}}

Focus on what's MISSING, not what's wrong."""
            
            candidate_prompts.append({
                "node_id": str(candidate["id"]),
                "content": candidate["content"][:100],
                "score": float(candidate["posterior_score"]) if candidate["posterior_score"] else 0.0,
                "prompt": seq_prompt
            })
        
        logger.info(f"v32: Returning {len(candidate_prompts)} prompts for sequential analysis")
        
        return {
            "success": True,
            "session_id": session_id,
            "system_prompt": "Find what's MISSING. Be adversarial. Return only valid JSON.",
            "candidate_prompts": candidate_prompts,
            "instructions": f"Process each prompt and call store_sequential_analysis(session_id='{session_id}', results='[{{\"node_id\": \"...\", \"gaps\": [...], \"revisions_needed\": [...]}}]')"
        }
        
    except Exception as e:
        logger.error(f"prepare_sequential_analysis failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_sequential_analysis(
    session_id: str,
    results: str  # JSON array: [{"node_id": "...", "gaps": [...], "revisions_needed": [...]}]
) -> dict[str, Any]:
    """
    v32: Store sequential analysis results from agent.
    
    Called after agent processes prompts from prepare_sequential_analysis.
    Aggregates systemic gaps and stores for finalize_session.
    
    Args:
        session_id: The reasoning session UUID
        results: JSON array of gap analysis per candidate
        
    Returns:
        Confirmation with systemic gaps detected
    """
    try:
        # Parse results
        parsed_results = json.loads(results)
        
        if not isinstance(parsed_results, list):
            return {"success": False, "error": "results must be a JSON array"}
        
        # Collect all gaps for systemic detection
        all_gaps = []
        for result in parsed_results:
            gaps = result.get("gaps", [])
            if isinstance(gaps, list):
                all_gaps.extend(gaps)
        
        # Detect systemic gaps (>50% candidates have same gap)
        systemic_gaps = []
        if all_gaps and len(parsed_results) > 1:
            from collections import Counter
            gap_counts = Counter(all_gaps)
            threshold = len(parsed_results) * 0.5
            systemic_gaps = [g for g, c in gap_counts.items() if c >= threshold]
        
        # v37: Mark nodes as sequential_analyzed in DB for finalize_session check
        conn = get_db_connection()
        cur = conn.cursor()
        for result in parsed_results:
            node_id = result.get("node_id")
            if node_id:
                cur.execute("""
                    UPDATE thought_nodes 
                    SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"sequential_analyzed": "true"}'::jsonb
                    WHERE id = %s
                """, (node_id,))
        
        # v54: Mark session-level flag for negative space enforcement
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = COALESCE(context, '{}'::jsonb) || '{"sequential_analysis_called": true}'::jsonb
            WHERE id = %s
        """, (session_id,))
        
        conn.commit()
        safe_close_connection(conn)
        
        logger.info(f"v32: Stored sequential analysis - {len(parsed_results)} candidates, {len(systemic_gaps)} systemic gaps")
        
        return {
            "success": True,
            "session_id": session_id,
            "candidates_analyzed": len(parsed_results),
            "sequential_analysis": parsed_results,
            "systemic_gaps": systemic_gaps,
            "next_step": f"Call finalize_session(session_id='{session_id}')"
        }
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"store_sequential_analysis failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_reasoning_tree(session_id: str, max_depth: int = 10) -> dict[str, Any]:
    """
    Get the full reasoning tree for a session.
    
    Args:
        session_id: The reasoning session UUID
        max_depth: Maximum tree depth to retrieve
        
    Returns:
        All thought nodes ordered by path
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT id, path, content, node_type, depth, prior_score, likelihood, posterior_score, created_at
            FROM thought_nodes WHERE session_id = %s AND depth <= %s ORDER BY path
            """,
            (session_id, max_depth)
        )
        
        nodes = []
        for r in cur.fetchall():
            nodes.append({
                "id": str(r["id"]), "path": r["path"], "content": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"],
                "type": r["node_type"], "depth": r["depth"],
                "posterior": float(r["posterior_score"]) if r["posterior_score"] else None
            })
        
        return {"success": True, "session_id": session_id, "nodes": nodes, "count": len(nodes)}
        
    except Exception as e:
        logger.error(f"get_reasoning_tree failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_best_path(session_id: str) -> dict[str, Any]:
    """
    Find the highest-scoring reasoning path through the tree.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        The best leaf node and its ancestor chain
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find best leaf (node with highest posterior that has no children)
        cur.execute(
            """
            SELECT t1.id, t1.path, t1.content, t1.posterior_score, t1.depth
            FROM thought_nodes t1
            WHERE t1.session_id = %s
            AND NOT EXISTS (SELECT 1 FROM thought_nodes t2 WHERE t2.session_id = t1.session_id AND t2.path <@ t1.path AND t2.path != t1.path)
            ORDER BY t1.posterior_score DESC NULLS LAST LIMIT 1
            """,
            (session_id,)
        )
        best = cur.fetchone()
        if not best:
            return {"success": False, "error": "No thoughts in session"}
        
        # Get ancestors
        cur.execute(
            "SELECT id, path, content, posterior_score, depth FROM thought_nodes WHERE session_id = %s AND %s <@ path ORDER BY depth",
            (session_id, best["path"])
        )
        
        path_nodes = [{"id": str(r["id"]), "path": r["path"], "content": r["content"][:80], "score": float(r["posterior_score"]) if r["posterior_score"] else None} for r in cur.fetchall()]
        
        return {
            "success": True,
            "best_node_id": str(best["id"]),
            "best_score": float(best["posterior_score"]) if best["posterior_score"] else None,
            "path": path_nodes
        }
        
    except Exception as e:
        logger.error(f"get_best_path failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v39: Interview constants and get_interview_context moved to interview_helpers.py
# - DEFAULT_INTERVIEW_CONFIG, DEFAULT_QUALITY_THRESHOLDS, get_interview_context
# =============================================================================


def archive_interview_to_history(cur, session_id: str, goal: str, interview: dict):
    """
    Archive completed interview Q&A to interview_history table for self-learning.
    
    Extracts domain from goal and stores each answered question with its context.
    """
    from pas.engine import get_embedding
    
    # Extract domain from goal (first meaningful word after common prefixes)
    goal_lower = goal.lower()
    domain = "general"
    domain_keywords = {
        "ui": "ui_design", "ux": "ui_design", "dashboard": "ui_design", "widget": "ui_design",
        "cache": "architecture", "api": "architecture", "database": "architecture", "schema": "architecture",
        "debug": "debugging", "fix": "debugging", "error": "debugging", "bug": "debugging",
        "test": "testing", "verify": "testing", "validate": "testing",
        "design": "design", "architect": "architecture", "pattern": "patterns"
    }
    for keyword, dom in domain_keywords.items():
        if keyword in goal_lower:
            domain = dom
            break
    
    pending = interview.get("pending_questions", [])
    answered_questions = [q for q in pending if q.get("answered")]
    
    for position, q in enumerate(answered_questions, 1):
        try:
            # Get embedding for semantic clustering
            embedding = get_embedding(q.get("text") or q.get("question_text", ""))
            
            # Determine if follow-up was triggered
            follow_up_triggered = False
            for rule in q.get("follow_up_rules", []):
                if rule.get("when_answer") == q.get("answer"):
                    follow_up_triggered = True
                    break
            
            # Find answer text from choices
            answer_text = None
            answer_given = q.get("answer")
            if answer_given and "choices" in q:
                for choice in q["choices"]:
                    if choice.get("label") == answer_given:
                        answer_text = choice.get("text", "")
                        break
            
            cur.execute(
                """
                INSERT INTO interview_history 
                (session_id, question_id, question_text, question_embedding, 
                 question_category, answer_given, answer_text, position_in_flow,
                 was_skipped, follow_up_triggered, goal_domain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    q.get("id", f"q_{position}"),
                    q.get("text") or q.get("question_text", ""),
                    embedding,
                    q.get("category", "general"),
                    answer_given,
                    answer_text,
                    position,
                    False,  # was_skipped
                    follow_up_triggered,
                    domain
                )
            )
        except Exception as e:
            logger.warning(f"Failed to archive question {q.get('id')}: {e}")
            continue
    
    logger.info(f"Archived {len(answered_questions)} interview questions for session {session_id}")


@mcp.tool()
async def identify_gaps(session_id: str) -> dict[str, Any]:
    """
    Analyze session goal and context to identify clarifying questions needed.
    
    Generates structured questions with multiple choice answers, pros/cons,
    and follow-up rules. Questions are prioritized and limited by safety config.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Interview config and generated questions, or indication that no gaps exist
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session with session_mode
        cur.execute("SELECT goal, context, session_mode FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        goal = session["goal"]
        context = session["context"] or {}
        session_mode = session.get("session_mode", "implementation")
        interview = get_interview_context(context)
        
        # Check if interview already has questions
        if interview["pending_questions"]:
            return {
                "success": True,
                "session_id": session_id,
                "status": "interview_in_progress",
                "interview_config": interview["config"],
                "pending_count": len([q for q in interview["pending_questions"] if not q.get("answered")]),
                "message": "Interview already has pending questions. Use get_next_question to continue."
            }
        
        # v16d.2: Query historical failures (skip for constraint_discovery - those questions are for LLM agents)
        historical_questions = []
        if session_mode != "constraint_discovery":
            historical_questions = query_historical_failures(cur, conn, session_id, logger)
        
        # v76: Check for constraint_discovery mode - prioritize project_constraints dimension
        domain_questions = []
        detected_domains = []
        
        if session_mode == "constraint_discovery":
            # Load questions from project_constraints dimension specifically
            cur.execute("""
                SELECT q.id, q.question_template, q.question_type, q.choices, q.constraint_mapping,
                       d.id as dimension_id, d.dimension_name
                FROM interview_questions q
                JOIN interview_dimensions d ON q.dimension_id = d.id
                WHERE d.dimension_name = 'project_constraints'
                ORDER BY q.created_at
            """)
            constraint_rows = cur.fetchall()
            for row in constraint_rows:
                domain_questions.append({
                    "id": str(row["id"]),
                    "text": row["question_template"],
                    "type": row["question_type"],
                    "choices": row["choices"] or [],
                    "constraint_mapping": row["constraint_mapping"],
                    "dimension_id": str(row["dimension_id"]),
                    "dimension_name": row["dimension_name"]
                })
            logger.info(f"v76: Loaded {len(domain_questions)} constraint discovery questions")
            detected_domains = [{"id": None, "name": "constraint_setup"}]
            context["detected_domains"] = detected_domains
        else:
            # v19: Domain detection + dimension questions (normal flow)
            detected_domains = detect_domains(cur, session_id, logger)
            if detected_domains:
                context["detected_domains"] = detected_domains
                domain_ids = [d["id"] for d in detected_domains]
                domain_questions, dimension_coverage = load_dimension_questions(cur, domain_ids, context, logger)
                context["dimension_coverage"] = dimension_coverage
        
        # v21: LLM question prompt (agent processes later via store_gaps_questions)
        # build_goal_question_prompt(goal) - available but not awaited here
        
        # Prioritize and combine questions
        questions = prioritize_questions(domain_questions, [], historical_questions, goal)
        
        # =====================================================================
        # v73: Phase 7b - Confirmation Loop Questions
        # =====================================================================
        confirmation_questions = generate_confirmation_questions(
            session_context=context,
            hypotheses=None,  # No hypotheses at identify_gaps stage
            logger=logger
        )
        if confirmation_questions:
            # Prepend confirmation questions (highest priority)
            questions = confirmation_questions + questions
            logger.info(f"v73: Added {len(confirmation_questions)} confirmation questions")
        
        # Update interview context
        interview["pending_questions"] = questions
        interview["config"]["questions_remaining"] = len(questions)
        
        # Save to session
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s, updated_at = NOW() WHERE id = %s",
            (json.dumps(context), session_id)
        )
        conn.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "interview_config": interview["config"],
            "questions_generated": len(questions),
            "detected_domains": [d["name"] for d in detected_domains] if detected_domains else None,
            "message": "Interview questions generated. Call get_next_question to start."
        }
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"identify_gaps failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        safe_close_connection(conn)


@mcp.tool()
async def get_next_question(session_id: str) -> dict[str, Any]:
    """
    Get the next unanswered question from the interview queue.
    
    Returns ONE question at a time with formatted choices including pros/cons.
    Includes progress indicator showing current position.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Single question with choices, or indication that interview is complete
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        context = session["context"] or {}
        interview = get_interview_context(context)
        
        # =====================================================================
        # v23: Check if early termination suggested
        # =====================================================================
        if interview.get("early_termination_suggested") and not interview.get("early_exit_declined"):
            remaining = len([q for q in interview.get("pending_questions", []) if not q.get("answered")])
            return {
                "success": True,
                "session_id": session_id,
                "early_exit_offered": True,
                "questions_remaining": remaining,
                "reason": interview.get("early_termination_reason", "diminishing_returns"),
                "message": "We've learned enough about your preferences. Continue (submit any answer) or proceed to prepare_expansion."
            }
        
        # Find first unanswered, unlocked question
        pending = interview.get("pending_questions", [])
        answered_ids = {q["id"] for q in pending if q.get("answered")}
        
        for q in sorted(pending, key=lambda x: x.get("priority", 999)):
            if q.get("answered"):
                continue

            # Check dependencies
            deps = q.get("depends_on", [])
            if all(d in answered_ids for d in deps):
                # This question is ready
                total_questions = len(pending)
                answered_count = interview["config"].get("questions_answered", 0)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "progress": f"Question {answered_count + 1} of ~{total_questions}",
                    "question": {
                        "id": q["id"],
                        "text": q.get("text") or q.get("question_text", ""),
                        "type": q.get("question_type", "single_choice"),
                        "choices": q.get("choices", [])  # v37 FIX: Defensive access
                    }
                }
        
        # No more questions
        return {
            "success": True,
            "session_id": session_id,
            "interview_complete": True,
            "message": "All questions answered. Ready for prepare_expansion."
        }
        
    except Exception as e:
        logger.error(f"get_next_question failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def submit_answer(session_id: str, question_id: str, answer: str) -> dict[str, Any]:
    """
    Submit an answer to an interview question.
    
    Updates the question as answered, stores in history, triggers context propagation
    including potential follow-up question injection (respecting depth/count limits).
    
    Args:
        session_id: The reasoning session UUID
        question_id: The question ID being answered
        answer: The selected answer (e.g., "A", "B", "C")
        
    Returns:
        Confirmation with any injected follow-up questions
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        context = session["context"] or {}
        interview = get_interview_context(context)
        pending = interview.get("pending_questions", [])
        config = interview["config"]
        
        # v23: If user continues after early exit offer, mark as declined
        if interview.get("early_termination_suggested"):
            interview["early_exit_declined"] = True
            logger.info("v23: User declined early exit, continuing interview")
        
        # Find and validate question
        question, error = find_and_validate_question(pending, question_id)
        if error:
            return {"success": False, "error": error}
        
        # Record answer and extract choice details
        config["questions_answered"] = config.get("questions_answered", 0) + 1
        hidden_value, _ = record_answer_with_choice(question, answer, interview)
        
        # Process side effects (v19 dimension tracking + follow-ups + v23 evidence)
        dimension_covered, injected = process_answer_side_effects(
            cur, conn, session_id, question, answer, hidden_value,
            pending, interview, config, context, logger
        )
        
        # Save to session
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s, updated_at = NOW() WHERE id = %s",
            (json.dumps(context), session_id)
        )
        conn.commit()
        
        return {
            "success": True,
            "session_id": session_id,
            "question_id": question_id,
            "answer_recorded": answer,
            "dimension_covered": dimension_covered,
            "questions_remaining": config["questions_remaining"],
            "follow_ups_injected": injected if injected else None,
            "message": f"Answer recorded. {config['questions_remaining']} questions remaining."
        }
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"submit_answer failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        safe_close_connection(conn)


@mcp.tool()
async def check_interview_complete(session_id: str) -> dict[str, Any]:
    """
    Check if the interview has gathered enough context to proceed.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Whether interview is complete and summary of gathered context
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT goal, context, session_mode FROM reasoning_sessions WHERE id = %s", (session_id,))
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        goal = session["goal"]
        context = session["context"] or {}
        session_mode = session.get("session_mode", "implementation")
        interview = get_interview_context(context)
        pending = interview.get("pending_questions", [])
        
        unanswered = [q for q in pending if not q.get("answered")]
        answered = [q for q in pending if q.get("answered")]
        is_complete = len(unanswered) == 0 and len(answered) > 0
        
        # Phase 5: Use extracted helper for context summary
        context_summary = build_context_summary(answered) if is_complete else {}
        
        # v21/v22 Trait inference (only when complete)
        latent_traits: list[dict[str, Any]] = []
        hidden_value_counts: dict[str, int] = {}
        
        if is_complete:
            # Phase 5: Use orchestrated helper for all trait inference
            latent_traits, hidden_value_counts = run_trait_inference(
                cur, interview, get_embedding
            )
            
            # Phase 5: Persist traits and archive interview
            persist_interview_context(
                cur, conn, session_id, goal, context, interview,
                latent_traits, hidden_value_counts, archive_interview_to_history
            )
        
        # -------------------------------------------------------------------------
        # v76: Constraint Discovery - Map answers to project_constraints
        # -------------------------------------------------------------------------
        constraints_created = 0
        if is_complete and session_mode == "constraint_discovery":
            from pas.helpers.interview import constraint_mapper
            
            # Gather interview answers from answer_history
            # v82: Include question_id for accurate constraint_mapping lookup
            interview_answers = []
            for q in answered:
                if q.get("dimension_id"):
                    interview_answers.append({
                        "dimension_id": q["dimension_id"],
                        "question_id": q.get("id"),  # v82: pass question_id for exact lookup
                        "answer_label": q.get("answer"),
                        "hidden_value": q.get("selected_hidden_value", "")
                    })
            
            constraints = constraint_mapper(
                session_id,
                interview_answers,
                latent_traits,
                cur
            )
            
            # Get project UUID from project_id string
            project_id_str = context.get("project_id", "default")
            cur.execute("SELECT id FROM project_registry WHERE project_id = %s", (project_id_str,))
            project_row = cur.fetchone()
            if not project_row:
                logger.warning(f"v76: Project not found: {project_id_str}, skipping constraint storage")
            else:
                project_uuid = project_row["id"]
                
                # Store constraints with correct column names
                for c in constraints:
                    constraint_data = json.dumps({
                        "value": c["value"],
                        "priority": c.get("priority"),
                        "source_dimension": c.get("source_dimension"),
                        "source_answer": c.get("source_answer"),
                        "source_session": c.get("source_session")
                    })
                    
                    # Expire existing constraint with same key (temporal versioning)
                    cur.execute("""
                        UPDATE project_constraints 
                        SET valid_to = NOW()
                        WHERE project_id = %s AND constraint_key = %s AND valid_to IS NULL
                    """, (project_uuid, c["key"]))
                    
                    # Insert new version
                    cur.execute("""
                        INSERT INTO project_constraints 
                            (project_id, constraint_type, constraint_key, constraint_data, enforcement_level, source)
                        VALUES (%s, %s, %s, %s, %s, 'inferred')
                    """, (
                        project_uuid, 
                        c["type"], 
                        c["key"], 
                        constraint_data,
                        c["enforcement"]
                    ))
                
                constraints_created = len(constraints)
                conn.commit()
                logger.info(f"v76: Created {constraints_created} constraints from interview")
            
            # v76: Auto-export to GEMINI.md if project_path available
            gemini_exported = False
            project_path = context.get("project_path")
            if project_path and constraints_created > 0:
                try:
                    from pas.helpers.gemini_sync import export_constraints_to_markdown, write_gemini_export
                    
                    export_result = export_constraints_to_markdown(project_id_str)
                    if export_result.get("success"):
                        write_result = write_gemini_export(
                            project_path=project_path,
                            content=export_result["export_content"],
                            mode="append"
                        )
                        gemini_exported = write_result.get("success", False)
                        if gemini_exported:
                            logger.info(f"v76: Auto-exported {constraints_created} constraints to GEMINI.md")
                except Exception as e:
                    logger.warning(f"v76: Failed to auto-export to GEMINI.md: {e}")
        
        return {
            "success": True,
            "session_id": session_id,
            "is_complete": is_complete,
            "questions_answered": len(answered),
            "questions_remaining": len(unanswered),
            "context_summary": context_summary if is_complete else None,
            "latent_traits": latent_traits if latent_traits else None,
            "hidden_value_counts": hidden_value_counts if hidden_value_counts else None,
            "archived_for_learning": interview.get("archived", False),
            "constraints_created": constraints_created if constraints_created > 0 else None,
            "gemini_exported": gemini_exported if constraints_created > 0 else None,
            "message": "Ready for prepare_expansion" if is_complete else f"{len(unanswered)} questions remaining"
        }
        
    except Exception as e:
        logger.error(f"check_interview_complete failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Finalization Tool - Auto-Critique & Recommendation (Phase 3)
# =============================================================================

HEURISTIC_PENALTIES = {
    "unchallenged": 0.10,      # Never critiqued
    "shallow_alternatives": 0.05,  # <2 siblings at same level
    "monoculture": 0.05,       # v13b: All siblings match same law
}

DEPTH_BONUS_PER_LEVEL = 0.02  # Reward deeper refinement

# =============================================================================
# v11a: UCT-based selection for close decisions
# =============================================================================
UCT_EXPLORATION_C = 1.4  # Standard exploration constant
UCT_THRESHOLD = 0.05     # Apply UCT when gap < threshold

# =============================================================================
# v11b: Law-grounded rollout configuration  
# =============================================================================
ROLLOUT_WEIGHT = 0.2     # Blend: final = (1-weight)*posterior + weight*rollout

# =============================================================================
# v35: TRAIT_RULES and _infer_traits_from_hidden_values moved to
#      helpers/interview.py in Phase 5 refactoring
# =============================================================================


# =============================================================================
# v39: Pure helper functions moved to reasoning_helpers.py
# - _apply_heuristic_penalties
# - _get_outcome_multiplier
# - _compute_critique_accuracy
# - _compute_law_effective_weight
# - _compute_ensemble_prior
# - _generate_suggested_tags
# - _compute_decision_quality
# - _apply_uct_tiebreaking
# - _build_processed_candidate
# =============================================================================


# =============================================================================
# v20: Adaptive Depth Quality Metrics
# =============================================================================

def compute_quality_metrics(
    cur,
    session_id: str,
    candidates: list,
    gap: float,
    thresholds: Optional[dict] = None
) -> dict:
    """
    Compute 4 quality signals to determine if reasoning is sufficient.
    
    Returns quality_sufficient flag, metrics breakdown, and suggestions.
    """
    import statistics
    
    thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS
    
    # 1. Gap score (already computed by caller)
    gap_ok = gap >= thresholds["gap_score"]
    
    # 2. Critique coverage - % of top candidates that have been critiqued
    # Critiqued nodes have likelihood reduced below their original confidence
    critiqued_count = 0
    uncritiqued_nodes = []
    for c in candidates[:3]:  # Check top 3
        node_id = c.get("node_id") or c.get("id")
        likelihood = c.get("likelihood", 0.5)
        # A node is considered critiqued if likelihood < initial confidence
        # (store_critique reduces likelihood, initial is typically 0.7-0.9)
        if likelihood < 0.8:  # Threshold: critiqued if reduced
            critiqued_count += 1
        else:
            if node_id:
                uncritiqued_nodes.append(str(node_id))
    
    critique_coverage = critiqued_count / max(len(candidates[:3]), 1)
    critique_ok = critique_coverage >= thresholds["critique_coverage"]
    
    # 3. Tree depth - maximum depth reached
    cur.execute("""
        SELECT MAX(depth) as max_depth FROM thought_nodes WHERE session_id = %s
    """, (session_id,))
    depth_result = cur.fetchone()
    max_depth = depth_result["max_depth"] if depth_result and depth_result["max_depth"] else 0
    depth_ok = max_depth >= thresholds["min_depth"]
    
    # 4. Confidence variance - stability of scores
    confidences = [c.get("likelihood", 0.5) for c in candidates]
    if len(confidences) > 1:
        variance = statistics.variance(confidences)
    else:
        variance = 0
    variance_ok = variance <= thresholds["max_confidence_variance"]
    
    # Compute overall sufficiency
    quality_sufficient = gap_ok and critique_ok and depth_ok and variance_ok
    
    # Generate suggestions for improvement
    suggestions = []
    if not critique_ok and uncritiqued_nodes:
        suggestions.append({
            "action": "critique",
            "node_id": uncritiqued_nodes[0],
            "reason": f"Top candidate unchallenged ({critiqued_count}/{len(candidates[:3])} critiqued)"
        })
    if not depth_ok:
        best_node_id = candidates[0].get("node_id") or candidates[0].get("id") if candidates else None
        if best_node_id:
            suggestions.append({
                "action": "expand",
                "node_id": str(best_node_id),
                "reason": f"Tree too shallow (depth {max_depth}, need ≥{thresholds['min_depth']})"
            })
    if not gap_ok:
        suggestions.append({
            "action": "expand_alternatives",
            "node_id": None,
            "reason": f"Gap too small ({gap:.3f}, need ≥{thresholds['gap_score']}). Explore more options."
        })
    if not variance_ok:
        suggestions.append({
            "action": "refine_confidences",
            "node_id": None,
            "reason": f"High uncertainty (variance {variance:.3f}). Add evidence or critique."
        })
    
    return {
        "sufficient": quality_sufficient,
        "metrics": {
            "gap_score": round(gap, 4),
            "gap_ok": gap_ok,
            "critique_coverage": round(critique_coverage, 2),
            "critique_ok": critique_ok,
            "tree_depth": max_depth,
            "depth_ok": depth_ok,
            "confidence_variance": round(variance, 4),
            "variance_ok": variance_ok
        },
        "suggestions": suggestions,
        "thresholds": thresholds
    }


@mcp.tool()
async def finalize_session(
    session_id: str,
    top_n: int = 3,
    deep_critique: bool = False,
    terminal_output: Optional[str] = None,  # v17b: RLVR auto-record
    # v31b: Exhaustive gap check (sequential-thinking style)
    exhaustive_check: bool = True,
    # v36: Quality thresholds from config (defaults from PAS_CONFIG)
    min_score_threshold: Optional[float] = None,
    min_gap_threshold: Optional[float] = None,
    # v33: Quality gate enforcement (opt-out, not opt-in)
    skip_quality_gate: bool = False,
    # v37: Sequential analysis enforcement (opt-out, not opt-in)
    skip_sequential_analysis: bool = False,
    # v82: Critique gate enforcement (Phase 4 hardening)
    skip_critique_check: bool = False,
    critique_bypass_reason: Optional[str] = None
) -> dict[str, Any]:
    """
    Finalize a reasoning session by auto-critiquing top hypotheses.
    
    Applies heuristic penalties (unchallenged, shallow tree), compares
    top candidates, and returns a final recommendation with confidence.
    
    Args:
        session_id: The reasoning session UUID
        top_n: Number of top candidates to consider (default: 3)
        deep_critique: If True, returns critique requests for LLM
        terminal_output: v17b - Raw terminal output to auto-parse and record outcome
        exhaustive_check: v31b - Run layer-by-layer gap analysis on recommendation
        min_score_threshold: v31d - Minimum score required (default: 0.9)
        min_gap_threshold: v31d - Minimum gap between top candidates (default: 0.08)
        skip_quality_gate: v33 - If True, bypass enforcement (for debugging only)
        skip_sequential_analysis: v37 - If True, skip sequential gap analysis check
        
    Returns:
        Final recommendation with adjusted score and decision quality
    """
    try:
        # v37: Enforce sequential analysis (hard gate with override)
        # Phase 1 Refactor: Use extracted helper
        conn = get_db_connection()
        cur = conn.cursor()
        gate_result = check_sequential_gate(cur, session_id, skip_sequential_analysis)
        if gate_result:
            safe_close_connection(conn)
            return gate_result

        # v36: Apply config defaults - Phase 1 Refactor
        min_score_threshold, min_gap_threshold = apply_config_defaults(
            min_score_threshold, min_gap_threshold, PAS_CONFIG
        )
        
        # Get session info - Phase 1 Refactor
        session = fetch_session(cur, session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Get candidate hypotheses - Phase 1 Refactor
        candidates = fetch_candidates(cur, session_id, top_n)
        if not candidates:
            return {
                "success": False, 
                "error": "No hypotheses found. Run prepare_expansion first."
            }
        
        # Fetch sibling data for penalty calculations - Phase 1 Refactor
        sibling_counts, law_diversity = fetch_sibling_data(cur, session_id)
        
        # Process each candidate - Phase 1 Refactor
        processed = process_candidates(
            cur, candidates, sibling_counts, law_diversity,
            _apply_heuristic_penalties, _build_processed_candidate
        )
        
        # v40: Complementarity Detection - Phase 1 Refactor
        processed, complementarity_result = check_complementarity(
            processed, top_n, detect_complementarity, 
            extract_addressed_goals, synthesize_hypothesis_text, session_id
        )
        
        # Deep critique mode - Phase 1 Refactor
        if deep_critique and len(processed) >= 1:
            critique_result = build_deep_critique_requests(cur, processed, count=2)
            if critique_result:
                return critique_result
        
        # Compare top 2 and apply UCT tiebreaking - Phase 1 Refactor
        recommendation, runner_up, decision_quality, gap_analysis, uct_applied, gap = apply_uct_and_compute_decision(
            processed, _apply_uct_tiebreaking, _compute_decision_quality, session_id
        )
        
        # v20: Compute adaptive depth quality metrics
        quality_result = compute_quality_metrics(
            cur=cur,
            session_id=session_id,
            candidates=processed,
            gap=gap
        )
        
        # Get interview context summary - Phase 1 Refactor
        context_summary = get_context_summary(session)
        
        # Determine next_step guidance - Phase 1 Refactor
        next_step = build_next_step_guidance(decision_quality, recommendation, session_id)
        
        # v8a: Outcome prompt to close self-learning loop
        # v17c: Focus on business value, not code quality (RLVR handles that)
        outcome_prompt = f"Did this solve your business problem? record_outcome(session_id='{session_id}', outcome='success'|'partial'|'failure'). Note: Code quality is validated automatically by RLVR."
        
        # v48: Parallel Critique Window - Phase 1 Refactor
        pending_critiques, explore_alternatives_prompt = identify_pending_critiques(
            PAS_CONFIG,
            processed,
            recommendation["final_score"],
            session.get("context"),
            cur,
            conn,
            session_id
        )
        
        # v8c: Implementation checklist - Phase 1 Refactor: Use extracted helper
        winning_content = recommendation["content"]
        winning_scope = recommendation.get("declared_scope", "")
        implementation_checklist = build_implementation_checklist(winning_scope)
        
        # Phase 9: LSP session tracking import
        from pas.helpers.lsp_enrichment import get_lsp_summary
        
        # =====================================================================
        # Phase 3: LSP Impact Analysis - Gather blast radius for winning scope
        # =====================================================================
        lsp_impact_result = None
        if winning_scope:
            try:
                from pas.helpers.lsp_enrichment import get_lsp_impact_from_scope, get_lsp_summary
                from pas.lsp.lsp_pool import LspPool
                
                # Get project root from session context if available
                project_root = None
                if session.get("context"):
                    ctx = session["context"]
                    if isinstance(ctx, dict):
                        project_root = ctx.get("project_root") or ctx.get("project_path")
                
                # Try to get LSP pool for the project
                lsp_pool = None
                if project_root:
                    try:
                        lsp_pool = await LspPool.get(str(project_root))
                    except Exception as e:
                        logger.debug(f"LSP pool unavailable: {e}")
                
                lsp_impact_result = await get_lsp_impact_from_scope(
                    declared_scope=winning_scope,
                    project_root=project_root,
                    lsp_pool=lsp_pool,
                )
            except Exception as e:
                logger.warning(f"Phase 3: LSP impact gathering failed: {e}")
                lsp_impact_result = {
                    "lsp_available": False,
                    "error": str(e),
                    "scope_parsed": winning_scope,
                }
        

        # =====================================================================
        # v32b: Warning Persistence & v26: Tag Suggestions - Phase 1 Refactor
        # =====================================================================
        warnings_surfaced, suggested_tags, implementation_checklist = surface_warnings_and_tags(
            _search_relevant_failures,
            _generate_suggested_tags,
            session,
            recommendation,
            implementation_checklist,
            cur,
            conn,
            session_id
        )
        
        # v15b: Query past failures - Phase 1 Refactor
        past_failures = query_past_failures(cur, session)
        
        # v16b.1: Confidence calibration warning - Phase 1 Refactor
        calibration_warning = query_calibration_warning(cur)
        
        # v15b: Construct scope_guidance - Phase 1 Refactor
        scope_guidance = build_scope_guidance(session, recommendation, past_failures, calibration_warning)
        
        # v17b: RLVR Auto-Recording - parse terminal output and auto-record outcome
        rlvr_result = None
        if terminal_output and terminal_output.strip():
            try:
                rlvr_result = await parse_terminal_output(
                    session_id=session_id,
                    terminal_text=terminal_output,
                    auto_record=True
                )
                logger.info(f"v17b: RLVR auto-recorded for session {session_id}: {rlvr_result.get('signal')}")
            except Exception as e:
                logger.warning(f"v17b RLVR auto-record failed: {e}")
                rlvr_result = {"success": False, "error": str(e)}
        
        # v31b: Exhaustive Check - Phase 1 Refactor
        exhaustive_gaps = None
        exhaustive_prompt = build_exhaustive_prompt(recommendation, session, exhaustive_check)

        
        # v32: Sequential analysis moved to run_sequential_analysis tool
        # Call run_sequential_analysis(session_id) BEFORE finalize_session for gap analysis
        sequential_analysis: list[dict[str, Any]] = []  # Populated by run_sequential_analysis if called
        systemic_gaps: list[str] = []
        
        # v31d: Quality Gate - Phase 1 Refactor: Use extracted helpers
        winner_score = recommendation["adjusted_score"]
        runner_up_score = runner_up["adjusted_score"] if runner_up else 0.0
        
        quality_gate, quality_gate_enforced = compute_quality_gate(
            winner_score, runner_up_score,
            min_score_threshold, min_gap_threshold,
            skip_quality_gate
        )
        
        # v82: Phase 4 Hardening - Critique gate enforcement
        critique_gate = check_critique_gate(
            cur, recommendation["node_id"], skip_critique_check
        )
        
        # Log bypass if escape hatch used
        if skip_critique_check:
            cur.execute("""
                INSERT INTO tool_calls (session_id, tool_name, call_args)
                VALUES (%s, 'critique_check_bypass', %s)
            """, (session_id, json.dumps({
                "reason": critique_bypass_reason or "No reason provided",
                "bypass_by": "user_request"
            })))
            conn.commit()
        
        # Add critique_gate to quality_gate dict
        quality_gate["critique_gate"] = critique_gate
        
        # Block if critique_gate failed (unless skipped)
        if not critique_gate["passed"] and not skip_critique_check:
            quality_gate["passed"] = False
            if "blockers" not in quality_gate:
                quality_gate["blockers"] = []
            quality_gate["blockers"].append("uncritiqued_top_hypothesis")
            quality_gate_enforced = False
        
        # Phase 8: Mode-aware quality gate behavior
        session_mode = session.get("session_mode", "implementation")
        unconstrained = session.get("unconstrained", False)
        mode_config = MODE_CONFIG.get(session_mode, MODE_CONFIG["implementation"])
        
        # Research mode: softer enforcement (show warnings, don't block)
        if not mode_config["enforce_quality_gate"]:
            quality_gate_enforced = False
            # Add warning if score is below threshold (informational only)
            quality_warning = None
            if not quality_gate["passed"] and not unconstrained:
                quality_warning = f"Score {winner_score:.2f} below threshold (informational - research mode)"
            session["quality_warning"] = quality_warning  # Pass to return dict
        
        # v31e: Score Improvement Suggestions - Phase 1 Refactor: Use extracted helper
        score_improvement_suggestions = build_score_improvement_suggestions(
            quality_gate, winner_score, quality_gate["gap"],
            min_score_threshold, min_gap_threshold
        )
        
        # v33: If gate not passed AND not skipped, prefix [UNVERIFIED] - Phase 1 Refactor
        recommendation_content = apply_unverified_prefix(
            cur, conn, session_id, recommendation["content"],
            quality_gate_enforced, winner_score, quality_gate["gap"]
        )

        
        return {
            "success": True,
            "session_id": session_id,
            "recommendation": {
                "node_id": recommendation["node_id"],
                "content": recommendation_content,  # v33: may have [UNVERIFIED] prefix
                "original_score": recommendation["original_score"],
                "adjusted_score": recommendation["adjusted_score"],
                "penalties_applied": recommendation["penalties_applied"]
            },
            # v33: Quality gate enforcement status
            "quality_gate_enforced": quality_gate_enforced,
            "runner_up": {
                "node_id": runner_up["node_id"],
                "content": runner_up["content"],
                "adjusted_score": runner_up["adjusted_score"]
            } if runner_up else None,
            "decision_quality": decision_quality,
            "gap_analysis": gap_analysis,
            "candidates_evaluated": len(processed),
            "context_summary": context_summary,
            "next_step": next_step,
            "outcome_prompt": outcome_prompt if not rlvr_result else None,  # v17b: suppress if auto-recorded
            "implementation_checklist": implementation_checklist,
            "warnings_surfaced": warnings_surfaced,  # v32b: past failure warnings found
            "scope_guidance": scope_guidance,  # v15b
            "rlvr_result": rlvr_result,  # v17b: auto-outcome detection result
            # v20: Adaptive depth quality metrics
            "quality_sufficient": quality_result["sufficient"],
            "quality_breakdown": quality_result["metrics"],
            "deepen_suggestions": quality_result["suggestions"] if not quality_result["sufficient"] else [],
            # v26: Suggested tags for session organization
            "suggested_tags": suggested_tags,
            # v31b/v32: Exhaustive gap check - prompt for agent to process if needed
            "exhaustive_gaps": exhaustive_gaps,
            "exhaustive_prompt": exhaustive_prompt,  # v32: LLM prompt for agent to process
            # v31d: Quality gate (score + gap thresholds)
            "quality_gate": quality_gate,
            # v31e: How to improve if gate not passed
            "score_improvement_suggestions": score_improvement_suggestions,
            # v32: Always-on sequential analysis
            "sequential_analysis": sequential_analysis,
            "systemic_gaps": systemic_gaps,
            # v40: Complementarity detection
            "complementarity": complementarity_result,
            # v48: Parallel critique window
            "pending_critiques": pending_critiques if pending_critiques else None,
            "explore_alternatives_prompt": explore_alternatives_prompt,
            # v82: Dual recommendation - compute first for use in plan_template_prompt
            "dual_recommendation": (dual_rec := compute_dual_recommendation(processed, recommendation)),
            # v49/v69: Plan template prompt (when gate passes)
            # Detects multi-phase work by keywords in goal, adds dual-plan section if aspirational differs
            "plan_template_prompt": _build_plan_template_prompt(
                quality_gate, session.get("goal", ""), dual_rec
            ) if quality_gate["passed"] else None,
            # v51: Effort-Benefit ROI analysis
            "roi_analysis": build_roi_analysis(processed, recommendation),
            # Phase 3: LSP blast radius analysis for implementation plan
            "lsp_impact": lsp_impact_result,
            # Phase 9: LSP session tracking summary
            "lsp_summary": get_lsp_summary(session_id, conn) if session_id else None,
            # Phase 8: Mode-aware session info
            "session_mode": session_mode,
            "mode_label": mode_config["finalize_label"],  # "recommendation" or "synthesis"
            "quality_warning": session.get("quality_warning"),  # Informational for research mode
        }



        
    except Exception as e:
        logger.error(f"finalize_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v40: Hybrid Synthesis Tools (Complementarity Detection)
# =============================================================================


@mcp.tool()
async def synthesize_hypotheses(
    session_id: str,
    node_ids: list[str]
) -> dict[str, Any]:
    """
    Combine complementary hypotheses into a unified hybrid.
    
    Use this when finalize_session returns complementarity_detected=True,
    indicating that top candidates address different goals and should be
    combined rather than selected winner-takes-all.
    
    Args:
        session_id: The reasoning session UUID
        node_ids: List of hypothesis node IDs to synthesize (2-5 nodes)
        
    Returns:
        Created hybrid node with synthesis lineage
    """
    if len(node_ids) < 2:
        return {"success": False, "error": "Need at least 2 nodes to synthesize"}
    if len(node_ids) > 5:
        return {"success": False, "error": "Maximum 5 nodes for synthesis"}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Validate session
        cur.execute(
            "SELECT id, goal FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Fetch source nodes
        cur.execute(
            """
            SELECT id, path, content, prior_score, likelihood, posterior_score,
                   supporting_laws, metadata
            FROM thought_nodes
            WHERE id = ANY(%s::uuid[]) AND session_id = %s
            """,
            (node_ids, session_id)
        )
        source_nodes = cur.fetchall()
        
        if len(source_nodes) != len(node_ids):
            found_ids = [str(n["id"]) for n in source_nodes]
            missing = [nid for nid in node_ids if nid not in found_ids]
            return {"success": False, "error": f"Nodes not found: {missing}"}
        
        # Build candidate dicts for synthesis
        candidates = []
        all_supporting_laws = set()
        for node in source_nodes:
            # Get declared scope from metadata if present
            metadata = node.get("metadata") or {}
            scope = metadata.get("declared_scope", "")
            
            candidates.append({
                "content": node["content"],
                "scope": scope,
                "node_id": str(node["id"]),
                "score": float(node["posterior_score"] or 0.5)
            })
            
            if node["supporting_laws"]:
                all_supporting_laws.update(node["supporting_laws"])
        
        # Generate hybrid content
        hybrid_content = synthesize_hypothesis_text(candidates, include_details=True)
        merged_scope = merge_scopes(candidates)
        
        # Calculate hybrid prior (average of source posteriors)
        avg_posterior = sum(c["score"] for c in candidates) / len(candidates)
        
        # Determine hybrid path (sibling of first source node)
        first_path = str(source_nodes[0]["path"])
        if "." in first_path:
            parent_path = ".".join(first_path.rsplit(".", 1)[:-1])
        else:
            parent_path = "root"
        hybrid_path = f"{parent_path}.hybrid_1"
        
        # Generate embedding for hybrid
        embedding = get_embedding(hybrid_content)
        
        # Store hybrid node
        cur.execute(
            """
            INSERT INTO thought_nodes (
                session_id, path, content, node_type,
                prior_score, likelihood, embedding,
                supporting_laws, synthesized_from, metadata
            ) VALUES (%s, %s, %s, 'hypothesis', %s, %s, %s, %s, %s::uuid[], %s)
            RETURNING id, path, posterior_score
            """,
            (
                session_id,
                hybrid_path,
                hybrid_content,
                avg_posterior,  # Prior is average of source posteriors
                0.85,  # Initial likelihood (starts high, can be critiqued)
                embedding,
                list(all_supporting_laws) if all_supporting_laws else [],
                node_ids,  # synthesized_from tracks lineage
                json.dumps({
                    "declared_scope": merged_scope,
                    "is_hybrid": True,
                    "source_count": len(node_ids),
                    "addressed_goals": list(set(
                        g for c in candidates 
                        for g in extract_addressed_goals(c["content"], c.get("scope", ""))
                    ))
                })
            )
        )
        hybrid = cur.fetchone()
        conn.commit()
        
        logger.info(f"v40: Created hybrid hypothesis {hybrid['id']} from {len(node_ids)} sources")
        
        return {
            "success": True,
            "hybrid_node": {
                "node_id": str(hybrid["id"]),
                "path": str(hybrid["path"]),
                "content": hybrid_content,
                "posterior_score": float(hybrid["posterior_score"]),
                "synthesized_from": node_ids,
                "declared_scope": merged_scope
            },
            "source_nodes": [
                {"node_id": c["node_id"], "score": c["score"]} 
                for c in candidates
            ],
            "next_step": f"Critique the hybrid hypothesis. Call prepare_critique(node_id='{hybrid['id']}')"
        }
        
    except Exception as e:
        logger.error(f"synthesize_hypotheses failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def infer_file_purpose(
    project_id: str,
    file_path: str,
    force_refresh: bool = False
) -> dict[str, Any]:
    """
    Get or infer 3-tier purpose for a file.
    
    Returns cached purpose if available, or a prompt for LLM inference.
    After LLM processes the prompt, call store_file_purpose with the result.
    
    Args:
        project_id: Project identifier
        file_path: Path to the file (relative or absolute)
        force_refresh: If True, ignore cache and return inference prompt
        
    Returns:
        Cached purpose or inference prompt
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Normalize file path
        if not file_path.startswith('/'):
            # Try to find in file_registry
            cur.execute(
                "SELECT file_path FROM file_registry WHERE project_id = %s AND file_path LIKE %s LIMIT 1",
                (project_id, f'%{file_path}%')
            )
            result = cur.fetchone()
            if result:
                file_path = result["file_path"]
        
        # Get file from registry
        cur.execute(
            """
            SELECT file_path, file_hash, content, purpose_cache
            FROM file_registry
            WHERE project_id = %s AND file_path = %s
            """,
            (project_id, file_path)
        )
        file_record = cur.fetchone()
        
        if not file_record:
            return {"success": False, "error": f"File not found in registry: {file_path}. Run sync_project first."}
        
        current_hash = file_record["file_hash"]
        cached = file_record["purpose_cache"]
        
        # Check cache validity
        if not force_refresh and validate_purpose_cache(cached, current_hash):
            purposes = cached.get("purposes", {})
            return {
                "success": True,
                "status": "cached",
                "file_path": file_path,
                "file_hash": current_hash,
                "purposes": purposes,
                "purpose_context": build_purpose_context_entry(file_path, purposes, "cached")
            }
        
        # Cache miss or force refresh - return prompt for LLM
        content = file_record.get("content") or ""
        
        # v45a: Fetch indexed symbols for function-level analysis
        symbols = []
        try:
            cur.execute("""
                SELECT fs.symbol_name, fs.symbol_type, fs.line_start, fs.signature
                FROM file_symbols fs
                JOIN file_registry fr ON fs.file_id = fr.id
                WHERE fr.project_id = %s AND fr.file_path = %s
                ORDER BY fs.line_start
                LIMIT 30
            """, (project_id, file_path))
            symbols = [dict(r) for r in cur.fetchall()]
            logger.info(f"v45a: Found {len(symbols)} indexed symbols for {file_path}")
        except Exception as e:
            logger.debug(f"v45a: Symbol fetch failed (non-fatal): {e}")
        
        prompt = build_purpose_prompt(file_path, content, symbols=symbols)
        
        return {
            "success": True,
            "status": "needs_inference",
            "file_path": file_path,
            "file_hash": current_hash,
            "inference_prompt": prompt,
            "symbol_count": len(symbols),  # v45a: Report symbol count
            "instructions": f"Process this prompt with LLM, then call store_file_purpose(project_id='{project_id}', file_path='{file_path}', purpose_data='<JSON result>')"
        }
        
    except Exception as e:
        logger.error(f"infer_file_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_file_purpose(
    project_id: str,
    file_path: str,
    purpose_data: str  # JSON string for universal LLM compatibility
) -> dict[str, Any]:
    """
    Store LLM-inferred purpose in the file registry cache.
    
    Args:
        project_id: Project identifier
        file_path: Path to the file
        purpose_data: JSON string with function_purposes, module_purpose, project_contribution
        
    Returns:
        Confirmation with stored purpose summary
    """
    try:
        # Parse purpose data
        purposes = parse_purpose_response(purpose_data)
        if purposes is None:
            return {"success": False, "error": "Failed to parse purpose_data. Ensure valid JSON with function_purposes, module_purpose, project_contribution."}
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get current file hash
        cur.execute(
            "SELECT file_hash FROM file_registry WHERE project_id = %s AND file_path = %s",
            (project_id, file_path)
        )
        file_record = cur.fetchone()
        
        if not file_record:
            return {"success": False, "error": f"File not found: {file_path}"}
        
        file_hash = file_record["file_hash"]
        
        # Build cache entry
        cache_entry = {
            "file_hash": file_hash,
            "purposes": purposes,
            "inferred_at": datetime.now().isoformat()
        }
        
        # Store in purpose_cache
        cur.execute(
            """
            UPDATE file_registry
            SET purpose_cache = %s
            WHERE project_id = %s AND file_path = %s
            """,
            (json.dumps(cache_entry), project_id, file_path)
        )
        conn.commit()
        
        logger.info(f"v40: Stored purpose for {file_path} in project {project_id}")
        
        module_purpose = purposes.get("module_purpose", {})
        
        return {
            "success": True,
            "file_path": file_path,
            "file_hash": file_hash,
            "purpose_summary": {
                "problem": module_purpose.get("problem", "")[:200],
                "user_need": module_purpose.get("user_need", "")[:200],
                "function_count": len(purposes.get("function_purposes", []))
            },
            "purpose_context": build_purpose_context_entry(file_path, purposes, "cached")
        }
        
    except Exception as e:
        logger.error(f"store_file_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def infer_module_purpose(
    project_id: str,
    directory_path: str,
    max_files: int = 10
) -> dict[str, Any]:
    """
    Aggregate file purposes into module-level purpose.
    
    Scans files in directory, uses cached file purposes, and returns
    an aggregation prompt for LLM synthesis.
    
    Args:
        project_id: Project identifier
        directory_path: Path to directory/module
        max_files: Maximum files to include (default 10)
        
    Returns:
        Aggregation prompt or cached module purpose
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Find files in this directory
        cur.execute(
            """
            SELECT file_path, purpose_cache
            FROM file_registry
            WHERE project_id = %s 
              AND file_path LIKE %s
              AND file_path NOT LIKE %s
            ORDER BY file_path
            LIMIT %s
            """,
            (project_id, f'{directory_path}%', f'{directory_path}%/%', max_files)
        )
        files = cur.fetchall()
        
        if not files:
            return {
                "success": False, 
                "error": f"No files found in {directory_path}. Run sync_project first."
            }
        
        # Collect file purposes
        file_purposes = []
        needs_inference = []
        
        for f in files:
            file_path = f["file_path"]
            cache = f["purpose_cache"]
            
            if cache and cache.get("purposes"):
                purposes = cache["purposes"]
                module_purpose = purposes.get("module_purpose", {})
                file_purposes.append({
                    "file": file_path,
                    "problem": module_purpose.get("problem", ""),
                    "user_need": module_purpose.get("user_need", ""),
                    "contribution": purposes.get("project_contribution", "")  # v45b
                })
            else:
                needs_inference.append(file_path)
        
        # If any files need inference, return that info first
        if needs_inference and len(needs_inference) > len(file_purposes):
            return {
                "success": True,
                "status": "files_need_inference",
                "directory_path": directory_path,
                "files_ready": len(file_purposes),
                "files_pending": needs_inference[:5],  # Show first 5
                "instructions": f"Call infer_file_purpose for pending files, then retry infer_module_purpose"
            }
        
        # v45b: Fetch symbols for files in this module
        file_paths = [f["file_path"] for f in files]
        cur.execute("""
            SELECT fs.symbol_name, fs.symbol_type
            FROM file_symbols fs
            JOIN file_registry fr ON fs.file_id = fr.id
            WHERE fr.project_id = %s 
              AND fr.file_path = ANY(%s)
              AND fs.symbol_type IN ('function', 'class')
            ORDER BY fs.line_start
            LIMIT 20
        """, (project_id, file_paths))
        symbols = [{"name": r["symbol_name"], "type": r["symbol_type"]} for r in cur.fetchall()]
        
        # Build aggregation prompt with symbols (v45b)
        prompt = build_module_aggregation_prompt(directory_path, file_purposes, symbols)
        
        return {
            "success": True,
            "status": "ready_for_aggregation",
            "directory_path": directory_path,
            "file_count": len(file_purposes),
            "aggregation_prompt": prompt,
            "instructions": f"Process this prompt with LLM, then call store_module_purpose(project_id='{project_id}', directory_path='{directory_path}', purpose_data='<JSON result>')"
        }
        
    except Exception as e:
        logger.error(f"infer_module_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_module_purpose(
    project_id: str,
    directory_path: str,
    purpose_data: str  # JSON string
) -> dict[str, Any]:
    """
    Store aggregated module-level purpose.
    
    Args:
        project_id: Project identifier
        directory_path: Path to the module directory
        purpose_data: JSON with module_purpose, user_need, architecture_role
        
    Returns:
        Confirmation with stored purpose
    """
    try:
        # Parse purpose data
        purposes = parse_module_response(purpose_data)
        if purposes is None:
            return {
                "success": False, 
                "error": "Failed to parse purpose_data. Ensure valid JSON with module_purpose, user_need, architecture_role."
            }
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Store as a special entry in file_registry with directory path
        # Use a hash of the directory path as file_hash
        import hashlib
        dir_hash = hashlib.sha256(directory_path.encode()).hexdigest()[:16]
        
        cache_entry = {
            "file_hash": dir_hash,
            "scope_type": "module",
            "purposes": purposes,
            "inferred_at": datetime.now().isoformat()
        }
        
        # Check if entry exists
        cur.execute(
            "SELECT id FROM file_registry WHERE project_id = %s AND file_path = %s",
            (project_id, directory_path)
        )
        existing = cur.fetchone()
        
        if existing:
            cur.execute(
                "UPDATE file_registry SET purpose_cache = %s WHERE project_id = %s AND file_path = %s",
                (json.dumps(cache_entry), project_id, directory_path)
            )
        else:
            cur.execute(
                """
                INSERT INTO file_registry (project_id, file_path, file_hash, purpose_cache)
                VALUES (%s, %s, %s, %s)
                """,
                (project_id, directory_path, dir_hash, json.dumps(cache_entry))
            )
        
        conn.commit()
        logger.info(f"v40: Stored module purpose for {directory_path} in project {project_id}")
        
        return {
            "success": True,
            "directory_path": directory_path,
            "scope_type": "module",
            "purpose_summary": {
                "module_purpose": purposes.get("module_purpose", "")[:200],
                "architecture_role": purposes.get("architecture_role", "")[:200]
            }
        }
        
    except Exception as e:
        logger.error(f"store_module_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v43: Project Purpose Awareness Tools
# =============================================================================

@mcp.tool()
async def infer_project_purpose(
    project_id: str,
    force_refresh: bool = False
) -> dict[str, Any]:
    """
    Get or infer project-level purpose (mission, user needs, required modules).
    
    Returns cached purpose if available, or a prompt for LLM inference.
    After LLM processes the prompt, call store_project_purpose with the result.
    
    Args:
        project_id: Project identifier
        force_refresh: If True, ignore cache and return inference prompt
        
    Returns:
        Cached purpose or inference prompt
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if project exists in project_registry
        cur.execute(
            """
            SELECT project_path, purpose_hierarchy, detected_domain, domain_confidence, updated_at
            FROM project_registry
            WHERE project_id = %s
            """,
            (project_id,)
        )
        project = cur.fetchone()
        
        if not project:
            return {
                "success": False, 
                "error": f"Project not found: {project_id}. Run sync_project first."
            }
        
        # Check cache validity
        if not force_refresh and project["purpose_hierarchy"]:
            # v43 Phase 3: Staleness detection
            stale = False
            stale_message = None
            age_days = 0
            
            try:
                import yaml
                from datetime import datetime, timezone
                
                with open(Path(__file__).parent / "config" / "config.yaml") as f:
                    cfg = yaml.safe_load(f)
                staleness_days = cfg.get("purpose_inference", {}).get("staleness_days", 7)
                
                if project["updated_at"]:
                    updated_at = project["updated_at"]
                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)
                    
                    age_days = (datetime.now(timezone.utc) - updated_at).days
                    stale = age_days > staleness_days
                    if stale:
                        stale_message = f"Purpose is {age_days} days old (threshold: {staleness_days}). Consider calling with force_refresh=True."
            except Exception as e:
                logger.warning(f"Staleness check failed: {e}")
            
            return {
                "success": True,
                "status": "cached",
                "project_id": project_id,
                "project_path": project["project_path"],
                "purpose_hierarchy": project["purpose_hierarchy"],
                "detected_domain": project["detected_domain"],
                "domain_confidence": float(project["domain_confidence"]) if project["domain_confidence"] else None,
                "cached_at": project["updated_at"].isoformat() if project["updated_at"] else None,
                "stale": stale,
                "stale_message": stale_message
            }
        
        # Gather context for inference prompt
        # Get file count and languages
        cur.execute(
            """
            SELECT COUNT(*) as file_count, 
                   ARRAY_AGG(DISTINCT language) FILTER (WHERE language IS NOT NULL) as languages
            FROM file_registry
            WHERE project_id = %s
            """,
            (project_id,)
        )
        stats = cur.fetchone()
        file_count = stats["file_count"] or 0
        languages = stats["languages"] or []
        
        if file_count == 0:
            return {
                "success": False,
                "error": f"No files indexed for project {project_id}. Run sync_project first."
            }
        
        # v45c: First try to get aggregated module-level purposes
        cur.execute(
            """
            SELECT file_path, purpose_cache
            FROM file_registry
            WHERE project_id = %s 
              AND purpose_cache IS NOT NULL
              AND purpose_cache->>'scope_type' = 'module'
            ORDER BY file_path
            LIMIT 10
            """,
            (project_id,)
        )
        module_entries = cur.fetchall()
        
        module_summaries = []
        if module_entries:
            # Use aggregated module purposes (v45b)
            for m in module_entries:
                cache = m["purpose_cache"]
                if cache and "purposes" in cache:
                    purposes = cache["purposes"]
                    module_summaries.append({
                        "path": m["file_path"],
                        "purpose": purposes.get("module_purpose", "Unknown"),
                        "architecture_role": purposes.get("architecture_role", "")
                    })
        else:
            # Fallback: use individual file purposes (backward compatible)
            cur.execute(
                """
                SELECT file_path, purpose_cache
                FROM file_registry
                WHERE project_id = %s AND purpose_cache IS NOT NULL
                ORDER BY line_count DESC
                LIMIT 10
                """,
                (project_id,)
            )
            files = cur.fetchall()
            for f in files:
                cache = f["purpose_cache"]
                if cache and "purposes" in cache:
                    module_purpose = cache["purposes"].get("module_purpose", {})
                    module_summaries.append({
                        "path": f["file_path"],
                        "purpose": module_purpose.get("problem", "Unknown")
                    })
        
        # Build prompt using helper
        from pas.helpers.purpose import build_project_purpose_prompt
        prompt = build_project_purpose_prompt(
            project_id=project_id,
            project_path=project["project_path"],
            module_summaries=module_summaries,
            file_count=file_count,
            languages=languages
        )
        
        return {
            "success": True,
            "status": "needs_inference",
            "project_id": project_id,
            "project_path": project["project_path"],
            "file_count": file_count,
            "files_with_purpose": len(module_summaries),
            "inference_prompt": prompt,
            "instructions": f"Process this prompt with LLM, then call store_project_purpose(project_id='{project_id}', purpose_data='<JSON result>')"
        }
        
    except Exception as e:
        logger.error(f"infer_project_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_project_purpose(
    project_id: str,
    purpose_data: str  # JSON string for universal LLM compatibility
) -> dict[str, Any]:
    """
    Store LLM-inferred project purpose in the project registry.
    
    Args:
        project_id: Project identifier
        purpose_data: JSON string with mission, user_needs, must_have_modules, detected_domain, domain_confidence
        
    Returns:
        Confirmation with stored purpose summary
    """
    try:
        # Parse purpose data
        from pas.helpers.purpose import parse_project_purpose_response
        parsed = parse_project_purpose_response(purpose_data)
        if parsed is None:
            return {
                "success": False, 
                "error": "Failed to parse purpose_data. Ensure valid JSON with mission, user_needs, must_have_modules, detected_domain, domain_confidence."
            }
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify project exists
        cur.execute(
            "SELECT project_path FROM project_registry WHERE project_id = %s",
            (project_id,)
        )
        project = cur.fetchone()
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}
        
        # Generate embedding from mission text for semantic search
        mission = parsed["purpose_hierarchy"].get("mission", "")
        embedding = get_embedding(mission[:2000]) if mission else None
        
        # Update project registry
        cur.execute(
            """
            UPDATE project_registry
            SET purpose_hierarchy = %s,
                detected_domain = %s,
                domain_confidence = %s,
                purpose_embedding = %s,
                updated_at = NOW()
            WHERE project_id = %s
            """,
            (
                json.dumps(parsed["purpose_hierarchy"]),
                parsed["detected_domain"],
                parsed["domain_confidence"],
                embedding,
                project_id
            )
        )
        conn.commit()
        
        logger.info(f"v43: Stored project purpose for {project_id}")
        
        purpose_hierarchy = parsed["purpose_hierarchy"]
        return {
            "success": True,
            "project_id": project_id,
            "purpose_summary": {
                "mission": purpose_hierarchy.get("mission", "")[:200],
                "user_needs_count": len(purpose_hierarchy.get("user_needs", [])),
                "must_have_modules_count": len(purpose_hierarchy.get("must_have_modules", []))
            },
            "detected_domain": parsed["detected_domain"],
            "domain_confidence": parsed["domain_confidence"]
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_project_purpose failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def analyze_completeness(
    project_id: str
) -> dict[str, Any]:
    """
    Compare project's must_have_modules against actual indexed modules.
    
    Uses semantic similarity to match required modules from purpose_hierarchy
    against files in file_registry. Returns implemented vs missing modules.
    
    Args:
        project_id: Project identifier
        
    Returns:
        Completeness analysis with implemented/missing modules
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get project purpose
        cur.execute(
            """
            SELECT purpose_hierarchy, detected_domain
            FROM project_registry
            WHERE project_id = %s
            """,
            (project_id,)
        )
        project = cur.fetchone()
        
        if not project:
            return {
                "success": False,
                "error": f"Project not found: {project_id}. Run sync_project first."
            }
        
        purpose_hierarchy = project["purpose_hierarchy"]
        if not purpose_hierarchy:
            return {
                "success": False,
                "error": f"Project purpose not inferred yet. Run infer_project_purpose first."
            }
        
        must_have_modules = purpose_hierarchy.get("must_have_modules", [])
        if not must_have_modules:
            return {
                "success": True,
                "project_id": project_id,
                "completeness": 1.0,
                "message": "No must_have_modules defined in purpose hierarchy",
                "implemented": [],
                "missing": []
            }
        
        # Get similarity threshold from config
        threshold = 0.7  # Default
        try:
            import yaml
            with open(Path(__file__).parent / "config" / "config.yaml", "r") as f:
                config = yaml.safe_load(f)
                threshold = config.get("purpose_inference", {}).get(
                    "completeness_similarity_threshold", 0.7
                )
        except Exception:
            pass
        
        # Get all file embeddings for this project
        cur.execute(
            """
            SELECT file_path, content_embedding, purpose_cache
            FROM file_registry
            WHERE project_id = %s AND content_embedding IS NOT NULL
            """,
            (project_id,)
        )
        files = cur.fetchall()
        
        if not files:
            return {
                "success": False,
                "error": f"No files indexed for project {project_id}."
            }
        
        # Check each required module against file embeddings
        implemented = []
        missing = []
        
        for required_module in must_have_modules:
            # Generate embedding for required module name
            module_embedding = get_embedding(required_module)
            # Convert to list for comparison (v43 bugfix: numpy array truth value)
            if hasattr(module_embedding, 'tolist'):
                module_embedding = module_embedding.tolist()
            
            # Find best matching file
            best_match = None
            best_similarity = 0.0
            
            for f in files:
                file_embedding = f["content_embedding"]
                # Use explicit None check - numpy arrays fail on truthy check
                if file_embedding is not None:
                    # Convert file embedding to list if needed
                    if hasattr(file_embedding, 'tolist'):
                        file_embedding = file_embedding.tolist()
                    elif not isinstance(file_embedding, list):
                        file_embedding = list(file_embedding)
                    # Calculate cosine similarity (Euclidean distance approximation)
                    try:
                        similarity = 1 - (
                            sum((a - b) ** 2 for a, b in zip(module_embedding, file_embedding)) ** 0.5 / 2
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = f["file_path"]
                    except Exception as e:
                        logger.warning(f"Similarity calc failed: {e}")
                        continue
            
            if best_similarity >= threshold:
                implemented.append({
                    "required": required_module,
                    "matched_file": best_match,
                    "similarity": round(best_similarity, 3)
                })
            else:
                missing.append({
                    "required": required_module,
                    "best_candidate": best_match,
                    "similarity": round(best_similarity, 3) if best_match else 0.0
                })
        
        completeness = len(implemented) / len(must_have_modules) if must_have_modules else 1.0
        
        return {
            "success": True,
            "project_id": project_id,
            "detected_domain": project["detected_domain"],
            "completeness": round(completeness, 3),
            "threshold": threshold,
            "implemented": implemented,
            "missing": missing,
            "total_required": len(must_have_modules),
            "total_implemented": len(implemented)
        }
        
    except Exception as e:
        logger.error(f"analyze_completeness failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_purpose_chain(
    project_id: str,
    file_path: str
) -> dict[str, Any]:
    """
    Trace the purpose hierarchy from file to module to project.
    
    Returns the teleological context for a file, showing how it contributes
    to higher-level goals. Useful for hypothesis grounding.
    
    Args:
        project_id: Project identifier
        file_path: File path (relative or absolute)
        
    Returns:
        Purpose chain from file through module to project
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Normalize file path
        if file_path.startswith('/'):
            # Try to make relative
            cur.execute(
                "SELECT project_path FROM project_registry WHERE project_id = %s",
                (project_id,)
            )
            project = cur.fetchone()
            if project and file_path.startswith(project["project_path"]):
                file_path = file_path[len(project["project_path"]):].lstrip('/')
        
        # Get file purpose
        cur.execute(
            """
            SELECT file_path, purpose_cache, language
            FROM file_registry
            WHERE project_id = %s AND file_path = %s
            """,
            (project_id, file_path)
        )
        file_record = cur.fetchone()
        
        file_purpose = None
        if file_record and file_record["purpose_cache"]:
            cache = file_record["purpose_cache"]
            if "purposes" in cache:
                module_purpose = cache["purposes"].get("module_purpose", {})
                file_purpose = {
                    "file_path": file_record["file_path"],
                    "language": file_record["language"],
                    "problem": module_purpose.get("problem", ""),
                    "user_need": module_purpose.get("user_need", ""),
                    "project_contribution": cache["purposes"].get("project_contribution", "")
                }
        
        # Get module purpose (directory level)
        if file_record:
            import os
            directory = os.path.dirname(file_record["file_path"])
            cur.execute(
                """
                SELECT purpose_cache
                FROM file_registry
                WHERE project_id = %s AND file_path = %s AND purpose_cache->>'scope_type' = 'module'
                """,
                (project_id, directory)
            )
            module_record = cur.fetchone()
            
            module_purpose = None
            if module_record and module_record["purpose_cache"]:
                cache = module_record["purpose_cache"]
                if "purposes" in cache:
                    module_purpose = {
                        "directory": directory,
                        "module_purpose": cache["purposes"].get("module_purpose", ""),
                        "architecture_role": cache["purposes"].get("architecture_role", "")
                    }
        else:
            module_purpose = None
        
        # Get project purpose
        cur.execute(
            """
            SELECT purpose_hierarchy, detected_domain, domain_confidence
            FROM project_registry
            WHERE project_id = %s
            """,
            (project_id,)
        )
        project_record = cur.fetchone()
        
        project_purpose = None
        if project_record and project_record["purpose_hierarchy"]:
            project_purpose = {
                "mission": project_record["purpose_hierarchy"].get("mission", ""),
                "user_needs": project_record["purpose_hierarchy"].get("user_needs", []),
                "detected_domain": project_record["detected_domain"],
                "domain_confidence": float(project_record["domain_confidence"]) if project_record["domain_confidence"] else None
            }
        
        return {
            "success": True,
            "project_id": project_id,
            "file_path": file_path,
            "purpose_chain": {
                "file": file_purpose,
                "module": module_purpose,
                "project": project_purpose
            },
            "grounding_context": _build_grounding_context(file_purpose, module_purpose, project_purpose)
        }
        
    except Exception as e:
        logger.error(f"get_purpose_chain failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_system_map(
    project_id: str,
    include_weights: bool = True
) -> dict[str, Any]:
    """
    Build module dependency graph from symbol references.
    
    Aggregates cross-file references by directory to show which modules
    depend on which others. Part of v45d System Mapping.
    
    Args:
        project_id: Project identifier
        include_weights: If True, include call counts as edge weights (default: True)
        
    Returns:
        MODULE_RELATIONSHIP_GRAPH with nodes, edges, and stats
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify project exists
        cur.execute(
            "SELECT project_id FROM project_registry WHERE project_id = %s",
            (project_id,)
        )
        if not cur.fetchone():
            return {"success": False, "error": f"Project not found: {project_id}"}
        
        # Aggregate symbol_references by module directory
        # CTE extracts directories, then filter/group at module level
        cur.execute(
            """
            WITH module_deps AS (
                SELECT 
                    COALESCE(NULLIF(regexp_replace(source_file, '/[^/]+$', ''), ''), '.') as source_module,
                    COALESCE(NULLIF(regexp_replace(target_file, '/[^/]+$', ''), ''), '.') as target_module
                FROM symbol_references
                WHERE project_id = %s
            )
            SELECT source_module, target_module, COUNT(*) as weight
            FROM module_deps
            WHERE source_module != target_module
            GROUP BY source_module, target_module
            ORDER BY weight DESC
            LIMIT 500
            """,
            (project_id,)
        )
        rows = cur.fetchall()
        
        if not rows:
            return {
                "success": True,
                "project_id": project_id,
                "nodes": [],
                "edges": [],
                "stats": {"module_count": 0, "edge_count": 0},
                "note": "No cross-module dependencies found. Ensure project is synced and has symbol_references."
            }
        
        # Build unique nodes set and edges list
        nodes_set = set()
        edges = []
        for row in rows:
            nodes_set.add(row["source_module"])
            nodes_set.add(row["target_module"])
            edge = {
                "source": row["source_module"],
                "target": row["target_module"]
            }
            if include_weights:
                edge["weight"] = row["weight"]
            edges.append(edge)
        
        nodes = sorted(list(nodes_set))
        
        logger.info(f"v45d: System map for {project_id}: {len(nodes)} modules, {len(edges)} edges")
        
        return {
            "success": True,
            "project_id": project_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "module_count": len(nodes),
                "edge_count": len(edges)
            }
        }
        
    except Exception as e:
        logger.error(f"get_system_map failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def infer_schema_intent(
    project_id: str
) -> dict[str, Any]:
    """
    Extract domain entities from database schema using information_schema.
    
    Uses heuristic classification to identify entities, configurations,
    audit tables, and junction tables. Returns an enrichment_prompt for
    optional LLM semantic analysis.
    
    Part of v45e Schema→Intent.
    
    Args:
        project_id: Project identifier
        
    Returns:
        Extracted entities with enrichment_prompt for LLM refinement
    """
    try:
        conn = get_db_connection()
        
        # Reuse existing schema introspection
        from pas.helpers.self_awareness import get_schema_info
        schema_info = get_schema_info(conn)
        
        if "error" in schema_info:
            return {"success": False, "error": schema_info["error"]}
        
        # Extract entities using heuristics
        from pas.helpers.schema_intent import extract_schema_entities, build_enrichment_prompt
        extraction = extract_schema_entities(
            tables=schema_info.get("tables", {}),
            relationships=schema_info.get("relationships", [])
        )
        
        # Build optional enrichment prompt
        enrichment_prompt = build_enrichment_prompt(
            entities=extraction.get("entities", []),
            relationships=extraction.get("relationships", [])
        )
        
        logger.info(f"v45e: Extracted {len(extraction['entities'])} entities for {project_id}")
        
        return {
            "success": True,
            "project_id": project_id,
            **extraction,
            "enrichment_prompt": enrichment_prompt,
            "instructions": f"Optionally process enrichment_prompt with LLM, then call store_schema_intent(project_id='{project_id}', intent_json='<result>')"
        }
        
    except Exception as e:
        logger.error(f"infer_schema_intent failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def store_schema_intent(
    project_id: str,
    intent_json: str
) -> dict[str, Any]:
    """
    Store schema intent (enriched entities) in project_registry.
    
    Args:
        project_id: Project identifier
        intent_json: JSON string with entities and enrichment data
        
    Returns:
        Confirmation with stored entity count
    """
    try:
        intent_data = json.loads(intent_json)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify project exists
        cur.execute(
            "SELECT project_id FROM project_registry WHERE project_id = %s",
            (project_id,)
        )
        if not cur.fetchone():
            return {"success": False, "error": f"Project not found: {project_id}"}
        
        # Store detected entities
        cur.execute(
            """
            UPDATE project_registry 
            SET detected_entities = %s, updated_at = NOW()
            WHERE project_id = %s
            """,
            (json.dumps(intent_data), project_id)
        )
        conn.commit()
        
        entity_count = len(intent_data.get("entities", intent_data.get("enriched_entities", [])))
        logger.info(f"v45e: Stored {entity_count} entities for {project_id}")
        
        return {
            "success": True,
            "project_id": project_id,
            "entities_stored": entity_count,
            "message": f"Schema intent stored with {entity_count} entities"
        }
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_schema_intent failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def infer_config_assumptions(
    project_id: str,
    config_path: str = "config.yaml"
) -> dict[str, Any]:
    """
    Extract implicit assumptions from configuration files.
    
    Parses YAML/JSON config and identifies:
    - Threshold values (performance expectations)
    - Time-based values (scheduling requirements)
    - Path/URL values (deployment constraints)
    
    Part of v45f Config→Assumptions.
    
    Args:
        project_id: Project identifier
        config_path: Path to config file (relative to project or absolute)
        
    Returns:
        Extracted assumptions with enrichment_prompt for LLM refinement
    """
    try:
        from pas.helpers.config_assumptions import parse_config_file, extract_assumptions, build_enrichment_prompt
        from pathlib import Path
        
        # v61: Resolve config path relative to project root
        resolved_path = None
        project_root = None
        
        if not Path(config_path).is_absolute():
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT project_root FROM project_registry WHERE project_id = %s", (project_id,))
            row = cur.fetchone()
            conn.close()
            
            if row and row.get('project_root'):
                project_root = Path(row['project_root'])
                # Search common config locations
                search_dirs = [
                    project_root,
                    project_root / 'config',
                    project_root / 'src' / 'config',
                ]
                # Also check src/<pkg>/config/ pattern
                src_dir = project_root / 'src'
                if src_dir.exists():
                    for child in src_dir.iterdir():
                        if child.is_dir() and not child.name.startswith('.'):
                            config_subdir = child / 'config'
                            if config_subdir.exists():
                                search_dirs.append(config_subdir)
                
                # Try to find the config file
                for search_dir in search_dirs:
                    candidate = search_dir / config_path
                    if candidate.exists():
                        resolved_path = candidate
                        break
        
        # Use resolved path or original
        final_path = str(resolved_path) if resolved_path else config_path
        
        # Parse config
        config = parse_config_file(final_path)
        
        # Extract assumptions using heuristics
        assumptions = extract_assumptions(config)
        
        # Build enrichment prompt
        enrichment_prompt = build_enrichment_prompt(assumptions, final_path)
        
        # Categorize assumptions
        by_type: dict[str, list] = {}
        for a in assumptions:
            t = a.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(a)
        
        logger.info(f"v45f: Extracted {len(assumptions)} assumptions from {final_path}")
        
        return {
            "success": True,
            "project_id": project_id,
            "config_path": final_path,
            "assumptions": assumptions,
            "by_type": by_type,
            "stats": {
                "total": len(assumptions),
                "types": {k: len(v) for k, v in by_type.items()}
            },
            "enrichment_prompt": enrichment_prompt,
            "instructions": f"Optionally process enrichment_prompt with LLM, then call store_config_assumptions(project_id='{project_id}', assumptions_json='<result>')"
        }
        
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"infer_config_assumptions failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def store_config_assumptions(
    project_id: str,
    assumptions_json: str
) -> dict[str, Any]:
    """
    Store config assumptions in project_registry.
    
    Args:
        project_id: Project identifier
        assumptions_json: JSON string with assumptions data
        
    Returns:
        Confirmation with stored count
    """
    try:
        assumptions_data = json.loads(assumptions_json)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify project exists
        cur.execute(
            "SELECT project_id FROM project_registry WHERE project_id = %s",
            (project_id,)
        )
        if not cur.fetchone():
            return {"success": False, "error": f"Project not found: {project_id}"}
        
        # Store assumptions
        cur.execute(
            """
            UPDATE project_registry 
            SET config_assumptions = %s, updated_at = NOW()
            WHERE project_id = %s
            """,
            (json.dumps(assumptions_data), project_id)
        )
        conn.commit()
        
        count = len(assumptions_data.get("assumptions", assumptions_data.get("enriched_assumptions", [])))
        logger.info(f"v45f: Stored {count} assumptions for {project_id}")
        
        return {
            "success": True,
            "project_id": project_id,
            "assumptions_stored": count,
            "message": f"Config assumptions stored ({count} items)"
        }
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"store_config_assumptions failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def query_project_understanding(
    project_id: str
) -> dict[str, Any]:
    """
    Query unified project understanding.
    
    Aggregates v45 features into single response:
    - v45d: System map (module dependencies)
    - v45e: Schema intent (domain entities)
    - v45f: Config assumptions (implicit expectations)
    - Purpose chain (mission and user needs)
    
    Part of v45 Deep Project Understanding.
    
    Args:
        project_id: Project identifier
        
    Returns:
        PROJECT_CONTEXT with all understanding layers
    """
    result: dict[str, Any] = {
        "success": True,
        "project_id": project_id,
        "system_map": None,
        "schema_intent": None,
        "config_assumptions": None,
        "purpose": None,
        "stats": {"sections_populated": 0}
    }
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Get project purpose from registry
        cur.execute(
            """
            SELECT purpose_hierarchy, detected_entities, config_assumptions
            FROM project_registry 
            WHERE project_id = %s
            """,
            (project_id,)
        )
        row = cur.fetchone()
        
        if not row:
            return {"success": False, "error": f"Project not found: {project_id}"}
        
        # Extract cached data
        if row.get("purpose_hierarchy"):
            result["purpose"] = row["purpose_hierarchy"]
            result["stats"]["sections_populated"] += 1
            
        if row.get("detected_entities"):
            result["schema_intent"] = row["detected_entities"]
            result["stats"]["sections_populated"] += 1
            
        if row.get("config_assumptions"):
            result["config_assumptions"] = row["config_assumptions"]
            result["stats"]["sections_populated"] += 1
        
        # 2. Get system map (live aggregation)
        system_map = await get_system_map(project_id)
        if system_map.get("success") and system_map.get("nodes"):
            result["system_map"] = {
                "nodes": system_map["nodes"],
                "edges": system_map["edges"],
                "stats": system_map["stats"]
            }
            result["stats"]["sections_populated"] += 1
        
        # 3. Add summary
        result["summary"] = _build_understanding_summary(result)
        
        logger.info(f"v45: Project understanding for {project_id}: {result['stats']['sections_populated']}/4 sections")
        
        return result
        
    except Exception as e:
        logger.error(f"query_project_understanding failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


def _build_understanding_summary(understanding: dict) -> str:
    """Build human-readable summary of project understanding."""
    lines = []
    
    purpose = understanding.get("purpose")
    if purpose:
        mission = str(purpose.get("mission", "Unknown mission"))[:100]
        lines.append(f"**Mission**: {mission}")
    
    system_map = understanding.get("system_map")
    if system_map:
        nodes = len(system_map.get("nodes", []))
        edges = len(system_map.get("edges", []))
        lines.append(f"**Architecture**: {nodes} modules, {edges} dependencies")
    
    schema = understanding.get("schema_intent")
    if schema:
        entities = schema.get("entities", schema) if isinstance(schema, dict) else schema
        if isinstance(entities, list):
            lines.append(f"**Schema**: {len(entities)} domain entities")
    
    config = understanding.get("config_assumptions")
    if config:
        assumptions = config.get("assumptions", config) if isinstance(config, dict) else config
        if isinstance(assumptions, list):
            lines.append(f"**Config**: {len(assumptions)} implicit assumptions")
    
    return "\n".join(lines) if lines else "No project understanding data available."


def _build_grounding_context(
    file_purpose: dict | None,
    module_purpose: dict | None, 
    project_purpose: dict | None
) -> str:
    """Build grounding context string for hypothesis generation."""
    lines = []
    
    if project_purpose and project_purpose.get("mission"):
        lines.append(f"**Project Mission**: {project_purpose['mission']}")
    
    if module_purpose and module_purpose.get("module_purpose"):
        lines.append(f"**Module Role**: {module_purpose['module_purpose']}")
    
    if file_purpose and file_purpose.get("problem"):
        lines.append(f"**File Purpose**: {file_purpose['problem']}")
    
    if file_purpose and file_purpose.get("project_contribution"):
        lines.append(f"**Contribution**: {file_purpose['project_contribution']}")
    
    return "\n".join(lines) if lines else "No purpose context available"


@mcp.tool()
async def advance_metacognitive_stage(
    session_id: str,
    target_stage: int = None  # None = advance by 1
) -> dict[str, Any]:
    """
    Advance session to next metacognitive stage.
    
    Based on arXiv:2308.05342v4 5-stage metacognitive prompting.
    Stages: 1=Understanding, 2=Preliminary, 3=Critical, 4=Final, 5=Confidence
    
    Args:
        session_id: The reasoning session UUID
        target_stage: Specific stage to advance to (default: current + 1)
        
    Returns:
        Stage info with prompt for the new stage
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get current stage
        cur.execute(
            "SELECT metacognitive_stage, state FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        
        if not session:
            return {"success": False, "error": f"Session not found: {session_id}"}
        
        current_stage = session["metacognitive_stage"] or 0
        
        # Determine target stage
        if target_stage is None:
            target_stage = current_stage + 1
        
        # Validate progression
        validation = validate_stage_progression(current_stage, target_stage)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["reason"],
                "current_stage": current_stage,
                "current_stage_info": format_stage_status(current_stage)
            }
        
        # Update stage
        cur.execute(
            "UPDATE reasoning_sessions SET metacognitive_stage = %s WHERE id = %s",
            (target_stage, session_id)
        )
        conn.commit()
        
        # Get stage info
        stage_info = get_stage_info(target_stage)
        
        logger.info(f"v40: Session {session_id[:8]} advanced to metacognitive stage {target_stage}")
        
        return {
            "success": True,
            "session_id": session_id,
            "previous_stage": current_stage,
            "current_stage": target_stage,
            "stage_name": stage_info.get("name"),
            "stage_prompt": stage_info.get("prompt"),
            "required_output": stage_info.get("required_output"),
            "calibration_guidance": get_calibration_guidance(0.7) if target_stage == 5 else None
        }
        
    except Exception as e:
        logger.error(f"advance_metacognitive_stage failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_metacognitive_status(
    session_id: str
) -> dict[str, Any]:
    """
    Get current metacognitive stage for a session.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Current stage info and progress
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            "SELECT metacognitive_stage, state FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session = cur.fetchone()
        
        if not session:
            return {"success": False, "error": f"Session not found: {session_id}"}
        
        stage = session["metacognitive_stage"] or 0
        stage_info = get_stage_info(stage)
        
        return {
            "success": True,
            "session_id": session_id,
            "current_stage": stage,
            "stage_name": stage_info.get("name"),
            "progress": f"{stage}/5",
            "stages_remaining": 5 - stage,
            "next_stage_prompt": get_stage_prompt(stage + 1) if stage < 5 else None
        }
        
    except Exception as e:
        logger.error(f"get_metacognitive_status failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_calibration_stats(
    min_samples: int = 10
) -> dict[str, Any]:
    """
    Get current calibration metrics for CSR self-evaluation.
    
    Computes Brier score (lower = better calibration) and 
    overconfidence bias (positive = overconfident).
    
    Args:
        min_samples: Minimum samples required for stats (default: 10)
        
    Returns:
        Calibration statistics with warnings if overconfident
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get recent calibration records
        cur.execute(
            """
            SELECT predicted_confidence, actual_outcome
            FROM calibration_records
            ORDER BY recorded_at DESC
            LIMIT 100
            """
        )
        records = [{"predicted_confidence": r["predicted_confidence"], "actual_outcome": r["actual_outcome"]} for r in cur.fetchall()]
        
        if len(records) < min_samples:
            return {
                "success": True,
                "sample_count": len(records),
                "sufficient_samples": False,
                "message": f"Need {min_samples} samples for calibration stats, have {len(records)}"
            }
        
        stats = compute_calibration_stats(records)
        
        logger.info(f"v40: Calibration stats computed - Brier: {stats['brier_score']}, Bias: {stats['overconfidence_bias']}")
        
        return {
            "success": True,
            **format_calibration_for_response(stats)
        }
        
    except Exception as e:
        logger.error(f"get_calibration_stats failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def get_self_awareness() -> dict[str, Any]:
    """
    Get PAS self-knowledge: schema, tools, statistics, architecture.
    
    Enables PAS to understand its own capabilities and structure.
    Part of v40 Phase 4 Self-Awareness implementation.
    
    Returns:
        Combined self-awareness info including schema, tools, stats, architecture
    """
    try:
        conn = get_db_connection()
        
        # Get schema info
        schema_info = get_schema_info(conn)
        
        # Get tool registry
        tool_info = get_tool_registry(mcp)
        
        # Get session statistics
        stats = get_session_statistics(conn)
        
        logger.info(f"v40: Self-awareness retrieved - {schema_info.get('table_count', 0)} tables, {len(tool_info)} tools")
        
        return {
            "success": True,
            "schema": schema_info,
            "tools": {
                "count": len(tool_info),
                "categories": list(set(t.get("category") for t in tool_info if t.get("category"))),
                "tools": tool_info
            },
            "statistics": stats,
            "architecture": ARCHITECTURE_MAP
        }
        
    except Exception as e:
        logger.error(f"get_self_awareness failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Self-Learning Tools (Phase 4)
# =============================================================================


@mcp.tool()
async def tag_session(
    session_id: str,
    tags: str  # Comma-separated tags for universal LLM compatibility
) -> dict[str, Any]:
    """
    Add tags to a reasoning session for organization and retrieval.
    
    Tags are normalized (lowercase, trimmed) and deduplicated.
    If a tag has an alias defined, the canonical form is used.
    
    Args:
        session_id: The reasoning session UUID
        tags: Comma-separated list of tags (e.g., "backend, database, v27")
        
    Returns:
        Confirmation with normalized tags applied
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Parse and normalize tags
        tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        if not tag_list:
            return {"success": False, "error": "No valid tags provided"}
        
        # Check aliases and normalize
        normalized_tags = []
        for tag in tag_list:
            cur.execute(
                "SELECT canonical_tag FROM tag_aliases WHERE alias = %s",
                (tag,)
            )
            alias_row = cur.fetchone()
            canonical = alias_row["canonical_tag"] if alias_row else tag
            if canonical not in normalized_tags:
                normalized_tags.append(canonical)
        
        # Insert tags (ignore duplicates)
        inserted = 0
        for tag in normalized_tags:
            try:
                cur.execute(
                    """
                    INSERT INTO session_tags (session_id, tag)
                    VALUES (%s, %s)
                    ON CONFLICT (session_id, tag) DO NOTHING
                    """,
                    (session_id, tag)
                )
                if cur.rowcount > 0:
                    inserted += 1
            except Exception as e:
                logger.warning(f"v26 tag insert failed for '{tag}': {e}")
        
        conn.commit()
        
        # Get all tags for session
        cur.execute(
            "SELECT tag FROM session_tags WHERE session_id = %s ORDER BY created_at",
            (session_id,)
        )
        all_tags = [r["tag"] for r in cur.fetchall()]
        
        logger.info(f"v26: Tagged session {session_id} with {inserted} new tags")
        
        return {
            "success": True,
            "session_id": session_id,
            "tags_added": inserted,
            "all_tags": all_tags,
            "message": f"Session tagged with {len(all_tags)} tags"
        }
        
    except Exception as e:
        logger.error(f"tag_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def record_outcome(
    session_id: str,
    outcome: str,
    confidence: float = 1.0,
    notes: Optional[str] = None,
    failure_reason: Optional[str] = None,  # v15b: explicit failure reason for learning
    keep_open: bool = False
) -> dict[str, Any]:
    """
    Record the outcome of a reasoning session for learning.
    
    Stores the outcome with attribution to the winning path nodes.
    This data enables PAS to learn which laws correlate with success.
    
    Args:
        session_id: The reasoning session UUID
        outcome: 'success', 'partial', or 'failure'
        confidence: How confident is this assessment (0.0-1.0)
        notes: Optional notes about why this outcome
        failure_reason: v15b - Explicit reason for failure (enables semantic learning)
        keep_open: If True, don't auto-complete even on success/failure
        
    Returns:
        Confirmation with stats about attributed nodes
    """

    # Phase 8: Defer outcome validation until we know the session mode
    # Valid outcomes vary by mode (e.g., research allows 'knowledge_gained')
    
    confidence = max(0.0, min(1.0, confidence))
    
    try:
        conn = get_db_connection()
        conn.rollback()  # Defensive: ensure clean transaction state
        cur = conn.cursor()
        
        # Phase 8: Fetch session to get mode for outcome validation
        cur.execute(
            "SELECT session_mode FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        session_row = cur.fetchone()
        if not session_row:
            return {"success": False, "error": "Session not found"}
        
        session_mode = session_row.get("session_mode", "implementation")
        mode_config = MODE_CONFIG.get(session_mode, MODE_CONFIG["implementation"])
        valid_outcomes = mode_config["outcomes"]
        
        if outcome not in valid_outcomes:
            return {
                "success": False, 
                "error": f"Invalid outcome for {session_mode} mode. Valid: {valid_outcomes}"
            }
        
        # Get the best path for this session
        cur.execute(
            """
            SELECT id, path, content, supporting_laws
            FROM thought_nodes
            WHERE session_id = %s AND node_type = 'hypothesis'
            ORDER BY posterior_score DESC
            LIMIT 1
            """,
            (session_id,)
        )
        best_node = cur.fetchone()
        
        if not best_node:
            return {"success": False, "error": "No hypotheses found in session"}
        
        winning_path = best_node["path"]
        
        # =====================================================================
        # Phase 4 Refactor: Use extracted helpers
        # =====================================================================
        
        # Helper 1: v15b/v27 - Compute embeddings
        scope_embedding, failure_reason_embedding = compute_outcome_embeddings(
            cur, best_node["id"], failure_reason, get_embedding
        )
        
        # Helper 2: Insert + attribute + v12b success counts
        record, stats = insert_and_attribute_outcome(
            cur, session_id, outcome, confidence, winning_path,
            notes, failure_reason, scope_embedding, failure_reason_embedding
        )
        
        # Helper 3: v12a - Log training data
        log_training_data(cur, session_id, winning_path, outcome)
        
        # Helper 4: v13c - Record critique accuracy
        record_critique_accuracy(
            cur, session_id, winning_path, outcome, _compute_critique_accuracy
        )
        
        # Auto-complete session if outcome is definitive (not partial) and not keep_open
        session_completed = False
        if outcome in ('success', 'failure') and not keep_open:
            cur.execute(
                "UPDATE reasoning_sessions SET state = 'completed' WHERE id = %s",
                (session_id,)
            )
            session_completed = True
        
        # Helper 5: v40 - Log calibration data
        log_calibration_record(cur, session_id, winning_path, outcome, map_outcome_to_numeric)
        
        # Helper 6: v22 - Persist user traits
        traits_persisted = persist_user_traits(cur, session_id, outcome, _get_outcome_multiplier)
        
        conn.commit()
        
        # Helper 7: v8b - Auto-trigger refresh (post-commit)
        auto_refresh_result = await trigger_auto_refresh(cur, refresh_law_weights)
        
        # Helper 8: v34 - Auto-apply tags (post-commit)
        auto_tagged = apply_auto_tags(cur, conn, session_id, outcome)
        
        # Phase 4: Check if synthesis was skipped in critique flow
        synthesis_warning = check_synthesis_critique_warning(cur, session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "outcome_id": str(record["id"]),
            "outcome": outcome,
            "confidence": confidence,
            "winning_path": winning_path,
            "attributed_nodes": stats["node_count"] or 0,
            "laws_affected": stats["laws"] or [],
            "session_completed": session_completed,
            "auto_refresh_triggered": auto_refresh_result is not None,
            "auto_refresh_result": auto_refresh_result,
            "auto_tagged": auto_tagged,  # v34
            "synthesis_warning": synthesis_warning,  # Phase 4
            "message": f"Outcome recorded. {stats['node_count'] or 0} nodes in winning path." + 
                      (" Session auto-completed." if session_completed else "") +
                      (f" Auto-refreshed {auto_refresh_result.get('laws_updated', 0)} law weights." if auto_refresh_result else "") +
                      (f" Auto-tagged: {auto_tagged}" if auto_tagged else "") +
                      (f" {synthesis_warning['message']}" if synthesis_warning else "")
        }
        
    except Exception as e:
        logger.error(f"record_outcome failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Phase 12: Session Handoff/Onboard System
# =============================================================================


@mcp.tool()
async def create_handoff(
    session_id: str,
    summary: str,
    next_task: Optional[str] = None,
    context: Optional[str | dict] = None,
    linked_artifacts: Optional[str] = None,
    linked_sessions: Optional[str] = None
) -> dict[str, Any]:
    """
    Create a handoff record for session continuity.
    
    Call this at end of session to preserve context for future agents.
    The summary is embedded for semantic search via onboard_session.
    
    Args:
        session_id: PAS session being handed off
        summary: Agent-generated summary of work done
        next_task: Optional suggested next step
        context: Optional JSON string with key context (decisions, blockers)
        linked_artifacts: Optional comma-separated artifact paths
        linked_sessions: Optional comma-separated related session IDs
        
    Returns:
        Created handoff record with ID
    """
    from pas.helpers.handoff import create_handoff_record
    
    conn = get_db_connection()
    try:
        # Parse optional JSON/list parameters
        # Handle context as either JSON string or dict (MCP may pass either)
        if context is None:
            context_dict = None
        elif isinstance(context, dict):
            context_dict = context
        else:
            context_dict = json.loads(context)
        artifacts_list = [a.strip() for a in linked_artifacts.split(",")] if linked_artifacts else None
        sessions_list = [s.strip() for s in linked_sessions.split(",")] if linked_sessions else None
        
        result = create_handoff_record(
            conn,
            session_id=session_id,
            summary=summary,
            next_task=next_task,
            context=context_dict,
            linked_artifacts=artifacts_list,
            linked_sessions=sessions_list
        )
        
        if result["success"]:
            result["message"] = f"Handoff created. Use onboard_session(handoff_id='{result['handoff_id']}') to restore context."
        
        return result
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON in context parameter: {e}"}
    except Exception as e:
        logger.error(f"create_handoff failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        safe_close_connection(conn)


@mcp.tool()
async def onboard_session(
    project_id: Optional[str] = None,
    topic: Optional[str] = None,
    handoff_id: Optional[str] = None,
    limit: int = 5,
    mark_processed: bool = False
) -> dict[str, Any]:
    """
    Retrieve handoff context for session continuity.
    
    Call at start of session to restore context from previous work.
    
    Args:
        project_id: Filter handoffs by project
        topic: Semantic search query (e.g., "Phase 7 interview system")
        handoff_id: Get specific handoff by ID
        limit: Maximum results (default 5)
        mark_processed: If True, mark retrieved handoff(s) as processed
        
    Returns:
        Handoff context formatted for LLM consumption
    """
    from pas.helpers.handoff import (
        list_active_handoffs,
        search_handoffs,
        get_handoff_by_id,
        mark_handoff_processed
    )
    
    conn = get_db_connection()
    try:
        # Mode 1: Get specific handoff by ID
        if handoff_id:
            handoff = get_handoff_by_id(conn, handoff_id)
            if not handoff:
                return {"success": False, "error": f"Handoff {handoff_id} not found"}
            
            if mark_processed:
                mark_handoff_processed(conn, handoff_id)
                handoff["status"] = "processed"
            
            return {
                "success": True,
                "mode": "specific",
                "handoff": handoff,
                "formatted_context": _format_handoff_context(handoff)
            }
        
        # Mode 2: Semantic search by topic
        if topic:
            handoffs = search_handoffs(conn, topic, project_id, limit)
            if not handoffs:
                return {
                    "success": True,
                    "mode": "search",
                    "query": topic,
                    "handoffs": [],
                    "message": "No matching handoffs found"
                }
            
            if mark_processed and len(handoffs) == 1:
                mark_handoff_processed(conn, handoffs[0]["handoff_id"])
                handoffs[0]["status"] = "processed"
            
            return {
                "success": True,
                "mode": "search",
                "query": topic,
                "handoffs": handoffs,
                "formatted_context": _format_handoff_list(handoffs)
            }
        
        # Mode 3: List active handoffs
        handoffs = list_active_handoffs(conn, project_id, limit)
        return {
            "success": True,
            "mode": "list",
            "project_id": project_id,
            "handoffs": handoffs,
            "active_count": len(handoffs),
            "formatted_context": _format_handoff_list(handoffs) if handoffs else "No active handoffs"
        }
    except Exception as e:
        logger.error(f"onboard_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        safe_close_connection(conn)


def _format_handoff_context(handoff: dict) -> str:
    """Format a single handoff for LLM consumption."""
    lines = [
        f"## 🔄 Handoff: {handoff['handoff_id'][:8]}",
        f"**Session**: `{handoff['session_id']}`" if handoff['session_id'] else "",
        f"**Project**: {handoff['project_id']}" if handoff['project_id'] else "",
        f"**Status**: {handoff['status']}",
        f"**Created**: {handoff['created_at']}",
        "",
        "### Summary",
        handoff['summary'],
    ]
    
    if handoff.get('next_task'):
        lines.extend(["", "### Next Task", handoff['next_task']])
    
    if handoff.get('linked_artifacts'):
        lines.extend(["", "### Linked Artifacts"])
        for artifact in handoff['linked_artifacts']:
            lines.append(f"- [{artifact}](file://{artifact})")
    
    if handoff.get('context'):
        lines.extend(["", "### Context", "```json", json.dumps(handoff['context'], indent=2), "```"])
    
    return "\n".join([l for l in lines if l is not None])


def _format_handoff_list(handoffs: list) -> str:
    """Format multiple handoffs as a list."""
    lines = ["## 🔄 Active Handoffs", ""]
    
    for i, h in enumerate(handoffs, 1):
        summary = h.get('summary', 'No summary')
        lines.extend([
            f"### {i}. {h.get('next_task', 'Handoff')} ({h['handoff_id'][:8]})",
            f"**Summary**: {summary[:150]}..." if len(summary) > 150 else f"**Summary**: {summary}",
            f"**Created**: {h['created_at']}",
            ""
        ])
    
    return "\n".join(lines)


# =============================================================================
# v39: RLVR patterns moved to learning_helpers.py
# - SUCCESS_PATTERNS, FAILURE_PATTERNS, FAILURE_REASON_PATTERNS
# =============================================================================


@mcp.tool()
async def parse_terminal_output(
    session_id: str,
    terminal_text: str,
    auto_record: bool = False
) -> dict[str, Any]:
    """
    Parse terminal output for success/failure signals (v17a RLVR).
    
    Uses domain-agnostic regex patterns to detect test pass/fail,
    build success/error, and runtime crashes. Returns structured
    signal detection with confidence scoring.
    
    Args:
        session_id: The reasoning session UUID
        terminal_text: Raw terminal output to analyze
        auto_record: If True and signal is clear, auto-call record_outcome
        
    Returns:
        Detected signal with confidence and matches
    """
    # Handle empty input
    if not terminal_text or not terminal_text.strip():
        return {
            "success": True,
            "signal": "unknown",
            "confidence": 0.0,
            "matches": [],
            "message": "Empty terminal output provided"
        }
    
    # Delegate to helper for core pattern matching
    parsed = parse_terminal_signals(terminal_text)
    
    # Format MCP response
    result = {
        "success": True,
        "session_id": session_id,
        "signal": parsed["signal"],
        "confidence": round(parsed["confidence"], 2),
        "success_matches": parsed["success_matches"],
        "failure_matches": parsed["failure_matches"],
        "total_success_signals": parsed["success_count"],
        "total_failure_signals": parsed["failure_count"],
        "failure_reason": parsed["failure_reason"],
        "message": f"Detected {parsed['signal']} signal with {parsed['confidence']:.0%} confidence"
    }
    
    # Auto-record if enabled and confidence is high
    if auto_record and parsed["signal"] != "unknown" and parsed["confidence"] >= 0.7:
        try:
            outcome_result = await record_outcome(
                session_id=session_id,
                outcome=parsed["signal"],
                confidence=parsed["confidence"],
                notes=f"Auto-recorded by v17a RLVR. Matches: {parsed['matches'][:3]}",
                failure_reason=parsed["failure_reason"]
            )
            result["auto_recorded"] = True
            result["outcome_result"] = outcome_result
            result["message"] = str(result["message"]) + f" Auto-recorded as {parsed['signal']}."
            if parsed["failure_reason"]:
                result["message"] = str(result["message"]) + " Failure reason extracted."
        except Exception as e:
            result["auto_recorded"] = False
            result["auto_record_error"] = str(e)
    else:
        result["auto_recorded"] = False
        if auto_record and parsed["signal"] == "unknown":
            result["message"] = str(result["message"]) + " Signal too ambiguous for auto-record."
        elif auto_record and parsed["confidence"] < 0.7:
            result["message"] = str(result["message"]) + f" Confidence too low ({parsed['confidence']:.0%}) for auto-record."
    
    return result


@mcp.tool()
async def refresh_law_weights(
    min_samples: int = 5,
    blend_factor: float = 0.5
) -> dict[str, Any]:
    """
    Update law weights based on accumulated outcome data.
    
    Computes success rates from outcome_records and blends with
    original scientific_weight. Only updates laws with sufficient data.
    
    Args:
        min_samples: Minimum outcomes required before updating (default: 5)
        blend_factor: How much new data influences weight (0-1, default: 0.5)
        
    Returns:
        Number of laws updated and their new weights
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get success rates for laws with sufficient data
        cur.execute(
            """
            SELECT 
                law_id,
                law_name,
                total_outcomes,
                success_rate
            FROM law_domain_stats
            WHERE total_outcomes >= %s
            """,
            (min_samples,)
        )
        stats = cur.fetchall()
        
        if not stats:
            return {
                "success": True,
                "laws_updated": 0,
                "message": f"No laws have {min_samples}+ recorded outcomes yet."
            }
        
        updated = []
        for row in stats:
            # Blend new success rate with original weight
            cur.execute(
                "SELECT scientific_weight FROM scientific_laws WHERE id = %s",
                (row["law_id"],)
            )
            original = cur.fetchone()
            if not original:
                continue
                
            old_weight = float(original["scientific_weight"])
            new_weight = (float(row["success_rate"]) * blend_factor) + (old_weight * (1 - blend_factor))
            new_weight = round(max(0.1, min(1.0, new_weight)), 3)
            
            # Update the weight
            cur.execute(
                """
                UPDATE scientific_laws 
                SET scientific_weight = %s,
                    domain_weights = domain_weights || %s
                WHERE id = %s
                """,
                (new_weight, json.dumps({"_last_update": str(row["success_rate"])}), row["law_id"])
            )
            
            updated.append({
                "law_name": row["law_name"],
                "samples": row["total_outcomes"],
                "success_rate": float(row["success_rate"]),
                "old_weight": old_weight,
                "new_weight": new_weight
            })
        
        conn.commit()
        
        return {
            "success": True,
            "laws_updated": len(updated),
            "updates": updated,
            "message": f"Updated {len(updated)} law weights based on {sum(u['samples'] for u in updated)} outcomes."
        }
        
    except Exception as e:
        logger.error(f"refresh_law_weights failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Session Lifecycle Tools (Phase 5)
# =============================================================================

@mcp.tool()
async def find_or_create_session(
    goal_text: str,
    similarity_threshold: float = 0.8
) -> dict[str, Any]:
    """
    Smart session router - finds existing sessions or creates new ones.
    
    Uses semantic similarity to detect related sessions:
    - If >threshold similarity to ACTIVE session: returns existing
    - If >threshold similarity to COMPLETED session: creates continuation
    - Otherwise: creates new session
    
    Args:
        goal_text: The goal/question for reasoning
        similarity_threshold: Cosine similarity threshold (default: 0.8)
        
    Returns:
        Session info with action taken (existing/continuation/new)
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate embedding for the goal
        goal_embedding = get_embedding(goal_text)
        
        # Search for similar sessions
        cur.execute(
            """
            SELECT id, goal, state, 
                   1 - (goal_embedding <=> %s::vector) as similarity
            FROM reasoning_sessions
            WHERE goal_embedding IS NOT NULL
            ORDER BY goal_embedding <=> %s::vector
            LIMIT 5
            """,
            (goal_embedding, goal_embedding)
        )
        similar = cur.fetchall()
        
        # Check for matches above threshold
        best_match = None
        for row in similar:
            if row["similarity"] and row["similarity"] >= similarity_threshold:
                best_match = row
                break
        
        if best_match:
            if best_match["state"] == "active":
                # Return existing active session
                return {
                    "success": True,
                    "action": "existing",
                    "session_id": str(best_match["id"]),
                    "goal": best_match["goal"],
                    "similarity": round(best_match["similarity"], 3),
                    "message": f"Found existing active session (similarity: {best_match['similarity']:.2f})"
                }
            else:
                # Create continuation of completed session
                cur.execute(
                    """
                    INSERT INTO reasoning_sessions (goal, goal_embedding, parent_session_id, context)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (goal_text, goal_embedding, best_match["id"], 
                     json.dumps({"continued_from": str(best_match["id"])}))
                )
                new_session = cur.fetchone()
                
                # Copy root node from parent as context
                cur.execute(
                    """
                    INSERT INTO thought_nodes (session_id, path, content, node_type)
                    SELECT %s, 'root', content, 'root'
                    FROM thought_nodes
                    WHERE session_id = %s AND path = 'root'
                    """,
                    (new_session["id"], best_match["id"])
                )
                
                conn.commit()
                return {
                    "success": True,
                    "action": "continuation",
                    "session_id": str(new_session["id"]),
                    "parent_session_id": str(best_match["id"]),
                    "goal": goal_text,
                    "similarity": round(best_match["similarity"], 3),
                    "message": f"Created continuation of completed session (similarity: {best_match['similarity']:.2f})"
                }
        
        # No match - create new session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (goal, goal_embedding)
            VALUES (%s, %s)
            RETURNING id, created_at
            """,
            (goal_text, goal_embedding)
        )
        new_session = cur.fetchone()
        
        # Create root node
        cur.execute(
            """
            INSERT INTO thought_nodes (session_id, path, content, node_type, embedding)
            VALUES (%s, 'root', %s, 'root', %s)
            """,
            (new_session["id"], goal_text, goal_embedding)
        )
        
        conn.commit()
        return {
            "success": True,
            "action": "new",
            "session_id": str(new_session["id"]),
            "goal": goal_text,
            "message": "Created new reasoning session"
        }
        
    except Exception as e:
        logger.error(f"find_or_create_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def complete_session(
    session_id: str,
    notes: Optional[str] = None
) -> dict[str, Any]:
    """
    Explicitly complete a reasoning session.
    
    Use this to manually close sessions that shouldn't auto-complete
    or to close abandoned sessions.
    
    Args:
        session_id: The reasoning session UUID
        notes: Optional notes about why session is being closed
        
    Returns:
        Confirmation of completion
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            UPDATE reasoning_sessions 
            SET state = 'completed', 
                context = context || %s
            WHERE id = %s
            RETURNING state
            """,
            (json.dumps({"completion_notes": notes or "Manually completed"}), session_id)
        )
        result = cur.fetchone()
        
        if not result:
            return {"success": False, "error": "Session not found"}
        
        conn.commit()
        return {
            "success": True,
            "session_id": session_id,
            "state": "completed",
            "message": "Session marked as completed"
        }
        
    except Exception as e:
        logger.error(f"complete_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def explore_alternatives(
    session_id: str
) -> dict[str, Any]:
    """
    Process pending mid-range hypothesis critiques.
    
    Retrieves critique_candidates from session context and returns
    critique prompts for each. Clears queue after retrieval.
    
    Args:
        session_id: The reasoning session UUID
        
    Returns:
        Critique prompts for each alternative, or empty if none pending
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get session context
        cur.execute(
            "SELECT context FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": "Session not found"}
        
        context = row["context"] or {}
        pending = context.get("pending_critiques", [])
        
        if not pending:
            return {"success": True, "message": "No pending alternatives", "alternatives": []}
        
        # Build critique prompts
        alternatives = []
        for candidate in pending:
            alternatives.append({
                "node_id": candidate["node_id"],
                "content": candidate["content"],
                "score": candidate["score"],
                "critique_prompt": f"Critique this alternative hypothesis: {candidate['content']}"
            })
        
        # Clear pending
        context["pending_critiques"] = []
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s WHERE id = %s",
            (json.dumps(context), session_id)
        )
        conn.commit()
        
        logger.info(f"v48: Returning {len(alternatives)} alternatives for exploration in session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "alternatives": alternatives,
            "instructions": "Generate critiques for each alternative. Log insights via log_conversation with log_type='alternative_insight'"
        }
        
    except Exception as e:
        logger.error(f"explore_alternatives failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def resume_session(
    session_id: str
) -> dict[str, Any]:
    """
    Resume a completed or paused session.
    
    Creates a continuation session that inherits context from the original.
    Use this when you want to build on previous reasoning.
    
    Args:
        session_id: The session UUID to resume
        
    Returns:
        New continuation session info
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get original session
        cur.execute(
            "SELECT id, goal, goal_embedding, state FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        original = cur.fetchone()
        
        if not original:
            return {"success": False, "error": "Session not found"}
        
        if original["state"] == "active":
            return {
                "success": True,
                "action": "already_active",
                "session_id": session_id,
                "message": "Session is already active"
            }
        
        # Create continuation session
        cur.execute(
            """
            INSERT INTO reasoning_sessions (goal, goal_embedding, parent_session_id, context)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (original["goal"], original["goal_embedding"], session_id,
             json.dumps({"resumed_from": session_id}))
        )
        new_session = cur.fetchone()
        
        # Copy best hypothesis from parent as starting context
        cur.execute(
            """
            INSERT INTO thought_nodes (session_id, path, content, node_type, prior_score, likelihood)
            SELECT %s, 'root', content, 'root', posterior_score, posterior_score
            FROM thought_nodes
            WHERE session_id = %s
            ORDER BY posterior_score DESC
            LIMIT 1
            """,
            (new_session["id"], session_id)
        )
        
        conn.commit()
        return {
            "success": True,
            "action": "resumed",
            "session_id": str(new_session["id"]),
            "parent_session_id": session_id,
            "goal": original["goal"],
            "message": f"Created continuation session, inheriting from {session_id}"
        }
        
    except Exception as e:
        logger.error(f"resume_session failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v25: Conversation Logging Tools
# =============================================================================

@mcp.tool()
async def search_conversation_log(
    query: str,
    session_id: Optional[str] = None,
    log_type: Optional[str] = None,
    limit: int = 10
) -> dict[str, Any]:
    """
    Search conversation logs by semantic similarity.
    
    Finds past user inputs, feedback, and context that match the query.
    Useful for understanding what the user has said across sessions.
    
    Args:
        query: The search text (will be embedded for semantic matching)
        session_id: Optional - limit search to specific session
        log_type: Optional - filter by type ('user_input', 'feedback', 'interview_answer', 'context')
        limit: Maximum results to return (default: 10)
        
    Returns:
        Matching log entries with their linked sessions and thought nodes
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate query embedding
        query_embedding = get_embedding(query)
        
        # Build query based on filters
        base_query = """
            SELECT 
                cl.id,
                cl.session_id,
                cl.thought_node_id,
                cl.user_id,
                cl.log_type,
                cl.raw_text,
                cl.created_at,
                1 - (cl.embedding <=> %s::vector) as similarity,
                rs.goal as session_goal,
                tn.content as linked_thought
            FROM conversation_log cl
            JOIN reasoning_sessions rs ON cl.session_id = rs.id
            LEFT JOIN thought_nodes tn ON cl.thought_node_id = tn.id
            WHERE cl.embedding IS NOT NULL
        """
        params: list[Any] = [query_embedding]
        
        if session_id:
            base_query += " AND cl.session_id = %s"
            params.append(session_id)
        
        if log_type:
            base_query += " AND cl.log_type = %s"
            params.append(log_type)
        
        base_query += " ORDER BY cl.embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, limit])
        
        cur.execute(base_query, params)
        results = cur.fetchall()
        
        entries = []
        for r in results:
            entries.append({
                "id": str(r["id"]),
                "session_id": str(r["session_id"]),
                "thought_node_id": str(r["thought_node_id"]) if r["thought_node_id"] else None,
                "user_id": r["user_id"],
                "log_type": r["log_type"],
                "raw_text": r["raw_text"][:500] + "..." if len(r["raw_text"]) > 500 else r["raw_text"],
                "full_text_length": len(r["raw_text"]),
                "similarity": round(float(r["similarity"]), 4),
                "session_goal": r["session_goal"][:100] + "..." if len(r["session_goal"]) > 100 else r["session_goal"],
                "linked_thought": r["linked_thought"][:150] + "..." if r["linked_thought"] and len(r["linked_thought"]) > 150 else r["linked_thought"],
                "created_at": r["created_at"].isoformat()
            })
        
        return {
            "success": True,
            "query": query,
            "count": len(entries),
            "entries": entries,
            "filters_applied": {
                "session_id": session_id,
                "log_type": log_type
            }
        }
        
    except Exception as e:
        logger.error(f"search_conversation_log failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def log_conversation(
    session_id: str,
    raw_text: str,
    log_type: str = "context",
    thought_node_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Manually log free-form text to a session's conversation log.
    
    Use this to capture user context, feedback, or notes that aren't
    directly tied to hypothesis creation.
    
    Args:
        session_id: The reasoning session UUID
        raw_text: The text to log (no truncation)
        log_type: One of 'user_input', 'feedback', 'interview_answer', 'context' (default: 'context')
        thought_node_id: Optional - link to a specific thought node
        user_id: Optional - user identifier for multi-user scenarios
        
    Returns:
        Created log entry ID
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Verify session exists
        cur.execute("SELECT id FROM reasoning_sessions WHERE id = %s", (session_id,))
        if not cur.fetchone():
            return {"success": False, "error": "Session not found"}
        
        # Verify thought_node if provided
        if thought_node_id:
            cur.execute("SELECT id FROM thought_nodes WHERE id = %s", (thought_node_id,))
            if not cur.fetchone():
                return {"success": False, "error": "Thought node not found"}
        
        # Generate embedding (truncate for embedding, but store full text)
        embedding = get_embedding(raw_text[:2000])
        
        cur.execute(
            """
            INSERT INTO conversation_log (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
        )
        result = cur.fetchone()
        conn.commit()
        
        return {
            "success": True,
            "log_id": str(result["id"]),
            "session_id": session_id,
            "log_type": log_type,
            "text_length": len(raw_text),
            "created_at": result["created_at"].isoformat(),
            "message": f"Logged {len(raw_text)} characters to conversation_log"
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"log_conversation failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)

# =============================================================================
# v30: Global Codebase Understanding Tools
# =============================================================================

import hashlib
from pathlib import Path

# Language extensions for tree-sitter
LANGUAGE_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'c_sharp',
    '.rb': 'ruby',
    '.php': 'php',
}

# Patterns to skip during sync
SKIP_PATTERNS = {
    '__pycache__', 'node_modules', '.git', '.venv', 'venv',
    'dist', 'build', '.egg-info', '.tox', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', 'coverage', 'htmlcov'
}


def _normalize_project_id(project_path: str) -> str:
    """Derive consistent project_id from path."""
    # Get the final directory name, lowercase alphanumeric only
    path = Path(project_path).resolve()
    name = path.name.lower()
    # Remove non-alphanumeric chars
    return ''.join(c for c in name if c.isalnum() or c == '_')


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# v39: _extract_symbols moved to codebase_helpers.py
# =============================================================================


# =============================================================================
# Phase 7c: GEMINI.md Constraint Sync
# =============================================================================

@mcp.tool()
async def sync_gemini_constraints(
    project_id: str,
    project_path: str,
    direction: str = "detect_only"
) -> dict[str, Any]:
    """
    Sync constraints between GEMINI.md and database.
    
    Uses hybrid LLM/embedding approach:
    - Returns extraction_prompt for agent LLM to parse GEMINI.md
    - Uses embedding similarity for drift detection
    
    Args:
        project_id: Project identifier (e.g., 'mcp-pas')
        project_path: Path to project root containing GEMINI.md
        direction: Sync direction
            - detect_only: Just detect drift, return extraction_prompt
            - file_to_db: GEMINI.md overwrites DB (call after LLM extraction)
            - db_to_file: DB overwrites GEMINI.md (not implemented)
    
    Returns:
        Sync results with extraction_prompt if needed
    """
    try:
        from pas.helpers.gemini_sync import (
            sync_gemini_constraints as _sync_gemini_constraints
        )
        
        result = await _sync_gemini_constraints(project_id, project_path, direction)
        return result
        
    except Exception as e:
        logger.error(f"sync_gemini_constraints failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def store_extracted_constraints(
    project_id: str,
    constraints_json: str
) -> dict[str, Any]:
    """
    Store LLM-extracted constraints from GEMINI.md.
    
    Call this after processing the extraction_prompt from sync_gemini_constraints.
    
    Args:
        project_id: Project identifier
        constraints_json: JSON array of extracted constraints from LLM
            Format: [{"type": "philosophy", "key": "no_mvp", "value": true, "enforcement": "block"}, ...]
    
    Returns:
        Confirmation with stored count
    """
    try:
        import json
        from pas.helpers.gemini_sync import store_extracted_constraints as _store_constraints
        
        constraints = json.loads(constraints_json)
        if not isinstance(constraints, list):
            return {"success": False, "error": "constraints_json must be a JSON array"}
        
        result = _store_constraints(project_id, constraints)
        return result
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"store_extracted_constraints failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def detect_constraint_drift(
    project_id: str,
    constraints_json: str,
    threshold: float = 0.15
) -> dict[str, Any]:
    """
    Detect semantic drift between GEMINI.md and database constraints.
    
    Call after LLM extraction to check if values have changed semantically.
    
    Args:
        project_id: Project identifier
        constraints_json: JSON array of extracted constraints from GEMINI.md
        threshold: Drift threshold (default 0.15 = 15% semantic change)
    
    Returns:
        Drift detection results with resolution_prompt if drift detected
    """
    try:
        import json
        from pas.helpers.gemini_sync import detect_drift
        
        constraints = json.loads(constraints_json)
        result = detect_drift(project_id, constraints, threshold)
        return result
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"detect_constraint_drift failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def write_gemini_export(
    project_path: str,
    content: str,
    mode: Literal["append", "replace"] = "append"
) -> dict[str, Any]:
    """
    Write exported constraints to GEMINI.md file.
    
    Use this after reviewing the export_content from sync_gemini_constraints(direction='db_to_file').
    Supports idempotent section replacement - repeated calls update the same section.
    
    Args:
        project_path: Absolute path to project root containing GEMINI.md
        content: Exported markdown content from sync_gemini_constraints(direction='db_to_file')
        mode: 'append' (add/replace section - recommended) or 'replace' (overwrite entire file)
    
    Returns:
        Success status with file path and mode used
    """
    try:
        from pas.helpers.gemini_sync import write_gemini_export as _write_gemini_export
        
        result = _write_gemini_export(project_path, content, mode)
        return result
        
    except Exception as e:
        logger.error(f"write_gemini_export failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def resolve_constraint_drift(
    project_id: str,
    project_path: str,
    resolution: Literal["file_to_db", "db_to_file", "skip"],
    drifted_keys: Optional[str] = None,
    file_constraints_json: Optional[str] = None
) -> dict[str, Any]:
    """
    Resolve conflicts between GEMINI.md and database constraints.
    
    Call after detect_constraint_drift returns drifts. This applies
    the user's chosen resolution strategy.
    
    Args:
        project_id: Project identifier
        project_path: Absolute path to project root
        resolution: How to resolve:
            - file_to_db: GEMINI.md values overwrite DB (choice A)
            - db_to_file: DB values overwrite GEMINI.md (choice B)
            - skip: Leave unresolved for manual handling (choice C)
        drifted_keys: Optional JSON array of specific keys to resolve
        file_constraints_json: Required for file_to_db - JSON array of constraints from GEMINI.md
        
    Returns:
        Resolution results with counts and paths affected
    """
    try:
        from pas.helpers.gemini_sync import resolve_constraint_drift as _resolve
        
        # Parse optional JSON args
        drifted = None
        if drifted_keys:
            drifted = json.loads(drifted_keys)
            
        file_constraints = None
        if file_constraints_json:
            file_constraints = json.loads(file_constraints_json)
        
        result = _resolve(
            project_id=project_id,
            project_path=project_path,
            resolution=resolution,
            drifted_keys=drifted,
            file_constraints=file_constraints
        )
        return result
        
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}
    except Exception as e:
        logger.error(f"resolve_constraint_drift failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()

async def sync_project(
    project_path: str,
    project_id: Optional[str] = None,
    max_files: Optional[int] = None,
    max_file_size_kb: int = 100,
    include_references: bool = False,
    include_call_hierarchy: bool = False,
    auto_understand: bool = True
) -> dict[str, Any]:
    """
    Index a project directory for codebase understanding.
    
    v51 Delta Sync: Uses mtime gating to skip unchanged files and
    purges orphan records for deleted files.
    
    v61: auto_understand populates system_map, schema_intent, config_assumptions.
    
    Args:
        project_path: Absolute path to project root
        project_id: Optional custom project ID (auto-derived from path if not provided)
        max_files: Maximum files to index (None = unlimited, default None)
        max_file_size_kb: Skip files larger than this (default 100KB)
        include_references: If True, index symbol references via LSP (slower)
        include_call_hierarchy: If True, index call hierarchy via LSP (slower)
        auto_understand: If True (default), populate deep understanding data
    
    Returns:
        Sync results with counts and any errors
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        path = Path(project_path).resolve()
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {project_path}"}
        
        # Normalize project_id
        pid = project_id or _normalize_project_id(project_path)
        
        # v51: Get existing files with mtime for delta sync
        cur.execute(
            "SELECT file_path, file_hash, mtime_ns FROM file_registry WHERE project_id = %s",
            (pid,)
        )
        existing = {row['file_path']: {'hash': row['file_hash'], 'mtime': row['mtime_ns']} for row in cur.fetchall()}
        
        stats = {
            'files_scanned': 0,
            'files_added': 0,
            'files_updated': 0,
            'files_unchanged': 0,
            'files_skipped': 0,
            'files_purged': 0,  # v51: orphan purge count
            'symbols_extracted': 0,
            'lsp_symbols': 0,  # v52: LSP-sourced symbols
            'treesitter_fallback': 0,  # v52: files using tree-sitter fallback
            'references_indexed': 0,  # v68: LSP reference count
            'calls_indexed': 0,  # v68: LSP call hierarchy count
            'errors': []
        }
        seen_paths: set[str] = set()  # v51: track for orphan detection
        files_to_index_refs: list[dict] = []  # v68a: collect files for second-pass ref indexing
        
        # v52: Initialize LSP pool for symbol extraction
        lsp_pool = None
        try:
            from pas.lsp.lsp_pool import LspPool
            lsp_pool = await LspPool.get(str(path))
            logger.info(f"sync_project: LSP pool active for {path}")
        except Exception as lsp_err:
            logger.info(f"sync_project: LSP not available, using tree-sitter only: {lsp_err}")
        
        # Walk directory
        for file_path in path.rglob('*'):
            # v51: max_files is optional (None = unlimited)
            if max_files is not None and stats['files_scanned'] >= max_files:
                break
            
            # Skip directories and non-files
            if not file_path.is_file():
                continue
            
            # Skip hidden and unwanted directories
            rel_parts = file_path.relative_to(path).parts
            if any(p in SKIP_PATTERNS or p.startswith('.') for p in rel_parts[:-1]):
                continue
            
            # Skip large files
            try:
                size_kb = file_path.stat().st_size / 1024
                if size_kb > max_file_size_kb:
                    stats['files_skipped'] += 1
                    continue
            except OSError:
                continue
            
            # Check extension for language support
            ext = file_path.suffix.lower()
            language = LANGUAGE_EXTENSIONS.get(ext)
            if not language:
                stats['files_skipped'] += 1
                continue
            
            stats['files_scanned'] += 1
            rel_path = str(file_path.relative_to(path))
            seen_paths.add(rel_path)  # v51: track for orphan detection
            
            try:
                # v51: Get current mtime for delta detection
                file_stat = file_path.stat()
                current_mtime = file_stat.st_mtime_ns
                
                # v51: mtime gating - skip I/O if mtime unchanged
                if rel_path in existing:
                    stored = existing[rel_path]
                    if stored['mtime'] and current_mtime <= stored['mtime']:
                        stats['files_unchanged'] += 1
                        continue
                
                # Compute hash (only if mtime changed or new file)
                file_hash = _compute_file_hash(file_path)
                
                # Read content
                content = file_path.read_text(encoding='utf-8', errors='replace')
                line_count = content.count('\n') + 1
                
                # Generate embedding for file content (chunk if needed)
                # Use first 2000 chars for embedding
                embedding = get_embedding(content[:2000])
                
                # Upsert file_registry with mtime
                cur.execute(
                    """
                    INSERT INTO file_registry (project_id, file_path, file_hash, language, line_count, content_embedding, mtime_ns)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, file_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        language = EXCLUDED.language,
                        line_count = EXCLUDED.line_count,
                        content_embedding = EXCLUDED.content_embedding,
                        mtime_ns = EXCLUDED.mtime_ns,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (pid, rel_path, file_hash, language, line_count, embedding, current_mtime)
                )
                file_id = cur.fetchone()['id']
                
                if rel_path in existing:
                    stats['files_updated'] += 1
                else:
                    stats['files_added'] += 1
                
                # v52: Extract symbols - try LSP first, fallback to tree-sitter
                symbols = []
                if lsp_pool and language == "python":
                    try:
                        symbols = await _extract_symbols_lsp(str(file_path), lsp_pool)
                        if symbols:
                            stats['lsp_symbols'] += len(symbols)
                    except Exception as lsp_e:
                        logger.debug(f"LSP symbol extraction failed for {rel_path}: {lsp_e}")
                
                if not symbols:
                    # Fallback to tree-sitter
                    symbols = _extract_symbols(content, language)
                    if symbols:
                        stats['treesitter_fallback'] += 1
                
                # Clear old symbols for this file
                cur.execute("DELETE FROM file_symbols WHERE file_id = %s", (file_id,))
                
                for sym in symbols:
                    # Generate embedding for signature+docstring
                    embed_text = sym.get('signature', '') + '\n' + sym.get('docstring', '')
                    sym_embedding = get_embedding(embed_text[:500])
                    
                    cur.execute(
                        """
                        INSERT INTO file_symbols (file_id, symbol_type, symbol_name, line_start, line_end, signature, docstring, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (file_id, sym['type'], sym['name'], sym.get('line_start'), 
                         sym.get('line_end'), sym.get('signature'), sym.get('docstring'), sym_embedding)
                    )
                    stats['symbols_extracted'] += 1
                
                # v68a: Collect files for second-pass reference indexing
                if (include_references or include_call_hierarchy) and language == "python":
                    files_to_index_refs.append({
                        'file_path': str(file_path),
                        'rel_path': rel_path,
                        'symbols': symbols
                    })
                
            except Exception as e:
                stats['errors'].append(f"{rel_path}: {str(e)[:100]}")
                if len(stats['errors']) > 10:
                    break
        
        conn.commit()
        
        # v51: Purge orphan files (deleted from disk but still in DB)
        orphan_paths = set(existing.keys()) - seen_paths
        if orphan_paths:
            cur.execute(
                "DELETE FROM file_registry WHERE project_id = %s AND file_path = ANY(%s)",
                (pid, list(orphan_paths))
            )
            stats['files_purged'] = len(orphan_paths)
            conn.commit()
        
        # v68a: Second-pass reference/call hierarchy indexing (after LSP has analyzed project)
        if files_to_index_refs and lsp_pool and (include_references or include_call_hierarchy):
            logger.info(f"sync_project: Second pass - waiting 5s for LSP analysis to complete...")
            await asyncio.sleep(5)  # Give basedpyright time to analyze the project
            
            for file_info in files_to_index_refs:
                file_path_str = file_info['file_path']
                rel_path = file_info['rel_path']
                symbols = file_info['symbols']
                
                # Clear old references for this file
                cur.execute(
                    "DELETE FROM symbol_references WHERE source_file = %s AND project_id = %s",
                    (rel_path, pid)
                )
                
                # Index references
                if include_references:
                    for sym in symbols:
                        try:
                            sym_line = sym.get('line_start', 1)
                            lsp_line = sym_line - 1  # LSP is 0-indexed
                            # v68a: Column offset heuristic - cursor must be ON the symbol name
                            # Python: 'def ' = 4 chars, 'class ' = 6 chars, 'async def ' = 10 chars
                            sym_type = sym.get('type', '')
                            col_offset = 4 if sym_type in ('function', 'method') else 6 if sym_type == 'class' else 0
                            refs = await lsp_pool.find_references(file_path_str, lsp_line, col_offset)
                            for ref in refs:
                                target_uri = ref.get('uri', '')
                                if target_uri.startswith('file://'):
                                    target_file = target_uri[7:]
                                    try:
                                        target_rel = str(Path(target_file).relative_to(path))
                                    except ValueError:
                                        target_rel = target_file
                                    cur.execute(
                                        """
                                        INSERT INTO symbol_references 
                                        (project_id, source_file, source_line, source_symbol, target_file, target_line, target_symbol, relation_type)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                        ON CONFLICT DO NOTHING
                                        """,
                                        (pid, rel_path, sym_line, sym['name'], target_rel, ref.get('line', 0), sym['name'], 'reference')
                                    )
                                    stats['references_indexed'] += 1
                        except Exception as ref_e:
                            logger.debug(f"Reference indexing failed for {sym['name']}: {ref_e}")
                
                # Index call hierarchy
                if include_call_hierarchy:
                    for sym in symbols:
                        if sym.get('type') in ('function', 'method'):
                            try:
                                sym_line = sym.get('line_start', 1)
                                lsp_line = sym_line - 1
                                # v68a: Column 4 offset for functions (after 'def ')
                                calls = await lsp_pool.call_hierarchy(file_path_str, lsp_line, 4, 'incoming')
                                for call in calls:
                                    caller_uri = call.get('uri', '')
                                    if caller_uri.startswith('file://'):
                                        caller_file = caller_uri[7:]
                                        try:
                                            caller_rel = str(Path(caller_file).relative_to(path))
                                        except ValueError:
                                            caller_rel = caller_file
                                        caller_name = call.get('name', 'unknown')
                                        caller_line = call.get('line', 0)
                                        cur.execute(
                                            """
                                            INSERT INTO symbol_references 
                                            (project_id, source_file, source_line, source_symbol, target_file, target_line, target_symbol, relation_type)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                            ON CONFLICT DO NOTHING
                                            """,
                                            (pid, caller_rel, caller_line, caller_name, rel_path, sym_line, sym['name'], 'call')
                                        )
                                        stats['calls_indexed'] += 1
                            except Exception as call_e:
                                logger.debug(f"Call hierarchy indexing failed for {sym['name']}: {call_e}")
            
            conn.commit()
            logger.info(f"sync_project: Indexed {stats['references_indexed']} refs, {stats['calls_indexed']} calls")
        
        # v43/v51: Upsert project_registry with project_root
        try:
            cur.execute(
                """
                INSERT INTO project_registry (project_id, project_path, project_root)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id) DO UPDATE SET
                    project_path = EXCLUDED.project_path,
                    project_root = EXCLUDED.project_root,
                    updated_at = NOW()
                """,
                (pid, str(path), str(path))
            )
            conn.commit()
            stats['project_registered'] = True
        except Exception as reg_error:
            logger.warning(f"v51: project_registry upsert failed: {reg_error}")
            stats['project_registered'] = False
        
        # v61: Auto-populate project understanding
        understanding_result = None
        if auto_understand:
            try:
                from pas.helpers.codebase import populate_project_understanding
                understanding_result = await populate_project_understanding(
                    project_id=pid,
                    project_path=str(path),
                    conn=conn
                )
            except Exception as e:
                logger.warning(f"sync_project: understanding population failed: {e}")
        
        return {
            "success": True,
            "project_id": pid,
            "project_path": str(path),
            **stats,
            "understanding_populated": understanding_result is not None and understanding_result.get('stored', False),
            "understanding_errors": understanding_result.get('errors', []) if understanding_result else [],
            "message": f"Synced {stats['files_added']} new, {stats['files_updated']} updated files"
        }
        
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        logger.error(f"sync_project failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v52: Auto-Sync File Watcher
# =============================================================================

# Module-level watcher singleton
_active_watcher = None
_active_handler = None


@mcp.tool()
async def start_auto_sync(
    project_path: str,
    debounce_seconds: float = 2.0
) -> dict[str, Any]:
    """
    Start watching project for file changes and auto-sync.
    
    Uses inotify to detect file saves and updates symbols in real-time.
    Files are synced after a debounce period (default 2s) to batch rapid saves.
    
    Args:
        project_path: Absolute path to project root
        debounce_seconds: Wait time after last change before syncing (default: 2.0)
        
    Returns:
        Status including directories watched
    """
    global _active_watcher, _active_handler
    
    from pas.lsp.watcher import InotifyWatcher, DebouncedSyncHandler
    from pas.lsp.lsp_pool import LspPool
    from pas.helpers.codebase import sync_file_incremental
    
    try:
        path = Path(project_path).resolve()
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {project_path}"}
        
        # Stop existing watcher if running
        if _active_watcher and _active_watcher.is_running:
            await _active_watcher.stop()
            logger.info("Stopped previous auto-sync watcher")
        
        # Initialize LSP pool
        lsp_pool = None
        try:
            lsp_pool = await LspPool.get(str(path))
        except Exception as e:
            logger.warning(f"LSP not available for auto-sync: {e}")
        
        project_id = _normalize_project_id(str(path))
        
        # Create sync callback
        async def sync_callback(file_path: str):
            result = await sync_file_incremental(
                file_path, 
                project_id, 
                str(path),
                lsp_pool
            )
            logger.info(f"Auto-sync: {result.get('file', file_path)} -> {result.get('symbols', 0)} symbols")
        
        # Create handler and watcher
        _active_handler = DebouncedSyncHandler(sync_callback, debounce_seconds)
        _active_watcher = InotifyWatcher(str(path))
        await _active_watcher.start(_active_handler.on_change)
        
        return {
            "success": True,
            "project_path": str(path),
            "project_id": project_id,
            "directories_watched": _active_watcher.watched_directories,
            "debounce_seconds": debounce_seconds,
            "lsp_available": lsp_pool is not None,
            "message": f"Watching {_active_watcher.watched_directories} directories for changes"
        }
        
    except Exception as e:
        logger.error(f"start_auto_sync failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def stop_auto_sync() -> dict[str, Any]:
    """
    Stop watching project for file changes.
    
    Returns:
        Status confirmation
    """
    global _active_watcher, _active_handler
    
    if not _active_watcher or not _active_watcher.is_running:
        return {"success": True, "message": "No active watcher to stop"}
    
    try:
        await _active_watcher.stop()
        _active_watcher = None
        _active_handler = None
        
        return {
            "success": True,
            "message": "Auto-sync watcher stopped"
        }
        
    except Exception as e:
        logger.error(f"stop_auto_sync failed: {e}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def query_codebase(
    query: str,
    project_id: str,
    top_k: int = 10,
    cross_project: bool = False,
    search_symbols: bool = True,
    search_files: bool = True
) -> dict[str, Any]:
    """
    Semantic search over indexed codebase.
    
    Searches file content and symbols using pgvector similarity.
    Uses project_id for isolation unless cross_project=True.
    
    Args:
        query: Natural language query (e.g., "finalize session logic")
        project_id: Project to search (required unless cross_project=True)
        top_k: Maximum results per category (default 10)
        cross_project: If True, search all projects (default False)
        search_symbols: Include symbol matches (default True)
        search_files: Include file content matches (default True)
    
    Returns:
        Matching files and symbols with scores
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate query embedding
        query_embedding = get_embedding(query)
        
        results = {
            "success": True,
            "query": query,
            "project_id": project_id if not cross_project else "ALL",
            "files": [],
            "symbols": []
        }
        
        # Search files
        if search_files:
            if cross_project:
                cur.execute(
                    """
                    SELECT project_id, file_path, language, line_count,
                           1 - (content_embedding <=> %s::vector) as similarity
                    FROM file_registry
                    WHERE content_embedding IS NOT NULL
                    ORDER BY content_embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                    SELECT project_id, file_path, language, line_count,
                           1 - (content_embedding <=> %s::vector) as similarity
                    FROM file_registry
                    WHERE project_id = %s AND content_embedding IS NOT NULL
                    ORDER BY content_embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, project_id, query_embedding, top_k)
                )
            
            for row in cur.fetchall():
                results["files"].append({
                    "project_id": row['project_id'],
                    "file_path": row['file_path'],
                    "language": row['language'],
                    "line_count": row['line_count'],
                    "similarity": float(row['similarity'])
                })
        
        # Search symbols
        if search_symbols:
            if cross_project:
                cur.execute(
                    """
                    SELECT fr.project_id, fr.file_path, 
                           fs.symbol_type, fs.symbol_name, fs.line_start, fs.line_end,
                           fs.signature, fs.docstring,
                           1 - (fs.embedding <=> %s::vector) as similarity
                    FROM file_symbols fs
                    JOIN file_registry fr ON fs.file_id = fr.id
                    WHERE fs.embedding IS NOT NULL
                    ORDER BY fs.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k)
                )
            else:
                cur.execute(
                    """
                    SELECT fr.project_id, fr.file_path, 
                           fs.symbol_type, fs.symbol_name, fs.line_start, fs.line_end,
                           fs.signature, fs.docstring,
                           1 - (fs.embedding <=> %s::vector) as similarity
                    FROM file_symbols fs
                    JOIN file_registry fr ON fs.file_id = fr.id
                    WHERE fr.project_id = %s AND fs.embedding IS NOT NULL
                    ORDER BY fs.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, project_id, query_embedding, top_k)
                )
            
            for row in cur.fetchall():
                results["symbols"].append({
                    "project_id": row['project_id'],
                    "file_path": row['file_path'],
                    "symbol_type": row['symbol_type'],
                    "symbol_name": row['symbol_name'],
                    "line_start": row['line_start'],
                    "line_end": row['line_end'],
                    "signature": row['signature'][:200] if row['signature'] else None,
                    "similarity": float(row['similarity'])
                })
        
        results["file_count"] = len(results["files"])
        results["symbol_count"] = len(results["symbols"])
        
        return results
        
    except Exception as e:
        logger.error(f"query_codebase failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)



@mcp.tool()
async def find_references(
    project_id: str,
    symbol_name: str,
    include_definitions: bool = False,
    session_id: str = None
) -> dict[str, Any]:
    """
    Find all references to a symbol in the codebase.

    v52 Phase 4b: Uses LSP via LspManager with symbol-to-position lookup.
    Falls back to Jedi/LSIF if LSP is unavailable.

    Args:
        project_id: Project identifier
        symbol_name: Symbol to find references for (partial match supported)
        include_definitions: If True, also include definitions

    Returns:
        List of locations where symbol is referenced
    """
    from pas.lsp.manager import LspManager
    import time as _time  # Debug timing
    
    _t0 = _time.time()
    logger.info(f"[DEBUG] find_references ENTRY: {project_id}, {symbol_name}")
    
    references = []
    source_used = "lsp"
    conn = None
    
    try:
        conn = get_db_connection()
        logger.info(f"[TIMING] DB connection: {_time.time()-_t0:.2f}s")
        cur = conn.cursor()
        
        # Get project root from registry
        project_root = fetch_project_root(project_id, cur)
        
        if not project_root:
            # Fallback: try to resolve from file_registry
            cur.execute(
                "SELECT file_path FROM file_registry WHERE project_id = %s LIMIT 10",
                (project_id,)
            )
            rows = cur.fetchall()
            rel_paths = [r['file_path'] for r in rows]
            project_root = resolve_project_root(rel_paths)
            
            if not project_root:
                safe_close_connection(conn)
                return {"success": False, "error": f"No project root found for '{project_id}'"}
        
        # Step 1: Use query_codebase to find symbol position (semantic lookup)
        symbol_position = None
        try:
            # Search symbols for the exact name
            query_embedding = get_embedding(symbol_name)
            logger.info(f"[TIMING] Embedding: {_time.time()-_t0:.2f}s")
            cur.execute(
                """
                SELECT fr.file_path, fs.line_start, fs.symbol_name
                FROM file_symbols fs
                JOIN file_registry fr ON fs.file_id = fr.id
                WHERE fr.project_id = %s AND fs.symbol_name ILIKE %s
                ORDER BY fs.embedding <=> %s::vector
                LIMIT 1
                """,
                (project_id, f"%{symbol_name}%", query_embedding)
            )
            row = cur.fetchone()
            if row:
                symbol_position = {
                    "file_path": str(project_root / row['file_path']),
                    "line": row['line_start'] - 1,  # LSP is 0-indexed
                    "col": 0  # Start of line
                }
        except Exception as e:
            logger.debug(f"Symbol lookup failed, will try name-based: {e}")
        
        # Step 2: Try LSP if we have position (uses warm pooled servers)
        if symbol_position:
            try:
                from pas.lsp import get_pool
                logger.info(f"[LSP] Attempting LSP with position: {symbol_position}")
                logger.info(f"[TIMING] Pre-LSP: {_time.time()-_t0:.2f}s")
                lsp_pool = await get_pool(str(project_root))
                logger.info(f"[TIMING] LSP pool started: {lsp_pool._started}")
                logger.info(f"[TIMING] LSP pool: {_time.time()-_t0:.2f}s")
                
                # First try find_references directly from symbol position
                lsp_refs = await lsp_pool.find_references(
                    symbol_position["file_path"],
                    symbol_position["line"],
                    symbol_position["col"]
                )
                
                # If no refs, the position might be a reference not a definition
                # Jump to definition first, then try refs from there
                if not lsp_refs:
                    logger.info("[LSP] No refs from position, trying go_to_definition first")
                    defn = await lsp_pool.find_definition(
                        symbol_position["file_path"],
                        symbol_position["line"],
                        symbol_position["col"]
                    )
                    if defn and defn.get("uri"):
                        # Extract definition location and retry refs
                        def_file = defn.get("uri", "").replace("file://", "")
                        def_line = defn.get("line", 0)
                        logger.info(f"[LSP] Definition found at {def_file}:{def_line}, retrying refs")
                        lsp_refs = await lsp_pool.find_references(def_file, def_line, 0)
                
                logger.info(f"[LSP] find_references returned: {len(lsp_refs) if lsp_refs else 0} refs")
                
                if lsp_refs:
                    for ref in lsp_refs:
                        references.append({
                            "file": str(Path(ref.get("uri", ref.get("file", ""))).relative_to(project_root)) if ref.get("uri") or ref.get("file") else "",
                            "line": ref.get("range", {}).get("start", {}).get("line", ref.get("line", 0)) + 1,
                            "symbol": symbol_name,
                            "relation": "definition" if ref.get("is_definition") else "reference"
                        })
                    source_used = "lsp"
                else:
                    logger.info("[LSP] No refs from LSP - ensure project is synced")
                    source_used = "db"
            except Exception as e:
                logger.warning(f"LSP find_references failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                source_used = "db"
        else:
            logger.info(f"[LSP] No symbol_position found - will search DB")
            source_used = "db"
        
        # v53: DB-first - no Jedi/LSIF fallback
        # If LSP didn't find refs, return empty (project needs sync_project)

        
        # Deduplicate
        references = deduplicate_references(references, include_definitions)
        
    except Exception as e:
        logger.error(f"find_references failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        safe_close_connection(conn)
    
    # Phase 9: Log lookup to session context if session_id provided
    if session_id:
        from pas.helpers.lsp_enrichment import log_lsp_lookup
        log_lsp_lookup(session_id, symbol_name, "find_references", len(references))
    
    if not references:
        return {
            "success": True, "project_id": project_id, "symbol_name": symbol_name,
            "references": [], "count": 0, "source": source_used,
            "note": "No references found. Ensure project is synced with sync_project."
        }
    
    return {"success": True, "project_id": project_id, "symbol_name": symbol_name,
            "references": references[:100], "count": len(references), "source": source_used}




@mcp.tool()
async def go_to_definition(
    project_id: str,
    file_path: str,
    line: int,
    column: int = 0
) -> dict[str, Any]:
    """
    Jump to the definition of a symbol at a given location.

    Args:
        project_id: Project identifier
        file_path: File containing the reference
        line: Line number (0-indexed)
        column: Column number (0-indexed, optional)

    Returns:
        Definition location if found
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT target_file, target_line, target_symbol, symbol_qualified_name, symbol_type
            FROM symbol_references
            WHERE project_id = %s AND source_file LIKE %s AND source_line = %s
              AND relation_type = 'definition'
            LIMIT 1
            """,
            (project_id, f"%{file_path}%", line)
        )
        
        row = cur.fetchone()
        
        if not row:
            return {"success": True, "project_id": project_id,
                    "location": {"file": file_path, "line": line},
                    "definition": None, "note": "No definition found in LSIF index."}
        
        return {"success": True, "project_id": project_id,
                "location": {"file": file_path, "line": line},
                "definition": {"file": row['target_file'], "line": row['target_line'],
                               "symbol": row['target_symbol']}}
        
    except Exception as e:
        logger.error(f"go_to_definition failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


@mcp.tool()
async def call_hierarchy(
    project_id: str,
    symbol_name: str,
    direction: str = "incoming",
    max_depth: int = 3,
    session_id: str = None
) -> dict[str, Any]:
    """
    Build call hierarchy for a symbol.

    Args:
        project_id: Project identifier
        symbol_name: Symbol to analyze
        direction: "incoming" (callers) or "outgoing" (callees)
        max_depth: Maximum depth to traverse (default 3)

    Returns:
        Hierarchical tree of callers or callees
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if direction not in ("incoming", "outgoing"):
            return {"success": False, "error": "direction must be 'incoming' or 'outgoing'"}
        
        if direction == "incoming":
            cur.execute(
                """
                SELECT source_file, source_line, source_symbol,
                       target_file, target_line, target_symbol
                FROM symbol_references
                WHERE project_id = %s AND target_symbol ILIKE %s AND relation_type = 'call'
                ORDER BY source_file, source_line LIMIT 50
                """,
                (project_id, f"%{symbol_name}%")
            )
        else:
            cur.execute(
                """
                SELECT source_file, source_line, source_symbol,
                       target_file, target_line, target_symbol
                FROM symbol_references
                WHERE project_id = %s AND source_symbol ILIKE %s AND relation_type = 'call'
                ORDER BY target_file, target_line LIMIT 50
                """,
                (project_id, f"%{symbol_name}%")
            )
        
        rows = cur.fetchall()
        
        if not rows:
            return {"success": True, "project_id": project_id, "symbol_name": symbol_name,
                    "direction": direction, "hierarchy": [], "count": 0,
                    "note": "No call hierarchy found. Run sync_project first."}
        
        hierarchy = [{"caller": {"file": r['source_file'], "line": r['source_line'], "symbol": r['source_symbol']},
                      "callee": {"file": r['target_file'], "line": r['target_line'], "symbol": r['target_symbol']}} 
                     for r in rows]
        
        # Phase 9: Log lookup to session context if session_id provided
        if session_id:
            from pas.helpers.lsp_enrichment import log_lsp_lookup
            log_lsp_lookup(session_id, symbol_name, "call_hierarchy", len(hierarchy))
        
        return {"success": True, "project_id": project_id, "symbol_name": symbol_name,
                "direction": direction, "hierarchy": hierarchy, "count": len(hierarchy)}
        
    except Exception as e:
        logger.error(f"call_hierarchy failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# v52: Plan Validation Tool
# =============================================================================

@mcp.tool()
async def validate_plan(
    session_id: str,
    plan_text: str,
    skip_validation: bool = False,
    lsp_impact: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Validate implementation plan addresses all critiques.
    
    OPT-OUT: Enabled by default. Set skip_validation=True only with explicit
    user approval (will be logged for outcome correlation).
    
    Args:
        session_id: The reasoning session UUID
        plan_text: Full text of implementation plan to validate
        skip_validation: If True, bypass validation (logged)
        lsp_impact: Optional result from get_lsp_impact() to validate scope
        
    Returns:
        Validation result with missing/addressed critiques and scope warnings
    """
    if skip_validation:
        logger.warning(f"validate_plan SKIPPED for session {session_id}")
        return {
            "valid": True,
            "skipped": True,
            "warning": "Validation bypassed - outcomes may be affected"
        }
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get critique checklist
        from pas.helpers.finalize import build_critique_checklist
        checklist = build_critique_checklist(cur, session_id)
        
        plan_lower = plan_text.lower()
        
        # =====================================================================
        # v73 Phase 3: LSP scope validation (run before early returns)
        # =====================================================================
        scope_warnings = []
        if lsp_impact and lsp_impact.get("lsp_available"):
            callers_outside = lsp_impact.get("callers_outside_scope", [])
            for caller in callers_outside:
                # Check if caller file is mentioned in plan
                from pathlib import Path
                caller_name = Path(caller).name.lower()
                if caller_name not in plan_lower:
                    scope_warnings.append({
                        "type": "scope_miss",
                        "file": caller,
                        "message": f"File '{caller_name}' uses symbols from scope but not in plan"
                    })
        
        # Build LSP section check result
        lsp_section_check = None
        if lsp_impact:
            lsp_section_check = {
                "has_lsp_section": False,
                "lsp_required": True,
                "warning": None
            }
            
            # Check for LSP Impact Analysis section (case-insensitive)
            lsp_section_patterns = [
                "lsp impact analysis",
                "lsp impact",
                "impact analysis",
                "## lsp",
                "### lsp"
            ]
            for pattern in lsp_section_patterns:
                if pattern in plan_lower:
                    lsp_section_check["has_lsp_section"] = True
                    break
            
            # If LSP data provided but no section, add warning
            if not lsp_section_check["has_lsp_section"]:
                lsp_section_check["warning"] = "Plan should include 'LSP Impact Analysis' section when lsp_impact data is available"
            
            # If lsp_impact has external callers not addressed, mark as warning
            if scope_warnings:
                lsp_section_check["external_callers_not_addressed"] = len(scope_warnings)
        
        # Early return if no critiques - but still include LSP info
        if not checklist:
            return {
                "valid": True,
                "message": "No critiques to validate against",
                "checklist_count": 0,
                "scope_warnings": scope_warnings if scope_warnings else None,
                "lsp_section_check": lsp_section_check
            }
        
        # plan_lower already computed above
        addressed = []
        missing = []
        
        for item in checklist:
            # Extract key terms from critique text (words > 3 chars)
            terms = [t.strip().lower() for t in item["text"].split() if len(t) > 3]
            # Check if at least 2 key terms appear in plan
            matches = sum(1 for t in terms if t in plan_lower)
            threshold = min(2, len(terms))
            
            if matches >= threshold:
                addressed.append(item["id"])
            else:
                missing.append({
                    "id": item["id"],
                    "type": item["type"],
                    "text": item["text"],
                    "severity": item["severity"]
                })
        
        coverage = len(addressed) / len(checklist) if checklist else 1.0
        has_high_severity_missing = any(m["severity"] == "high" for m in missing)
        
        # Determine overall validity
        is_valid = len(missing) == 0 or not has_high_severity_missing
        
        return {
            "valid": is_valid,
            "coverage": round(coverage, 2),
            "total_critiques": len(checklist),
            "addressed_count": len(addressed),
            "addressed_ids": addressed,
            "missing_count": len(missing),
            "missing": missing,
            "scope_warnings": scope_warnings if scope_warnings else None,
            "lsp_section_check": lsp_section_check if lsp_impact else None,
            "recommendation": "Address missing HIGH severity items before proceeding" if has_high_severity_missing else None
        }
        
    except Exception as e:
        logger.error(f"validate_plan failed: {e}")
        return {"valid": False, "error": str(e)}
    finally:
        if 'conn' in locals():
            safe_close_connection(conn)


# =============================================================================
# Phase 6: Project Governance Tools
# =============================================================================


@mcp.tool()
async def get_project_governance(
    project_id: str
) -> dict[str, Any]:
    """
    Get full governance hierarchy for a project.
    
    Returns vision, roadmap phases, and their artifacts.
    
    Args:
        project_id: Project identifier
        
    Returns:
        Nested hierarchy: vision -> phases -> artifacts
    """
    try:
        hierarchy = get_governance_hierarchy(project_id)
        return {
            "success": True,
            **hierarchy
        }
    except Exception as e:
        logger.error(f"get_project_governance failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def store_governance_artifact(
    project_id: str,
    name: str,
    content: str,
    artifact_type: str = "implementation_plan",
    session_id: str | None = None,
    roadmap_phase_id: str | None = None,
    source_verbatim_log_id: str | None = None,
    tags: str | None = None
) -> dict[str, Any]:
    """
    Store a versioned artifact with governance linkage.
    
    Args:
        project_id: Project identifier
        name: Artifact name (used for versioning - same name = new version)
        content: Full artifact content
        artifact_type: roadmap, implementation_plan, walkthrough, vision, other
        session_id: Optional PAS session that generated this
        roadmap_phase_id: Optional link to roadmap phase
        source_verbatim_log_id: Optional link to verbatim prompt in conversation_log
        tags: Comma-separated tags (e.g., "feature,backend,v71")
        
    Returns:
        Created artifact with version number
    """
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        
        result = store_artifact(
            project_id=project_id,
            name=name,
            content=content,
            artifact_type=artifact_type,
            session_id=session_id,
            roadmap_phase_id=roadmap_phase_id,
            source_verbatim_log_id=source_verbatim_log_id,
            tags=tag_list
        )
        
        return {
            "success": True,
            **result,
            "message": f"Artifact '{name}' v{result['version']} stored"
        }
    except Exception as e:
        logger.error(f"store_governance_artifact failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_artifact_versions(
    project_id: str,
    name: str
) -> dict[str, Any]:
    """
    Get version history for an artifact.
    
    Args:
        project_id: Project identifier
        name: Artifact name
        
    Returns:
        List of versions (newest first)
    """
    try:
        versions = get_artifact_versions(project_id, name)
        return {
            "success": True,
            "project_id": project_id,
            "name": name,
            "versions": versions,
            "version_count": len(versions)
        }
    except Exception as e:
        logger.error(f"list_artifact_versions failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_artifacts(
    project_id: str,
    query: str | None = None,
    tags: str | None = None,
    limit: int = 10
) -> dict[str, Any]:
    """
    Search artifacts by semantic query or tags.
    
    Args:
        project_id: Project identifier
        query: Natural language query (semantic search)
        tags: Comma-separated tags to filter by (tag overlap search)
        limit: Maximum results
        
    Returns:
        Matching artifacts
    """
    try:
        results = []
        search_type = None
        
        if query:
            results = search_artifacts_semantic(project_id, query, limit)
            search_type = "semantic"
        elif tags:
            tag_list = [t.strip() for t in tags.split(",")]
            results = search_artifacts_by_tag(project_id, tag_list, limit)
            search_type = "tags"
        else:
            return {"success": False, "error": "Either query or tags required"}
        
        return {
            "success": True,
            "project_id": project_id,
            "search_type": search_type,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"search_artifacts failed: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def create_governance_phase(
    project_id: str,
    phase_name: str,
    description: str | None = None,
    status: str = "planned"
) -> dict[str, Any]:
    """
    Create a new roadmap phase for a project.
    
    Args:
        project_id: Project identifier
        phase_name: Name of the phase
        description: Phase description
        status: planned, active, complete, blocked
        
    Returns:
        Created phase with sequence number
    """
    try:
        # Ensure vision exists
        get_or_create_project_vision(project_id)
        
        phase = create_roadmap_phase(
            project_id=project_id,
            phase_name=phase_name,
            description=description,
            status=status
        )
        
        return {
            "success": True,
            **phase,
            "message": f"Phase '{phase_name}' created at sequence {phase['sequence']}"
        }
    except Exception as e:
        logger.error(f"create_governance_phase failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Entry Point
# =============================================================================



def main():
    """Run the MCP server."""
    logger.info("Starting PAS (Scientific Reasoning) MCP Server...")
    
    # Pre-warm embedding model for instant tool calls (avoids 10s cold start)
    try:
        logger.info("Pre-warming embedding model...")
        from pas.utils import get_embedding_model
        get_embedding_model()
        logger.info("Embedding model ready")
    except Exception as e:
        logger.warning(f"Embedding pre-warm failed (will load on first use): {e}")
    
    mcp.run()


if __name__ == "__main__":
    main()
