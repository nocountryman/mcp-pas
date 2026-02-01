"""
Finalize Session Helper Functions

Extracted from server.py to reduce finalize_session complexity.
Each function handles a distinct concern of the finalization process.
"""

from typing import Any, Optional
import logging
import json

logger = logging.getLogger("pas-server")


# =============================================================================
# v51: Effort-Benefit ROI Helpers
# =============================================================================

def classify_roi_quadrant(effort: int, benefit: int) -> dict:
    """
    Classify effort/benefit into quadrant for prioritization.
    
    Args:
        effort: 1=low, 2=medium, 3=high
        benefit: 1=low, 2=medium, 3=high
        
    Returns:
        Quadrant classification dict with emoji and action guidance
    """
    if benefit >= 2 and effort == 1:
        return {"quadrant": "quick_win", "emoji": "⭐", "action": "Prioritize"}
    elif benefit >= 2 and effort >= 2:
        return {"quadrant": "strategic", "emoji": "📋", "action": "Needs planning"}
    elif benefit == 1 and effort == 1:
        return {"quadrant": "low_priority", "emoji": "⏸️", "action": "Can defer"}
    else:  # benefit=1, effort>=2
        return {"quadrant": "avoid", "emoji": "⚠️", "action": "Reconsider"}


def detect_roi_gaming(candidates: list[dict]) -> tuple[Optional[str], bool]:
    """
    Detect if all candidates have identical effort/benefit (potential gaming).
    
    Goodhart's Law: When a measure becomes a target, it ceases to be a good measure.
    If all hypotheses have the same E/B values, the agent may be gaming.
    
    Args:
        candidates: List of processed candidate dicts with optional 'roi' key
        
    Returns:
        Tuple of (warning message or None, gaming_detected bool)
    """
    roi_values = []
    for c in candidates:
        roi = c.get("roi")
        if roi and isinstance(roi, dict):
            effort = roi.get("effort")
            benefit = roi.get("benefit")
            if effort is not None and benefit is not None:
                roi_values.append((effort, benefit))
    
    # Need at least 2 ROI values to detect gaming
    if len(roi_values) < 2:
        return None, False
    
    # Check if all values are identical
    if len(set(roi_values)) == 1:
        e, b = roi_values[0]
        return f"⚠️ Potential ROI gaming: All {len(roi_values)} hypotheses have identical effort={e}/benefit={b}", True
    
    return None, False


def build_roi_analysis(
    processed: list[dict],
    recommendation: dict
) -> Optional[dict]:
    """
    Build ROI analysis for finalize_session response.
    
    Args:
        processed: List of processed candidates
        recommendation: Winning recommendation
        
    Returns:
        ROI analysis dict or None if no ROI data
    """
    # v51: ROI is stored in metadata JSONB, not top-level
    metadata = recommendation.get("metadata") or {}
    roi_data = metadata.get("roi") if isinstance(metadata, dict) else None
    if not roi_data or not isinstance(roi_data, dict):
        return None
    
    effort = roi_data.get("effort")
    benefit = roi_data.get("benefit")
    
    if effort is None or benefit is None:
        return None
    
    quadrant = classify_roi_quadrant(effort, benefit)
    gaming_warning, gaming_detected = detect_roi_gaming(processed)
    
    return {
        "top_candidate": {
            "effort": effort,
            "benefit": benefit,
            **quadrant
        },
        "gaming_warning": gaming_warning,
        "gaming_detected": gaming_detected
    }


# =============================================================================
# v52: Critique Enforcement Helpers
# =============================================================================

def build_critique_checklist(
    cur,
    session_id: str
) -> list[dict]:
    """
    v52: Build checklist of all critiques for plan validation.
    
    Collects major_flaws, minor_flaws from thought_nodes.metadata,
    and gaps from sequential_analysis for opt-out validation.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        List of critique items with unique IDs
    """
    # Fetch all nodes with critique data
    cur.execute(
        """
        SELECT id, content, metadata->'critique' as critique
        FROM thought_nodes
        WHERE session_id = %s 
          AND metadata ? 'critique'
        ORDER BY created_at
        """,
        (session_id,)
    )
    rows = cur.fetchall()
    
    checklist = []
    for row in rows:
        critique = row["critique"]
        if not critique:
            continue
        node_id = str(row["id"])[:8]  # Short ID for readability
        
        # Add major flaws
        for j, flaw in enumerate(critique.get("major_flaws", [])):
            if flaw:
                checklist.append({
                    "id": f"MAJOR_{node_id}_{j}",
                    "type": "major_flaw",
                    "text": flaw,
                    "source_node": str(row["id"]),
                    "severity": "high"
                })
        
        # Add minor flaws
        for j, flaw in enumerate(critique.get("minor_flaws", [])):
            if flaw:
                checklist.append({
                    "id": f"MINOR_{node_id}_{j}",
                    "type": "minor_flaw", 
                    "text": flaw,
                    "source_node": str(row["id"]),
                    "severity": "low"
                })
    
    # Add sequential gaps from session context
    cur.execute(
        """
        SELECT context->>'sequential_analysis' as gaps
        FROM reasoning_sessions
        WHERE id = %s
        """,
        (session_id,)
    )
    session = cur.fetchone()
    if session and session.get("gaps"):
        gaps_data = session["gaps"]
        if isinstance(gaps_data, str):
            gaps_data = json.loads(gaps_data)
        for i, gap_entry in enumerate(gaps_data or []):
            for j, gap in enumerate(gap_entry.get("gaps", [])):
                if gap:
                    checklist.append({
                        "id": f"GAP_{i}_{j}",
                        "type": "gap",
                        "text": gap,
                        "source_node": gap_entry.get("node_id"),
                        "severity": "medium"
                    })
    
    return checklist


def check_sequential_gate(
    cur,
    session_id: str,
    skip_sequential_analysis: bool
) -> Optional[dict[str, Any]]:
    """
    Check if sequential gap analysis was performed.
    
    Returns None if check passes, or an error dict if analysis is required.
    
    Args:
        cur: Database cursor
        session_id: The reasoning session UUID
        skip_sequential_analysis: If True, bypass the check
        
    Returns:
        None if check passes, or error dict if sequential analysis required
    """
    if skip_sequential_analysis:
        return None
    
    cur.execute(
        "SELECT COUNT(*) as cnt FROM thought_nodes WHERE session_id = %s AND metadata->>'sequential_analyzed' = 'true'",
        (session_id,)
    )
    result = cur.fetchone()
    
    if not result or result['cnt'] == 0:
        return {
            "success": False,
            "sequential_analysis_required": True,
            "error": "Sequential gap analysis not done. Call prepare_sequential_analysis + store_sequential_analysis first, or pass skip_sequential_analysis=True to bypass.",
            "next_step": f"mcp_pas-server_prepare_sequential_analysis(session_id='{session_id}', top_n=3)"
        }
    
    return None


def compute_quality_gate(
    winner_score: float,
    runner_up_score: float,
    min_score_threshold: float,
    min_gap_threshold: float,
    skip_quality_gate: bool
) -> tuple[dict[str, Any], bool]:
    """
    Compute quality gate status (score and gap thresholds).
    
    Args:
        winner_score: Adjusted score of winning hypothesis
        runner_up_score: Adjusted score of runner-up (0.0 if none)
        min_score_threshold: Minimum required score
        min_gap_threshold: Minimum required gap between winner and runner-up
        skip_quality_gate: If True, gate is considered passed regardless
        
    Returns:
        Tuple of (quality_gate dict, quality_gate_enforced bool)
    """
    gap = winner_score - runner_up_score if runner_up_score else 1.0
    
    quality_gate = {
        "score": round(winner_score, 4),
        "score_threshold": min_score_threshold,
        "score_ok": winner_score >= min_score_threshold,
        "gap": round(gap, 4),
        "gap_threshold": min_gap_threshold,
        "gap_ok": gap >= min_gap_threshold,
        "passed": winner_score >= min_score_threshold and gap >= min_gap_threshold
    }
    
    quality_gate_enforced = quality_gate["passed"] or skip_quality_gate
    
    return quality_gate, quality_gate_enforced


def build_score_improvement_suggestions(
    quality_gate: dict[str, Any],
    winner_score: float,
    gap: float,
    min_score_threshold: float,
    min_gap_threshold: float
) -> list[dict[str, Any]]:
    """
    Generate suggestions for improving quality gate metrics.
    
    Args:
        quality_gate: Quality gate status dict
        winner_score: Current winning score
        gap: Current gap between winner and runner-up
        min_score_threshold: Threshold for score
        min_gap_threshold: Threshold for gap
        
    Returns:
        List of improvement suggestion dicts
    """
    suggestions = []
    
    if not quality_gate["passed"]:
        if not quality_gate["score_ok"]:
            suggestions.append({
                "lever": "score",
                "current": round(winner_score, 3),
                "threshold": min_score_threshold,
                "action": "Expand deeper with higher confidence (0.9+) or address critique penalties"
            })
        if not quality_gate["gap_ok"]:
            suggestions.append({
                "lever": "gap",
                "current": round(gap, 3),
                "threshold": min_gap_threshold,
                "action": "Explore more diverse alternatives to differentiate the best solution"
            })
    
    return suggestions


def apply_unverified_prefix(
    cur,
    conn,
    session_id: str,
    recommendation_content: str,
    quality_gate_enforced: bool,
    winner_score: float,
    gap: float
) -> str:
    """
    Apply [UNVERIFIED] prefix if quality gate not enforced.
    
    Args:
        cur: Database cursor
        conn: Database connection
        session_id: Session ID
        recommendation_content: Original recommendation content
        quality_gate_enforced: Whether quality gate passed or was skipped
        winner_score: Winning score for logging
        gap: Gap for logging
        
    Returns:
        Modified recommendation content (with [UNVERIFIED] prefix if applicable)
    """
    if quality_gate_enforced:
        return recommendation_content
    
    prefixed_content = f"[UNVERIFIED] {recommendation_content}"
    
    # Log enforcement violation to DB
    try:
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = COALESCE(context, '') || E'\n[ENFORCEMENT VIOLATION at ' || NOW()::text || ']'
            WHERE id = %s
        """, (session_id,))
        conn.commit()
        logger.warning(f"v33: Quality gate not enforced for session {session_id} (score={winner_score:.3f}, gap={gap:.3f})")
    except Exception as log_err:
        logger.error(f"Failed to log enforcement violation: {log_err}")
    
    return prefixed_content


def surface_warnings_and_tags(
    search_relevant_failures_fn,
    generate_suggested_tags_fn,
    session: dict,
    recommendation: dict,
    implementation_checklist: list[str],
    cur,
    conn,
    session_id: str
) -> tuple[list[dict], list[str], list[str]]:
    """
    Surface past failure warnings and suggest session tags.
    
    Args:
        search_relevant_failures_fn: Function to search for relevant failures (injected)
        generate_suggested_tags_fn: Function to generate tags (injected)
        session: Session dict with goal
        recommendation: Recommendation dict with content
        implementation_checklist: Checklist to prepend warnings to
        cur: Database cursor
        conn: Database connection
        session_id: Session ID for DB operations
        
    Returns:
        Tuple of (warnings_surfaced, suggested_tags, updated_checklist)
    """
    import json as json_module
    
    warnings_surfaced: list[dict[str, Any]] = []
    suggested_tags: list[str] = []
    winning_content = recommendation.get("content", "")
    
    # Surface warnings
    try:
        goal_warnings = search_relevant_failures_fn(session["goal"])
        rec_warnings = search_relevant_failures_fn(recommendation["content"])
        
        seen_patterns: set[str] = set()
        for w in goal_warnings + rec_warnings:
            pattern = w.get("pattern", "")
            if pattern and pattern not in seen_patterns:
                seen_patterns.add(pattern)
                warnings_surfaced.append(w)
        
        for w in reversed(warnings_surfaced):
            pattern = w.get("pattern", "UNKNOWN")
            warning_text = w.get("warning", "Review this warning")
            implementation_checklist.insert(0, f"[ ] ⚠️ {pattern}: {warning_text}")
        
        if warnings_surfaced:
            logger.info(f"v32b: Surfaced {len(warnings_surfaced)} warning(s) in finalize_session")
    except Exception as e:
        logger.warning(f"v32b warning surfacing failed: {e}")
    
    # Suggest tags
    try:
        suggested_tags = generate_suggested_tags_fn(session["goal"], winning_content)
        
        if suggested_tags:
            logger.info(f"v26: Suggested tags for session {session_id}: {suggested_tags}")
            try:
                cur.execute(
                    "UPDATE reasoning_sessions SET suggested_tags = %s WHERE id = %s",
                    (json_module.dumps(suggested_tags), session_id)
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"v34 suggested_tags DB write failed: {e}")
    except Exception as e:
        logger.warning(f"v26 tag suggestion failed: {e}")
    
    return warnings_surfaced, suggested_tags, implementation_checklist


def identify_pending_critiques(
    config: dict,
    processed: list[dict],
    recommendation_score: float,
    session_context: dict,
    cur,
    conn,
    session_id: str
) -> tuple[list[dict], Optional[str]]:
    """
    v48: Identify mid-range candidates for parallel critique window.
    
    Args:
        config: PAS config dict
        processed: List of processed candidates
        recommendation_score: Score of winning candidate
        session_context: Session context dict
        cur: Database cursor
        conn: Database connection
        session_id: Session ID
        
    Returns:
        Tuple of (pending_critiques list, explore_alternatives_prompt or None)
    """
    import json as json_module
    
    pending_critiques: list[dict[str, Any]] = []
    explore_alternatives_prompt = None
    
    try:
        pc_config = config.get("parallel_critique", {})
        if pc_config.get("enabled", False) and len(processed) > 1:
            ratio = pc_config.get("critique_ratio", 0.7)
            max_cand = pc_config.get("max_candidates", 3)
            threshold = recommendation_score * ratio
            
            for p in processed[1:]:  # Skip winner
                if p["final_score"] >= threshold and len(pending_critiques) < max_cand:
                    pending_critiques.append({
                        "node_id": p["node_id"],
                        "content": p["content"][:200],
                        "score": round(p["final_score"], 4)
                    })
            
            if pending_critiques:
                updated_context = session_context or {}
                updated_context["pending_critiques"] = pending_critiques
                cur.execute(
                    "UPDATE reasoning_sessions SET context = %s WHERE id = %s",
                    (json_module.dumps(updated_context), session_id)
                )
                conn.commit()
                explore_alternatives_prompt = f"Mid-range alternatives available. Call explore_alternatives(session_id='{session_id}') to review."
                logger.info(f"v48: Queued {len(pending_critiques)} alternatives for exploration in session {session_id}")
    except Exception as e:
        logger.warning(f"v48: Parallel critique identification failed: {e}")
    
    return pending_critiques, explore_alternatives_prompt


def build_implementation_checklist(
    winning_scope: str
) -> list[str]:
    """
    Build implementation checklist from winning hypothesis scope.
    
    Args:
        winning_scope: Declared scope of winning hypothesis
        
    Returns:
        Implementation checklist items
    """
    checklist = []
    
    if winning_scope:
        scope_items = [s.strip() for s in winning_scope.split(",") if s.strip()]
        for item in scope_items:
            checklist.append(f"[ ] Modify: {item}")
    
    if not checklist:
        checklist.append("[ ] Implement the recommended approach")
    
    checklist.extend([
        "[ ] Write/update tests",
        "[ ] Verify changes work as expected"
    ])
    
    return checklist


def query_past_failures(
    cur,
    session: dict,
    limit: int = 3
) -> list[dict]:
    """
    Query past failures with similar goals via semantic similarity.
    
    Args:
        cur: Database cursor
        session: Session dict with goal_embedding
        limit: Max number of failures to return
        
    Returns:
        List of past failure dicts with goal and reason
    """
    past_failures = []
    
    try:
        if session.get("goal_embedding"):
            cur.execute("""
                SELECT o.notes, o.failure_reason, s.goal 
                FROM outcome_records o 
                JOIN reasoning_sessions s ON o.session_id = s.id
                WHERE o.outcome != 'success' 
                AND (o.notes IS NOT NULL OR o.failure_reason IS NOT NULL)
                AND s.goal_embedding IS NOT NULL
                AND 1 - (s.goal_embedding <=> %s) > 0.7
                LIMIT %s
            """, (session["goal_embedding"], limit))
            for r in cur.fetchall():
                reason = r["failure_reason"] or r["notes"]
                if reason:
                    past_failures.append({"goal": r["goal"][:100], "reason": reason})
    except Exception as e:
        logger.warning(f"v15b past_failures lookup failed: {e}")
    
    return past_failures


def query_calibration_warning(
    cur
) -> Optional[str]:
    """
    Query confidence calibration stats and build warning if needed.
    
    Args:
        cur: Database cursor
        
    Returns:
        Calibration warning string or None
    """
    calibration_warning = None
    
    try:
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE o.outcome = 'success') as successes,
                COUNT(*) as total
            FROM outcome_records o
            JOIN thought_nodes t ON t.session_id = o.session_id
            WHERE t.likelihood >= 0.8
            AND o.created_at > NOW() - INTERVAL '30 days'
        """)
        cal = cur.fetchone()
        if cal and cal["total"] >= 5:
            success_rate = cal["successes"] / cal["total"]
            if success_rate < 0.7:
                calibration_warning = f"⚠️ Calibration: High-confidence (≥0.8) hypotheses succeed only {success_rate:.0%} of the time in recent sessions"
    except Exception as e:
        logger.warning(f"v16b.1 calibration query failed: {e}")
    
    return calibration_warning


def build_deep_critique_requests(
    cur,
    processed: list[dict],
    count: int = 2
) -> Optional[dict[str, Any]]:
    """
    Build deep critique requests for top candidates.
    
    Args:
        cur: Database cursor
        processed: List of processed candidates
        count: Number of top candidates to build requests for
        
    Returns:
        Critique response dict if deep_critique enabled, else None
    """
    if not processed:
        return None
    
    critique_requests = []
    for p in processed[:count]:
        cur.execute(
            """
            SELECT sl.law_name, sl.definition
            FROM scientific_laws sl
            JOIN thought_nodes tn ON sl.id = ANY(tn.supporting_laws)
            WHERE tn.id = %s
            LIMIT 3
            """,
            (p["node_id"],)
        )
        laws = cur.fetchall()
        critique_requests.append({
            "node_id": p["node_id"],
            "content": p["content"],
            "supporting_laws": [{"name": l["law_name"], "definition": l["definition"]} for l in laws]
        })
    
    return {
        "success": True,
        "needs_critique": True,
        "message": "Generate critiques for these nodes, then call store_critique for each, then call finalize_session again.",
        "critique_requests": critique_requests
    }


def check_complementarity(
    processed: list[dict],
    top_n: int,
    detect_fn,
    extract_goals_fn,
    synthesize_fn,
    session_id: str
) -> tuple[list[dict], Optional[dict[str, Any]]]:
    """
    Check if top candidates are complementary (address different goals).
    
    Args:
        processed: List of processed candidates
        top_n: Number of top candidates to check
        detect_fn: detect_complementarity function
        extract_goals_fn: extract_addressed_goals function
        synthesize_fn: synthesize_hypothesis_text function
        session_id: Session ID for logging
        
    Returns:
        Tuple of (updated processed list, complementarity_result or None)
    """
    complementarity_result = None
    
    if len(processed) < 2:
        return processed, None
    
    comp_candidates = [
        {"content": p["content"], "scope": p.get("declared_scope", "")}
        for p in processed[:top_n]
    ]
    is_complementary, covered_goals, avg_overlap = detect_fn(comp_candidates, threshold=0.5)
    
    if is_complementary:
        logger.info(f"v40: Complementarity detected in session {session_id}: {covered_goals}")
        for p in processed:
            goals = extract_goals_fn(p["content"], p.get("declared_scope", ""))
            p["addressed_goals"] = goals
        
        complementarity_result = {
            "detected": True,
            "covered_goals": covered_goals,
            "avg_overlap": round(avg_overlap, 3),
            "synthesis_suggestion": "Top candidates are complementary, not competitive. Consider using synthesize_hypotheses() to create a unified approach.",
            "synthesis_prompt": synthesize_fn(comp_candidates)
        }
    
    return processed, complementarity_result


def apply_uct_and_compute_decision(
    processed: list[dict],
    apply_uct_fn,
    compute_quality_fn,
    session_id: str
) -> tuple[dict, Optional[dict], str, str, bool, float]:
    """
    Apply UCT tiebreaking and compute decision quality.
    
    Args:
        processed: List of processed candidates (will be modified)
        apply_uct_fn: _apply_uct_tiebreaking helper function
        compute_quality_fn: _compute_decision_quality helper function
        session_id: Session ID
        
    Returns:
        Tuple of (recommendation, runner_up, decision_quality, gap_analysis, uct_applied, gap)
    """
    recommendation = processed[0]
    runner_up = processed[1] if len(processed) > 1 else None
    
    uct_applied = False
    gap = 0.0
    
    if runner_up:
        gap = recommendation["final_score"] - runner_up["final_score"]
        
        should_swap, uct_applied = apply_uct_fn(
            recommendation["final_score"], recommendation["depth"],
            runner_up["final_score"], runner_up["depth"]
        )
        if should_swap:
            recommendation, runner_up = runner_up, recommendation
            processed[0], processed[1] = processed[1], processed[0]
            gap = recommendation["final_score"] - runner_up["final_score"]
        
        rec_conf = recommendation.get("confidence", 0.5)
        run_conf = runner_up.get("confidence", 0.5)
        decision_quality, gap_analysis = compute_quality_fn(
            gap, rec_conf, run_conf, uct_applied
        )
    else:
        decision_quality = "medium"
        gap_analysis = "Only one candidate available."
    
    return recommendation, runner_up, decision_quality, gap_analysis, uct_applied, gap


def build_next_step_guidance(
    decision_quality: str,
    recommendation: dict,
    session_id: str
) -> Optional[str]:
    """
    Build next_step guidance based on decision quality and depth.
    
    Args:
        decision_quality: "low", "medium", or "high"
        recommendation: Winning recommendation dict
        session_id: Session ID
        
    Returns:
        Next step guidance string or None
    """
    winner_depth = recommendation.get("depth", 2)
    
    if decision_quality == "low" and winner_depth < 4:
        return f"Decision is close. Expand the winning hypothesis deeper. Call prepare_expansion(session_id='{session_id}', parent_node_id='{recommendation['node_id']}')"
    elif decision_quality == "medium" and winner_depth < 3:
        return f"Consider refining the recommendation. Call prepare_expansion(session_id='{session_id}', parent_node_id='{recommendation['node_id']}')"
    
    return None


def build_scope_guidance(
    session: dict,
    recommendation: dict,
    past_failures: list[dict],
    calibration_warning: Optional[str]
) -> dict[str, Any]:
    """
    Construct scope_guidance for domain-agnostic follow-up suggestions.
    
    Args:
        session: Session dict with goal
        recommendation: Winning recommendation dict
        past_failures: List of past failure dicts
        calibration_warning: Calibration warning string or None
        
    Returns:
        Scope guidance dict
    """
    winning_scope = recommendation.get("declared_scope", "") or ""
    return {
        "context": {
            "goal": session["goal"],
            "scope": winning_scope,
            "recommendation": recommendation["content"][:200]
        },
        "prompt": "What validation or follow-up steps are needed for this specific context? If this is pure reasoning with no action items, respond 'No follow-up needed'.",
        "past_failures": past_failures,
        "calibration_warning": calibration_warning
    }


def build_exhaustive_prompt(
    recommendation: dict,
    session: dict,
    should_generate: bool
) -> Optional[dict[str, str]]:
    """
    Build exhaustive check prompt for layer-by-layer gap analysis.
    
    Args:
        recommendation: Winning recommendation dict
        session: Session dict with goal
        should_generate: Whether to generate the prompt
        
    Returns:
        Exhaustive prompt dict or None
    """
    if not should_generate or not recommendation:
        return None
    
    logger.info("v32: Returning exhaustive_prompt for agent to process")
    
    return {
        "prompt": f"""Analyze this recommendation for what it does NOT address:

Recommendation: {recommendation["content"]}
Goal: {session["goal"]}

For EACH of these layers, identify what's covered vs missing:
1. CODE STRUCTURE: What code changes are needed? Any not mentioned?
2. DEPENDENCIES: What packages/systems are assumed but not stated?
3. DATA FLOW: What data moves where? Any gaps in the data path?
4. INTERFACES: What APIs/contracts are affected? Any missing?
5. WORKFLOWS: What user/system flows change? Any not addressed?

Return ONLY a valid JSON object:
{{
  "gaps": [
    {{"layer": "...", "covered": "...", "missing": "..."}}
  ],
  "critical_gaps": ["gaps that could cause failure"],
  "overall_coverage": 0.8
}}

Focus on OMISSIONS, not flaws.""",
        "system": "You are a gap analyst. Find what's MISSING, not what's wrong. Return only valid JSON.",
        "instructions": "Process this prompt and review the gaps. If critical gaps found, consider deepening your analysis."
    }


def get_context_summary(session: dict) -> Optional[dict[str, str]]:
    """
    Extract interview context summary from session.
    
    Args:
        session: Session dict with context
        
    Returns:
        Context summary dict or None
    """
    context = session.get("context") or {}
    interview = context.get("interview", {})
    
    if interview.get("answer_history"):
        return {
            h["question_id"]: h["answer"] 
            for h in interview["answer_history"]
        }
    
    return None


def fetch_sibling_data(
    cur,
    session_id: str
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    """
    Fetch sibling counts and law diversity data for penalty calculations.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        Tuple of (sibling_counts dict, law_diversity dict)
    """
    # Get sibling counts per parent path
    cur.execute(
        """
        SELECT SUBPATH(path, 0, NLEVEL(path) - 1) as parent_path,
               COUNT(*) as sibling_count
        FROM thought_nodes
        WHERE session_id = %s AND node_type = 'hypothesis'
        GROUP BY SUBPATH(path, 0, NLEVEL(path) - 1)
        """,
        (session_id,)
    )
    sibling_counts = {str(row["parent_path"]): row["sibling_count"] for row in cur.fetchall()}
    
    # Get unique law counts per parent path for monoculture detection
    cur.execute(
        """
        SELECT SUBPATH(path, 0, NLEVEL(path) - 1) as parent_path,
               COUNT(DISTINCT supporting_laws[1]) as unique_laws,
               COUNT(*) as total_siblings
        FROM thought_nodes
        WHERE session_id = %s AND node_type = 'hypothesis' 
              AND supporting_laws IS NOT NULL AND array_length(supporting_laws, 1) > 0
        GROUP BY SUBPATH(path, 0, NLEVEL(path) - 1)
        """,
        (session_id,)
    )
    law_diversity = {str(row["parent_path"]): (row["unique_laws"], row["total_siblings"]) 
                    for row in cur.fetchall()}
    
    return sibling_counts, law_diversity


def process_candidates(
    cur,
    candidates: list[dict],
    sibling_counts: dict[str, int],
    law_diversity: dict[str, tuple[int, int]],
    apply_penalties_fn,
    build_processed_fn
) -> list[dict]:
    """
    Process candidates by applying penalties and calculating rollout scores.
    
    Args:
        cur: Database cursor
        candidates: List of candidate nodes
        sibling_counts: Sibling count data
        law_diversity: Law diversity data
        apply_penalties_fn: _apply_heuristic_penalties function
        build_processed_fn: _build_processed_candidate function
        
    Returns:
        List of processed candidates sorted by final score
    """
    processed = []
    
    for node in candidates:
        adjusted_score, penalties = apply_penalties_fn(node, sibling_counts, law_diversity)
        
        # Calculate law-grounded rollout score
        supporting_law_ids = node["supporting_laws"] or []
        rollout_score = 0.5  # Default neutral
        if supporting_law_ids:
            cur.execute(
                """
                SELECT AVG(scientific_weight) as avg_weight
                FROM scientific_laws
                WHERE id = ANY(%s) AND scientific_weight IS NOT NULL
                """,
                (supporting_law_ids,)
            )
            rollout_result = cur.fetchone()
            if rollout_result and rollout_result["avg_weight"]:
                rollout_score = float(rollout_result["avg_weight"])
        
        processed.append(build_processed_fn(node, adjusted_score, penalties, rollout_score))
    
    # Sort by final score
    processed.sort(key=lambda x: x["final_score"], reverse=True)
    
    return processed


def apply_config_defaults(
    min_score_threshold: Optional[float],
    min_gap_threshold: Optional[float],
    config: dict
) -> tuple[float, float]:
    """
    Apply config defaults if thresholds not provided.
    
    Args:
        min_score_threshold: Provided threshold or None
        min_gap_threshold: Provided threshold or None
        config: PAS_CONFIG dict
        
    Returns:
        Tuple of (min_score_threshold, min_gap_threshold) with defaults applied
    """
    if min_score_threshold is None:
        min_score_threshold = config["quality_gate"]["min_score_threshold"]
    if min_gap_threshold is None:
        min_gap_threshold = config["quality_gate"]["min_gap_threshold"]
    return min_score_threshold, min_gap_threshold


def fetch_session(cur, session_id: str) -> Optional[dict]:
    """
    Fetch session info from database.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        Session dict or None
    """
    cur.execute(
        "SELECT goal, context FROM reasoning_sessions WHERE id = %s",
        (session_id,)
    )
    return cur.fetchone()


def fetch_candidates(cur, session_id: str, top_n: int) -> list[dict]:
    """
    Fetch candidate hypotheses ordered by posterior score.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        top_n: Maximum candidates to fetch
        
    Returns:
        List of candidate dicts
    """
    cur.execute(
        """
        SELECT id, path, content, node_type, depth,
               prior_score, likelihood, posterior_score,
               supporting_laws, metadata
        FROM thought_nodes
        WHERE session_id = %s AND node_type = 'hypothesis'
        ORDER BY posterior_score DESC
        LIMIT %s
        """,
        (session_id, top_n)
    )
    return cur.fetchall()
