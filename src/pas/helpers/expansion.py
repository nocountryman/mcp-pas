"""
Phase 2 & 3 Refactor: store_expansion and prepare_expansion helper functions.

Phase 2: Extracted store_expansion helpers to reduce CC from 73 to 18.
Phase 3: Extracted prepare_expansion helpers to reduce CC from 59 to ≤15.
"""
import uuid
import logging
import re
import json
from typing import Any, Optional

logger = logging.getLogger("pas_server")


def build_hypotheses_list(
    h1_text: str,
    h1_confidence: float,
    h1_scope: Optional[str],
    h2_text: Optional[str],
    h2_confidence: Optional[float],
    h2_scope: Optional[str],
    h3_text: Optional[str],
    h3_confidence: Optional[float],
    h3_scope: Optional[str],
    # v51: Effort-Benefit scoring
    h1_effort: Optional[int] = None,
    h1_benefit: Optional[int] = None,
    h2_effort: Optional[int] = None,
    h2_benefit: Optional[int] = None,
    h3_effort: Optional[int] = None,
    h3_benefit: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Build hypotheses list from flattened parameters.
    
    Args:
        h1_text-h3_text: Hypothesis texts
        h1_confidence-h3_confidence: Confidence values
        h1_scope-h3_scope: Declared scopes
        h1_effort-h3_effort: Optional effort estimates (1=low, 2=medium, 3=high)
        h1_benefit-h3_benefit: Optional benefit estimates (1=low, 2=medium, 3=high)
        
    Returns:
        List of hypothesis dicts with optional ROI data
    """
    def _build_roi(effort: Optional[int], benefit: Optional[int]) -> Optional[dict]:
        """Build ROI dict if both values provided and valid."""
        if effort is not None and benefit is not None:
            # Clamp to 1-3 range
            e = min(max(int(effort), 1), 3)
            b = min(max(int(benefit), 1), 3)
            return {"effort": e, "benefit": b}
        return None
    
    hypotheses = []
    if h1_text:
        h = {"hypothesis": h1_text, "confidence": h1_confidence or 0.5, "scope": h1_scope}
        roi = _build_roi(h1_effort, h1_benefit)
        if roi:
            h["roi"] = roi
        hypotheses.append(h)
    if h2_text:
        h = {"hypothesis": h2_text, "confidence": h2_confidence or 0.5, "scope": h2_scope}
        roi = _build_roi(h2_effort, h2_benefit)
        if roi:
            h["roi"] = roi
        hypotheses.append(h)
    if h3_text:
        h = {"hypothesis": h3_text, "confidence": h3_confidence or 0.5, "scope": h3_scope}
        roi = _build_roi(h3_effort, h3_benefit)
        if roi:
            h["roi"] = roi
        hypotheses.append(h)
    return hypotheses



def verify_active_session(cur, session_id: str) -> Optional[dict]:
    """
    Verify session exists and is active.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        Error dict if invalid, None if valid
    """
    cur.execute("SELECT goal FROM reasoning_sessions WHERE id = %s AND state = 'active'", (session_id,))
    if not cur.fetchone():
        return {"success": False, "error": "Session not found or not active"}
    return None


def resolve_parent_path(
    cur,
    session_id: str,
    parent_node_id: Optional[str],
    get_embedding_fn
) -> tuple[str, str, Optional[dict]]:
    """
    Resolve parent path, creating root node if needed.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        parent_node_id: Parent node ID or None
        get_embedding_fn: Embedding function
        
    Returns:
        Tuple of (parent_path, parent_node_id, error_dict or None)
    """
    if parent_node_id:
        cur.execute("SELECT path FROM thought_nodes WHERE id = %s", (parent_node_id,))
        parent = cur.fetchone()
        if not parent:
            return "", "", {"success": False, "error": "Parent node not found"}
        return parent["path"], parent_node_id, None
    
    # Create or get root
    cur.execute("SELECT id, path FROM thought_nodes WHERE session_id = %s AND path = 'root'", (session_id,))
    root = cur.fetchone()
    if root:
        return "root", str(root["id"]), None
    
    # Create root
    cur.execute("SELECT goal FROM reasoning_sessions WHERE id = %s", (session_id,))
    goal = cur.fetchone()["goal"]
    root_id = str(uuid.uuid4())
    root_emb = get_embedding_fn(goal)
    cur.execute(
        "INSERT INTO thought_nodes (id, session_id, path, content, node_type, prior_score, likelihood, embedding) VALUES (%s, %s, 'root', %s, 'root', 0.5, 0.5, %s)",
        (root_id, session_id, goal, root_emb)
    )
    return "root", root_id, None


def match_laws_and_compute_prior(
    cur,
    hypothesis_text: str,
    hyp_emb,
    compute_ensemble_fn
) -> tuple[float, list[str], Optional[str]]:
    """
    Match laws and compute ensemble prior for a hypothesis.
    
    Args:
        cur: Database cursor
        hypothesis_text: Hypothesis text
        hyp_emb: Hypothesis embedding
        compute_ensemble_fn: _compute_ensemble_prior function
        
    Returns:
        Tuple of (prior, supporting_law_ids, law_name)
    """
    MIN_SIMILARITY = 0.2
    
    # Query top-3 similar laws
    cur.execute(
        """
        SELECT id, law_name, definition, scientific_weight, selection_count, success_count,
               1 - (embedding <=> %s::vector) as similarity
        FROM scientific_laws WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector LIMIT 3
        """,
        (hyp_emb, hyp_emb)
    )
    laws = cur.fetchall()
    
    # Filter by similarity threshold
    matching_laws = [l for l in laws if l["similarity"] >= MIN_SIMILARITY]
    
    if matching_laws:
        prior, supporting_law_ids, law_name = compute_ensemble_fn(
            matching_laws, hypothesis_text
        )
        
        # Track law selection
        for law in matching_laws:
            cur.execute(
                "UPDATE scientific_laws SET selection_count = selection_count + 1 WHERE id = %s",
                (law["id"],)
            )
        return prior, supporting_law_ids, law_name
    
    return 0.5, [], None


def log_conversation_source(
    cur,
    session_id: str,
    source_text: Optional[str],
    user_id: Optional[str],
    first_node_id: str,
    get_embedding_fn
) -> Optional[str]:
    """
    Log source_text to conversation_log for semantic search.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        source_text: Source text to log
        user_id: User ID
        first_node_id: First created node ID
        get_embedding_fn: Embedding function
        
    Returns:
        Conversation log ID or None
    """
    if not source_text:
        return None
    
    try:
        source_embedding = get_embedding_fn(source_text[:2000])
        
        cur.execute(
            """
            INSERT INTO conversation_log (session_id, thought_node_id, user_id, log_type, raw_text, embedding)
            VALUES (%s, %s, %s, 'user_input', %s, %s)
            RETURNING id
            """,
            (session_id, first_node_id, user_id, source_text, source_embedding)
        )
        log_entry = cur.fetchone()
        conversation_log_id = str(log_entry["id"]) if log_entry else None
        logger.info(f"v25: Created conversation_log entry {conversation_log_id} for source_text ({len(source_text)} chars)")
        return conversation_log_id
    except Exception as log_err:
        logger.warning(f"v25: Failed to create conversation_log entry: {log_err}")
        return None


def compute_workflow_nudges(
    created_nodes: list[dict],
    is_revision: bool,
    revises_node_id: Optional[str]
) -> dict[str, Any]:
    """
    Compute workflow nudges (next_step, confidence, revision).
    
    Args:
        created_nodes: List of created node dicts
        is_revision: Whether this is a revision
        revises_node_id: Node being revised
        
    Returns:
        Dict with next_step, confidence_nudge, revision_info, revision_nudge
    """
    result = {
        "next_step": None,
        "confidence_nudge": None,
        "revision_info": None,
        "revision_nudge": None
    }
    
    if not created_nodes:
        return result
    
    top_node = max(created_nodes, key=lambda n: n.get("posterior_score") or 0)
    result["next_step"] = f"Challenge your top hypothesis. Call prepare_critique(node_id='{top_node['node_id']}')"
    
    # Confidence nudge
    avg_confidence = sum(n.get("likelihood", 0.5) for n in created_nodes) / len(created_nodes)
    if avg_confidence < 0.65:
        result["confidence_nudge"] = f"Low confidence detected (avg: {avg_confidence:.2f}). Consider: (1) expand deeper on uncertain hypothesis, (2) add alternative perspectives, (3) gather more context before deciding."
    
    # Revision tracking
    if is_revision:
        result["revision_info"] = {
            "is_revision": True,
            "revises_node_id": revises_node_id,
            "message": "Revision noted. Original hypothesis preserved for comparison."
        }
        result["revision_nudge"] = f"Revision recorded. Consider critiquing the revised hypothesis. Call prepare_critique(node_id='{top_node['node_id']}')"
    
    return result


def query_scope_failures(
    cur,
    conn,
    created_nodes: list[dict]
) -> list[dict]:
    """
    Query historical failures by scope patterns (v21).
    
    Args:
        cur: Database cursor
        conn: Database connection
        created_nodes: List of created node dicts
        
    Returns:
        List of scope warning dicts
    """
    scope_warnings = []
    try:
        # Collect all declared scopes
        all_scopes = []
        for node in created_nodes:
            scope = node.get("declared_scope")
            if scope:
                all_scopes.extend([s.strip() for s in scope.split(',') if s.strip()])
        
        if not all_scopes:
            return []
        
        # Build scope patterns
        scope_patterns = ['%' + s.split(':')[0] + '%' for s in all_scopes if s]
        
        cur.execute("""
            SELECT DISTINCT o.failure_reason, s.goal, t.declared_scope
            FROM outcome_records o
            JOIN reasoning_sessions s ON o.session_id = s.id
            JOIN thought_nodes t ON t.session_id = s.id
            WHERE o.outcome = 'failure'
            AND o.failure_reason IS NOT NULL
            AND t.declared_scope IS NOT NULL
            AND t.declared_scope ILIKE ANY(%s)
            ORDER BY o.created_at DESC
            LIMIT 3
        """, (scope_patterns,))
        
        related_failures = cur.fetchall()
        for r in related_failures:
            scope_warnings.append({
                "past_goal": r["goal"][:60] + "..." if len(r["goal"]) > 60 else r["goal"],
                "past_scope": r["declared_scope"][:80] if r["declared_scope"] else None,
                "failure_reason": r["failure_reason"][:120] + "..." if len(r["failure_reason"]) > 120 else r["failure_reason"]
            })
        
        if scope_warnings:
            logger.info(f"v21: Found {len(scope_warnings)} scope-related historical failures")
    except Exception as e:
        conn.rollback()
        logger.warning(f"v21: Scope-based failure matching failed: {e}")
    
    return scope_warnings


def run_preflight_checks(
    cur,
    conn,
    session_id: str,
    skip_preflight: bool,
    created_nodes: list[dict]
) -> tuple[list[dict], bool]:
    """
    Run v41 preflight enforcement checks.
    
    Args:
        cur: Database cursor
        conn: Database connection
        session_id: Session ID
        skip_preflight: Whether to skip checks
        created_nodes: List of created node dicts
        
    Returns:
        Tuple of (preflight_warnings, preflight_bypassed)
    """
    preflight_warnings = []
    preflight_bypassed = False
    
    try:
        from pas.helpers.preflight import check_preflight_conditions, log_tool_call
        
        if skip_preflight:
            log_tool_call(cur, session_id, "store_expansion_bypass", {"reason": "skip_preflight=True"})
            preflight_bypassed = True
            logger.info("v41: Preflight check bypassed via skip_preflight=True")
        else:
            # Get preflight context from session metadata
            cur.execute(
                "SELECT call_metadata FROM session_call_log WHERE session_id = %s AND tool_name = 'prepare_expansion' ORDER BY created_at DESC LIMIT 1",
                (session_id,)
            )
            prep_call = cur.fetchone()
            
            if prep_call and prep_call.get("call_metadata"):
                import json
                metadata = prep_call["call_metadata"] if isinstance(prep_call["call_metadata"], dict) else json.loads(prep_call["call_metadata"])
                
                preflight_warnings = check_preflight_conditions(
                    cur,
                    session_id,
                    has_suggested_lookups=bool(metadata.get("has_suggested_lookups")),
                    schema_check_required=metadata.get("schema_check_required", False),
                    has_failure_warnings=bool(metadata.get("has_failure_warnings")),
                    has_project_id=bool(metadata.get("has_project_id"))
                )
                
                if preflight_warnings:
                    logger.warning(f"v41: Preflight warnings: {[w['type'] for w in preflight_warnings]}")
        
        # Log this store_expansion call
        log_tool_call(cur, session_id, "store_expansion", {
            "node_count": len(created_nodes),
            "preflight_bypassed": preflight_bypassed,
            "preflight_warning_count": len(preflight_warnings)
        })
        conn.commit()
    except Exception as e:
        logger.warning(f"v41: Preflight check failed (non-fatal): {e}")
    
    return preflight_warnings, preflight_bypassed


def surface_scope_failures(
    cur,
    conn,
    session_id: str,
    created_nodes: list[dict],
    search_relevant_failures_fn
) -> list[dict]:
    """
    Search and surface scope-matched failures (v42b).
    
    Args:
        cur: Database cursor
        conn: Database connection
        session_id: Session ID
        created_nodes: List of created node dicts
        search_relevant_failures_fn: _search_relevant_failures function
        
    Returns:
        List of scope failure warning dicts
    """
    scope_failure_warnings = []
    try:
        # Get schema_check_required from session call log
        schema_check = False
        try:
            cur.execute(
                "SELECT call_metadata FROM session_call_log WHERE session_id = %s AND tool_name = 'prepare_expansion' ORDER BY created_at DESC LIMIT 1",
                (session_id,)
            )
            prep_call = cur.fetchone()
            if prep_call and prep_call.get("call_metadata"):
                import json
                metadata = prep_call["call_metadata"] if isinstance(prep_call["call_metadata"], dict) else json.loads(prep_call["call_metadata"])
                schema_check = metadata.get("schema_check_required", False)
        except:
            pass
        
        # Get already-surfaced warning IDs for deduplication
        cur.execute("SELECT COALESCE(surfaced_warning_ids, '{}') FROM reasoning_sessions WHERE id = %s", (session_id,))
        surfaced_row = cur.fetchone()
        exclude_ids = set(surfaced_row[0]) if surfaced_row and surfaced_row[0] else set()
        
        # Search failures for each hypothesis scope
        new_surfaced_ids = []
        for node in created_nodes:
            scope = node.get("declared_scope", "")
            if scope:
                failures = search_relevant_failures_fn(
                    scope,
                    context_type="scope",
                    schema_check_required=schema_check,
                    exclude_ids=exclude_ids
                )
                for f in failures:
                    f["hypothesis_path"] = node["path"]
                    f["matched_scope"] = scope
                    scope_failure_warnings.append(f)
                    if f.get("id"):
                        new_surfaced_ids.append(f["id"])
        
        # Update surfaced_warning_ids for future deduplication
        if new_surfaced_ids:
            cur.execute(
                "UPDATE reasoning_sessions SET surfaced_warning_ids = COALESCE(surfaced_warning_ids, '{}') || %s WHERE id = %s",
                (new_surfaced_ids, session_id)
            )
            conn.commit()
        
        if scope_failure_warnings:
            logger.info(f"v42b: Surfaced {len(scope_failure_warnings)} scope-matched failure(s)")
    except Exception as e:
        logger.warning(f"v42b: Scope failure surfacing failed (non-fatal): {e}")
    
    return scope_failure_warnings


# =============================================================================
# Phase 3: prepare_expansion helper functions
# =============================================================================

def validate_session_and_get_parent(
    cur,
    session_id: str,
    parent_node_id: Optional[str]
) -> tuple[Optional[dict], Optional[str], Optional[str], Optional[str], Optional[dict]]:
    """
    Validate session exists/active and resolve parent node.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        parent_node_id: Parent node ID or None
        
    Returns:
        Tuple of (session, parent_content, parent_path, resolved_parent_id, error_dict)
        If error, returns (None, None, None, None, error_dict)
    """
    # Get session
    cur.execute("SELECT id, goal, state FROM reasoning_sessions WHERE id = %s", (session_id,))
    session = cur.fetchone()
    if not session:
        return None, None, None, None, {"success": False, "error": f"Session {session_id} not found"}
    
    if session["state"] != "active":
        return None, None, None, None, {"success": False, "error": f"Session is {session['state']}, not active"}
    
    # Get parent node or use goal as root
    if parent_node_id:
        cur.execute(
            "SELECT id, path, content FROM thought_nodes WHERE id = %s AND session_id = %s",
            (parent_node_id, session_id)
        )
        parent = cur.fetchone()
        if not parent:
            return None, None, None, None, {"success": False, "error": f"Parent node {parent_node_id} not found"}
        return session, parent["content"], parent["path"], parent_node_id, None
    
    # No parent_node_id - use goal as root
    parent_content = session["goal"]
    parent_path = None
    resolved_parent_id = None
    
    # Check for existing root
    cur.execute(
        "SELECT id, path FROM thought_nodes WHERE session_id = %s AND path = 'root'",
        (session_id,)
    )
    existing_root = cur.fetchone()
    if existing_root:
        resolved_parent_id = str(existing_root["id"])
        parent_path = "root"
    
    return session, parent_content, parent_path, resolved_parent_id, None


def fetch_and_boost_laws(
    cur,
    parent_content: str,
    session_id: str,
    get_embedding_fn
) -> tuple[list[dict], list[str]]:
    """
    Fetch relevant laws and apply v22 trait-based boosting.
    
    Args:
        cur: Database cursor
        parent_content: Parent node content for embedding
        session_id: Session ID for trait lookup
        get_embedding_fn: Embedding function
        
    Returns:
        Tuple of (laws_list, boosted_law_names)
    """
    # Find relevant laws for this context
    embedding = get_embedding_fn(parent_content)
    cur.execute(
        """
        SELECT id, law_name, definition, scientific_weight,
               1 - (embedding <=> %s::vector) as similarity
        FROM scientific_laws WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector LIMIT 3
        """,
        (embedding, embedding)
    )
    
    laws = [{"id": r["id"], "law_name": r["law_name"], "definition": r["definition"],
             "weight": float(r["scientific_weight"]), "similarity": round(float(r["similarity"]), 4)}
            for r in cur.fetchall()]
    
    # v22: Get session context for traits
    cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
    context_row = cur.fetchone()
    session_context = context_row["context"] if context_row and context_row["context"] else {}
    
    # Combine persistent and latent traits
    all_trait_names = []
    persistent_traits = session_context.get("persistent_traits", [])
    latent_traits = session_context.get("latent_traits", [])
    
    for t in persistent_traits:
        if t.get("trait"):
            all_trait_names.append(t["trait"])
    for t in latent_traits:
        if t.get("trait"):
            all_trait_names.append(t["trait"])
    
    # Apply boosting if we have traits
    boosted_laws = []
    if all_trait_names and laws:
        cur.execute("""
            SELECT law_id, SUM(boost_factor) as total_boost
            FROM trait_law_correlations
            WHERE trait_name = ANY(%s)
            GROUP BY law_id
        """, (all_trait_names,))
        
        boosts = {row["law_id"]: row["total_boost"] for row in cur.fetchall()}
        
        for law in laws:
            if law["id"] in boosts:
                # Cap boost at 50% to prevent runaway
                boost = min(boosts[law["id"]], 0.5)
                original_weight = law["weight"]
                law["weight"] = round(law["weight"] * (1 + boost), 4)
                law["boosted_by"] = boost
                boosted_laws.append(law["law_name"])
                logger.debug(f"v22: Boosted {law['law_name']} from {original_weight} to {law['weight']}")
    
    return laws, boosted_laws


def build_trait_instructions(
    cur,
    session_id: str
) -> tuple[str, list[dict]]:
    """
    Build v21 trait-aware instructions.
    
    Args:
        cur: Database cursor
        session_id: Session ID
        
    Returns:
        Tuple of (instructions_string, latent_traits_list)
    """
    base_instructions = "Consider: What is requested? What files/modules might be affected? For each hypothesis, declare SCOPE as specific file paths. Optionally prefix with layer if helpful: [API] routes.py, [DB] models.py, [tests] test_auth.py. Generate 3 hypotheses with confidence (0.0-1.0). Call store_expansion(h1_text=..., h1_confidence=..., h1_scope='[layer] file1.py, file2.py', ...)."
    
    # Get latent_traits from session context
    cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
    context_row = cur.fetchone()
    session_context = context_row["context"] if context_row and context_row["context"] else {}
    latent_traits = session_context.get("latent_traits", [])
    
    # Trait-to-guidance mapping
    TRAIT_GUIDANCE = {
        "RISK_AVERSE": "User prefers SAFE, REVERSIBLE approaches. Prioritize hypotheses with clear rollback strategies and minimal blast radius.",
        "MINIMALIST": "User values SIMPLICITY. Favor minimal implementations that solve the core problem without unnecessary complexity.",
        "CONTROL_ORIENTED": "User wants EXPLICIT CONTROL. Prefer solutions that give user visibility and manual override options.",
        "AUTOMATION_TRUSTING": "User trusts AUTOMATION. Feel free to propose AI/automated solutions without excessive manual oversight.",
        "ACCESSIBILITY_FOCUSED": "User prioritizes ACCESSIBILITY. Consider beginner-friendly approaches with good documentation.",
        "PRAGMATIST": "User seeks BALANCED solutions. Blend theory with practicality, avoid extremes.",
        "EXPLICIT_CONTROL": "User wants TRANSPARENCY. Prefer explicit configuration over magic/convention.",
        "SPEED_FOCUSED": "User prioritizes PERFORMANCE. Consider efficiency and fast execution in hypotheses.",
        "SAFETY_CONSCIOUS": "User values ERROR PREVENTION. Include validation and safety checks in proposed solutions.",
        "AUTONOMY_FOCUSED": "User wants FREEDOM and flexibility. Avoid overly prescriptive or locked-down solutions.",
    }
    
    # Build trait guidance section
    trait_guidance = ""
    if latent_traits:
        guidance_lines = []
        for trait_info in latent_traits:
            trait_name = trait_info.get("trait")
            confidence = trait_info.get("confidence", 0)
            if trait_name in TRAIT_GUIDANCE and confidence >= 0.7:
                guidance_lines.append(f"- {TRAIT_GUIDANCE[trait_name]}")
        
        if guidance_lines:
            trait_guidance = "\n\nUSER PREFERENCES DETECTED:\n" + "\n".join(guidance_lines)
            logger.info(f"v21 Phase 3: Added {len(guidance_lines)} trait guidance(s) to instructions")
    
    return base_instructions + trait_guidance, latent_traits


def surface_past_failures(
    search_relevant_failures_fn,
    parent_content: str
) -> list[dict]:
    """
    Surface v31 past failure warnings based on parent content.
    
    Args:
        search_relevant_failures_fn: _search_relevant_failures function
        parent_content: Content to search for relevant failures
        
    Returns:
        List of past failure warning dicts
    """
    past_failure_warnings = search_relevant_failures_fn(parent_content)
    if past_failure_warnings:
        logger.info(f"v31: Surfaced {len(past_failure_warnings)} failure warning(s)")
    return past_failure_warnings


def extract_symbol_suggestions(
    cur,
    project_id: Optional[str],
    goal: str,
    parent_content: str
) -> tuple[list[dict], str]:
    """
    Extract v38c symbol suggestions from goal/parent text.
    
    Args:
        cur: Database cursor
        project_id: Project ID for symbol lookup
        goal: Session goal
        parent_content: Parent content
        
    Returns:
        Tuple of (suggested_lookups, instruction_addition)
    """
    if not project_id:
        return [], ""
    
    try:
        # Extract snake_case and CamelCase patterns from text
        text_to_search = f"{goal} {parent_content}"
        
        # Pattern for Python identifiers
        snake_pattern = r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b'
        camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        
        candidates = set()
        candidates.update(re.findall(snake_pattern, text_to_search))
        candidates.update(re.findall(camel_pattern, text_to_search))
        
        # Remove common false positives
        false_positives = {'should_be', 'will_be', 'may_be', 'can_be', 'must_be'}
        candidates = candidates - false_positives
        
        if not candidates:
            return [], ""
        
        # Validate against file_symbols table
        cur.execute(
            """
            SELECT fs.symbol_name, fr.file_path, fs.line_start
            FROM file_symbols fs
            JOIN file_registry fr ON fs.file_id = fr.id
            WHERE fr.project_id = %s AND fs.symbol_name = ANY(%s)
            ORDER BY fs.symbol_name, fr.file_path
            LIMIT 20
            """,
            (project_id, list(candidates))
        )
        
        symbol_rows = cur.fetchall()
        if not symbol_rows:
            return [], ""
        
        suggested_lookups = []
        for row in symbol_rows:
            suggested_lookups.append({
                "symbol": row["symbol_name"],
                "file": row["file_path"],
                "line": row["line_start"],
                "match_type": "exact"
            })
        
        # Build instruction addition
        symbol_names = [s["symbol"] for s in suggested_lookups[:3]]
        instruction_addition = f"\n\n⚠️ SUGGESTED: Before generating hypotheses, call find_references(project_id='{project_id}', symbol_name='...') for: {', '.join(symbol_names)}. This will show all callers/usages to inform your scope."
        
        logger.info(f"v38c: Found {len(suggested_lookups)} symbol suggestions for project {project_id}")
        return suggested_lookups, instruction_addition
        
    except Exception as e:
        logger.warning(f"v38c: Symbol suggestion failed (non-fatal): {e}")
        return [], ""


def search_related_modules(
    cur,
    project_id: Optional[str],
    goal: str,
    get_embedding_fn
) -> tuple[list[dict], str]:
    """
    Search v42a related modules using semantic search.
    
    Args:
        cur: Database cursor
        project_id: Project ID
        goal: Session goal
        get_embedding_fn: Embedding function
        
    Returns:
        Tuple of (related_modules, instruction_addition)
    """
    if not project_id:
        return [], ""
    
    try:
        goal_embedding = get_embedding_fn(goal[:1000])
        cur.execute(
            """
            SELECT file_path, purpose_cache,
                   1 - (content_embedding <=> %s::vector) as similarity
            FROM file_registry
            WHERE project_id = %s
              AND content_embedding IS NOT NULL
            ORDER BY content_embedding <=> %s::vector
            LIMIT 5
            """,
            (goal_embedding, project_id, goal_embedding)
        )
        related_rows = cur.fetchall()
        
        if not related_rows:
            return [], ""
        
        related_modules = []
        for row in related_rows:
            module_info = {
                "file": row["file_path"],
                "similarity": round(row["similarity"], 3) if row["similarity"] else None
            }
            # Include purpose if cached
            if row.get("purpose_cache"):
                try:
                    cache = row["purpose_cache"] if isinstance(row["purpose_cache"], dict) else json.loads(row["purpose_cache"])
                    if cache.get("module_purpose"):
                        module_info["purpose"] = cache["module_purpose"][:100]
                except:
                    pass
            related_modules.append(module_info)
        
        instruction_addition = f"\n\n📂 EXISTING MODULES FOUND: Review these {len(related_modules)} related files before hypothesizing: " + ", ".join([m['file'] for m in related_modules[:3]])
        logger.info(f"v42a: Found {len(related_modules)} related modules for goal")
        
        return related_modules, instruction_addition
        
    except Exception as e:
        logger.warning(f"v42a: Auto semantic search failed (non-fatal): {e}")
        return [], ""


def fetch_project_grounding(
    cur,
    project_id: Optional[str]
) -> tuple[Optional[dict], str]:
    """
    Fetch v43 project purpose grounding.
    
    Args:
        cur: Database cursor
        project_id: Project ID
        
    Returns:
        Tuple of (project_grounding_dict, instruction_addition)
    """
    if not project_id:
        return None, ""
    
    try:
        cur.execute(
            """
            SELECT purpose_hierarchy, detected_domain
            FROM project_registry
            WHERE project_id = %s
            """,
            (project_id,)
        )
        project_purpose_row = cur.fetchone()
        
        if not project_purpose_row or not project_purpose_row["purpose_hierarchy"]:
            return None, ""
        
        purpose = project_purpose_row["purpose_hierarchy"]
        project_grounding = {
            "mission": purpose.get("mission", ""),
            "user_needs": purpose.get("user_needs", []),
            "detected_domain": project_purpose_row["detected_domain"]
        }
        
        mission = purpose.get("mission", "")[:150]
        instruction_addition = ""
        if mission:
            instruction_addition = f"\n\n🎯 PROJECT MISSION: {mission}. Ensure hypotheses align with this purpose."
            logger.info(f"v43: Added project grounding for {project_id}")
        
        return project_grounding, instruction_addition
        
    except Exception as e:
        logger.warning(f"v43: Project grounding failed (non-fatal): {e}")
        return None, ""


def fetch_historical_patterns(
    cur,
    session_id: str,
    goal: str,
    get_embedding_fn
) -> list[dict]:
    """
    Fetch v44d historical patterns from past sessions.
    
    Args:
        cur: Database cursor
        session_id: Current session ID (to exclude)
        goal: Session goal
        get_embedding_fn: Embedding function
        
    Returns:
        List of historical pattern dicts
    """
    try:
        goal_embedding = get_embedding_fn(goal[:1000])
        
        historical_patterns = []
        seen_patterns = set()
        
        # 1. Goal-similarity patterns (semantic match)
        cur.execute("""
            SELECT pattern_type, source_phrase, confidence,
                   1 - (embedding <=> %s::vector) as similarity
            FROM detected_patterns
            WHERE session_id != %s
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 3
        """, (goal_embedding, session_id, goal_embedding))
        
        for row in cur.fetchall():
            key = (row["pattern_type"], row["source_phrase"][:50])
            if key not in seen_patterns:
                seen_patterns.add(key)
                historical_patterns.append({
                    "type": row["pattern_type"],
                    "phrase": row["source_phrase"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.8,
                    "similarity": round(float(row["similarity"]), 3) if row["similarity"] else None,
                    "source": "goal_similarity"
                })
        
        # 2. Recent patterns (last 7 days)
        cur.execute("""
            SELECT pattern_type, source_phrase, confidence, created_at
            FROM detected_patterns
            WHERE session_id != %s
              AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT 3
        """, (session_id,))
        
        for row in cur.fetchall():
            key = (row["pattern_type"], row["source_phrase"][:50])
            if key not in seen_patterns:
                seen_patterns.add(key)
                historical_patterns.append({
                    "type": row["pattern_type"],
                    "phrase": row["source_phrase"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.8,
                    "source": "recency"
                })
        
        if historical_patterns:
            logger.info(f"v44d: Found {len(historical_patterns[:5])} historical patterns for context")
        
        return historical_patterns[:5]
        
    except Exception as e:
        logger.warning(f"v44d: Historical pattern synthesis failed (non-fatal): {e}")
        return []

