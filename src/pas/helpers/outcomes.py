"""
Phase 4: record_outcome helpers for outcome recording and learning.

Extracted from server.py to reduce cyclomatic complexity.
Contains helpers for:
- v15b/v27: Outcome embedding generation
- v12b: Thompson sampling law updates
- v12a: PRM training data logging
- v13c: Critique accuracy tracking
- v40: CSR calibration logging
- v22: User trait persistence
- v8b: Auto-refresh triggering
- v34: Auto-tagging
"""

import json
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


def compute_outcome_embeddings(
    cur,
    best_node_id: str,
    failure_reason: Optional[str],
    get_embedding_fn: Callable
) -> tuple[Optional[list], Optional[list]]:
    """
    Compute v15b scope embedding and v27 failure_reason embedding.
    
    Args:
        cur: Database cursor
        best_node_id: ID of the best hypothesis node
        failure_reason: Optional failure reason text
        get_embedding_fn: Function to generate embeddings
        
    Returns:
        (scope_embedding, failure_reason_embedding) - either may be None
    """
    scope_embedding = None
    failure_reason_embedding = None
    
    # v15b: Get scope embedding from best node's declared_scope
    try:
        cur.execute(
            "SELECT declared_scope FROM thought_nodes WHERE id = %s",
            (best_node_id,)
        )
        scope_row = cur.fetchone()
        if scope_row and scope_row.get("declared_scope"):
            scope_embedding = get_embedding_fn(scope_row["declared_scope"])
    except Exception as e:
        logger.warning(f"v15b scope embedding failed: {e}")
    
    # v27: Embed failure_reason for semantic search
    if failure_reason:
        try:
            failure_reason_embedding = get_embedding_fn(failure_reason[:2000])
            logger.info(f"v27: Embedded failure_reason")
        except Exception as e:
            logger.warning(f"v27 failure_reason embedding failed: {e}")
    
    return scope_embedding, failure_reason_embedding


def insert_and_attribute_outcome(
    cur,
    session_id: str,
    outcome: str,
    confidence: float,
    winning_path: str,
    notes: Optional[str],
    failure_reason: Optional[str],
    scope_embedding: Optional[list],
    failure_reason_embedding: Optional[list]
) -> tuple[dict, dict]:
    """
    Insert outcome record and compute attribution stats.
    Also updates v12b success_count for Thompson Sampling.
    
    Args:
        cur: Database cursor
        session_id: Session UUID
        outcome: 'success', 'partial', or 'failure'
        confidence: Confidence score 0.0-1.0
        winning_path: ltree path of winning hypothesis
        notes: Optional notes
        failure_reason: Optional failure reason
        scope_embedding: v15b scope embedding
        failure_reason_embedding: v27 failure reason embedding
        
    Returns:
        (record, stats) - inserted record and attribution statistics
    """
    # Insert outcome record
    cur.execute(
        """
        INSERT INTO outcome_records (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding, failure_reason_embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at
        """,
        (session_id, outcome, confidence, winning_path, notes, failure_reason, scope_embedding, failure_reason_embedding)
    )
    record = cur.fetchone()
    
    # Count attributed nodes and laws
    cur.execute(
        """
        SELECT COUNT(*) as node_count,
               ARRAY_AGG(DISTINCT unnest) as laws
        FROM thought_nodes, UNNEST(supporting_laws)
        WHERE session_id = %s AND path <@ %s
        """,
        (session_id, winning_path)
    )
    stats = cur.fetchone()
    
    # v12b: Update success_count for Thompson Sampling on success outcomes
    if outcome == 'success' and stats and stats["laws"]:
        attributed_laws = [l for l in stats["laws"] if l is not None]
        if attributed_laws:
            cur.execute(
                "UPDATE scientific_laws SET success_count = success_count + 1 WHERE id = ANY(%s)",
                (attributed_laws,)
            )
    
    return record, stats


def log_training_data(
    cur,
    session_id: str,
    winning_path: str,
    outcome: str
) -> int:
    """
    Log v12a training data for PRM fine-tuning.
    
    Args:
        cur: Database cursor
        session_id: Session UUID
        winning_path: ltree path of winning hypothesis
        outcome: Outcome string
        
    Returns:
        Number of training records inserted
    """
    count = 0
    try:
        cur.execute(
            """
            SELECT content, path, supporting_laws FROM thought_nodes
            WHERE session_id = %s AND path <@ %s AND node_type = 'hypothesis'
            """,
            (session_id, winning_path)
        )
        for node in cur.fetchall():
            # Get first supporting law name
            law_name = None
            if node["supporting_laws"]:
                cur.execute("SELECT law_name FROM scientific_laws WHERE id = %s", (node["supporting_laws"][0],))
                law_row = cur.fetchone()
                law_name = law_row["law_name"] if law_row else None
            
            cur.execute(
                """
                INSERT INTO training_data (hypothesis_text, goal_text, outcome, depth, law_name, session_id)
                VALUES (%s, (SELECT goal FROM reasoning_sessions WHERE id = %s), %s, %s, %s, %s)
                """,
                (node["content"], session_id, outcome, len(str(node["path"]).split(".")), law_name, session_id)
            )
            count += 1
    except Exception as e:
        logger.warning(f"v12a training data logging failed: {e}")
    
    return count


def record_critique_accuracy(
    cur,
    session_id: str,
    winning_path: str,
    outcome: str,
    compute_critique_accuracy_fn: Callable
) -> int:
    """
    Track v13c critique accuracy for self-learning calibration.
    
    Args:
        cur: Database cursor
        session_id: Session UUID
        winning_path: ltree path of winning hypothesis
        outcome: Outcome string
        compute_critique_accuracy_fn: Function to compute accuracy
        
    Returns:
        Number of critique accuracy records inserted
    """
    count = 0
    try:
        # Find all critiqued nodes (nodes where likelihood was modified from initial)
        cur.execute(
            """
            SELECT tn.id, tn.path, tn.likelihood, tn.posterior_score
            FROM thought_nodes tn
            WHERE tn.session_id = %s 
              AND tn.node_type = 'hypothesis'
              AND tn.likelihood NOT IN (0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.75, 0.7)
            """,
            (session_id,)
        )
        critiqued_nodes = cur.fetchall()
        
        for cnode in critiqued_nodes:
            is_winner = str(cnode["path"]).startswith(str(winning_path)) or str(winning_path).startswith(str(cnode["path"]))
            critique_accurate = compute_critique_accuracy_fn(cnode["path"], winning_path, outcome)
            
            cur.execute(
                """
                INSERT INTO critique_accuracy 
                    (session_id, node_id, critique_severity, was_top_hypothesis, actual_outcome, critique_accurate)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (session_id, cnode["id"], 1.0 - float(cnode["likelihood"]), is_winner, outcome, critique_accurate)
            )
            count += 1
    except Exception as e:
        logger.warning(f"v13c critique accuracy tracking failed: {e}")
    
    return count


def log_calibration_record(
    cur,
    session_id: str,
    winning_path: str,
    outcome: str,
    map_outcome_fn: Callable
) -> bool:
    """
    Log v40 Phase 3 calibration data for CSR.
    
    Args:
        cur: Database cursor
        session_id: Session UUID
        winning_path: ltree path of winning hypothesis
        outcome: Outcome string
        map_outcome_fn: Function to map outcome to numeric (0, 0.5, 1)
        
    Returns:
        True if calibration record was logged
    """
    try:
        cur.execute(
            """
            SELECT tn.id, tn.posterior_score
            FROM thought_nodes tn
            WHERE tn.session_id = %s AND tn.path::text = %s
            """,
            (session_id, winning_path)
        )
        winning_node = cur.fetchone()
        
        if winning_node:
            predicted_conf = winning_node["posterior_score"]
            actual_outcome = map_outcome_fn(outcome)
            
            cur.execute(
                """
                INSERT INTO calibration_records 
                (session_id, winning_node_id, predicted_confidence, actual_outcome)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, winning_node["id"], predicted_conf, actual_outcome)
            )
            logger.info(f"v40: Logged calibration record - predicted: {predicted_conf:.3f}, actual: {actual_outcome}")
            return True
    except Exception as e:
        logger.warning(f"v40: Calibration logging failed: {e}")
    return False


def persist_user_traits(
    cur,
    session_id: str,
    outcome: str,
    get_outcome_multiplier_fn: Callable
) -> int:
    """
    Persist v22 Feature 1b user traits with outcome-based weighting.
    
    Args:
        cur: Database cursor
        session_id: Session UUID
        outcome: Outcome string
        get_outcome_multiplier_fn: Function to get outcome multiplier
        
    Returns:
        Number of traits persisted
    """
    traits_persisted = 0
    try:
        # Get session context for user_id and latent_traits
        cur.execute("SELECT context FROM reasoning_sessions WHERE id = %s", (session_id,))
        ctx_row = cur.fetchone()
        session_context = ctx_row["context"] if ctx_row and ctx_row["context"] else {}
        
        user_id = session_context.get("user_id")
        latent_traits = session_context.get("latent_traits", [])
        
        if user_id and latent_traits:
            outcome_multiplier = get_outcome_multiplier_fn(outcome)
            
            for trait_info in latent_traits:
                trait_name = trait_info.get("trait")
                if not trait_name:
                    continue
                
                evidence = trait_info.get("confidence", 0.5) * outcome_multiplier
                
                cur.execute("""
                    INSERT INTO user_trait_profiles (user_id, trait_name, cumulative_score, evidence_count)
                    VALUES (%s, %s, %s, 1)
                    ON CONFLICT (user_id, trait_name) DO UPDATE SET
                        cumulative_score = user_trait_profiles.cumulative_score + %s,
                        evidence_count = user_trait_profiles.evidence_count + 1,
                        last_reinforced_at = now()
                """, (user_id, trait_name, evidence, evidence))
                traits_persisted += 1
            
            if traits_persisted > 0:
                logger.info(f"v22: Persisted {traits_persisted} traits for user {user_id[:8]}...")
                
    except Exception as e:
        logger.warning(f"v22: Failed to persist traits: {e}")
    
    return traits_persisted


async def trigger_auto_refresh(
    cur,
    refresh_fn: Callable,
    min_samples: int = 5
) -> Optional[dict]:
    """
    Check and trigger v8b auto-refresh of law weights.
    
    Args:
        cur: Database cursor
        refresh_fn: The refresh_law_weights async function
        min_samples: Minimum samples threshold
        
    Returns:
        Refresh result dict if triggered, None otherwise
    """
    try:
        cur.execute("SELECT COUNT(*) FROM outcome_records")
        total_outcomes = cur.fetchone()[0]
        if total_outcomes >= min_samples and total_outcomes % min_samples == 0:
            # Trigger refresh on every Nth outcome (5, 10, 15, etc.)
            return await refresh_fn(min_samples=min_samples)
    except Exception as refresh_err:
        logger.warning(f"Auto-refresh check failed: {refresh_err}")
    return None


def apply_auto_tags(
    cur,
    conn,
    session_id: str,
    outcome: str
) -> Optional[list]:
    """
    Apply v34 auto-tags on success/partial outcomes.
    
    Args:
        cur: Database cursor
        conn: Database connection (for commit)
        session_id: Session UUID
        outcome: Outcome string
        
    Returns:
        List of applied tags, or None if not applicable
    """
    if outcome not in ('success', 'partial'):
        return None
    
    try:
        cur.execute(
            "SELECT suggested_tags FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        tags_row = cur.fetchone()
        if tags_row and tags_row.get("suggested_tags"):
            suggested = tags_row["suggested_tags"]
            if isinstance(suggested, str):
                suggested = json.loads(suggested)
            if suggested:
                # Apply tags using existing tag_session logic
                for tag in suggested:
                    # Normalize tag
                    normalized = tag.strip().lower()
                    if normalized:
                        cur.execute(
                            """
                            INSERT INTO session_tags (session_id, tag)
                            VALUES (%s, %s)
                            ON CONFLICT (session_id, tag) DO NOTHING
                            """,
                            (session_id, normalized)
                        )
                conn.commit()
                logger.info(f"v34: Auto-tagged session {session_id} with {suggested}")
                return suggested
    except Exception as e:
        logger.warning(f"v34 auto-tagging failed: {e}")
    return None
