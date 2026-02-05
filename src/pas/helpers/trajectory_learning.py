"""
Phase 34: Trajectory Learning System

Captures patterns from Antigravity agent trajectories and injects
successful tool sequences as few-shot examples in prepare_expansion.
"""
import logging
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger("pas-server")

# Connection timeout for trajectory capture (sync, not async)
CAPTURE_TIMEOUT = 5  # seconds


def capture_current_trajectory(
    session_id: str, 
    outcome: str,
    session_goal: str,
    conn,
    get_embedding_fn
) -> Optional[str]:
    """
    Capture the current Antigravity trajectory after record_outcome.
    
    Called synchronously with timeout to avoid async thread reliability issues.
    
    Args:
        session_id: PAS reasoning session ID
        outcome: 'success', 'partial', or 'failure'
        session_goal: Goal text for matching cascades
        conn: Database connection
        get_embedding_fn: Function to generate embeddings
        
    Returns:
        trajectory_id if captured, None otherwise
    """
    try:
        from pas.helpers.antigravity_client import get_antigravity_client, AntigravityClient
        
        # Force re-discovery to handle Antigravity restarts
        global _client
        _client = None
        client = get_antigravity_client()
        
        if not client.primary:
            logger.debug("No Antigravity instance found for trajectory capture")
            return None
        
        # Get all trajectories and find matching cascade
        trajectories = client.get_all_trajectories()
        trajectory_dict = trajectories.get("trajectorySummaries", {})
        
        cascade_id = match_current_cascade(trajectory_dict, session_goal)
        if not cascade_id:
            logger.debug(f"No matching cascade found for goal: {session_goal[:50]}...")
            return None
        
        # Check if already captured (idempotency)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM trajectory_patterns WHERE cascade_id = %s",
            (cascade_id,)
        )
        existing = cur.fetchone()
        if existing:
            logger.debug(f"Trajectory {cascade_id} already captured")
            return str(existing["id"])
        
        # Get full trajectory details
        trajectory_data = trajectory_dict.get(cascade_id, {})
        summary = trajectory_data.get("summary", "")
        
        # Use GetCascadeTrajectorySteps for reliable step retrieval
        # (GetCascadeTrajectory times out for large sessions)
        steps = client.get_trajectory_steps(cascade_id, limit=200)
        tool_sequence = extract_tool_sequence({"steps": steps})
        step_count = trajectory_data.get("stepCount", 0) or len(steps)

        
        # Generate embedding for summary (handle async functions)
        summary_embedding = None
        if summary and get_embedding_fn:
            try:
                import asyncio
                import inspect
                if inspect.iscoroutinefunction(get_embedding_fn):
                    # Async function - run in event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Already in async context - can't use run_until_complete
                        # Skip embedding for now, will be added later
                        summary_embedding = None
                    except RuntimeError:
                        # No running loop - safe to create one
                        summary_embedding = asyncio.run(get_embedding_fn(summary[:1000]))
                else:
                    summary_embedding = get_embedding_fn(summary[:1000])
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        
        # Insert pattern
        cur.execute("""
            INSERT INTO trajectory_patterns 
            (cascade_id, session_id, summary, summary_embedding, outcome, 
             tool_sequence, step_count, workspace_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            cascade_id,
            session_id,
            summary,
            summary_embedding,
            outcome,
            json.dumps(tool_sequence),
            step_count,
            client.primary.workspace_id
        ))
        
        result = cur.fetchone()
        conn.commit()
        
        pattern_id = str(result[0]) if result else None
        logger.info(f"Phase 34: Captured trajectory {cascade_id} ({step_count} steps, {outcome})")
        return pattern_id
        
    except Exception as e:
        logger.warning(f"Trajectory capture failed (non-fatal): {e}")
        return None


def match_current_cascade(
    trajectory_dict: Dict[str, Any], 
    session_goal: str
) -> Optional[str]:
    """
    Find the cascade_id that matches the current PAS session.
    
    Uses summary similarity heuristic - the most recent cascade
    with a summary containing keywords from the goal.
    
    Args:
        trajectory_dict: Dict of cascade_id -> trajectory data
        session_goal: PAS session goal text
        
    Returns:
        Matching cascade_id or None
    """
    if not trajectory_dict or not session_goal:
        return None
    
    # Extract keywords from goal
    goal_words = set(session_goal.lower().split()[:10])
    
    best_match = None
    best_score = 0
    
    for cascade_id, data in trajectory_dict.items():
        summary = (data.get("summary") or "").lower()
        if not summary:
            continue
            
        # Simple keyword overlap score
        summary_words = set(summary.split())
        overlap = len(goal_words & summary_words)
        
        if overlap > best_score:
            best_score = overlap
            best_match = cascade_id
    
    # Require at least 2 keyword matches
    if best_score >= 2:
        return best_match
    
    return None


def extract_tool_sequence(trajectory: Dict[str, Any]) -> List[Dict]:
    """
    Extract ordered tool calls from a trajectory.
    
    Handles GetCascadeTrajectorySteps format where:
    - step.type = CORTEX_STEP_TYPE_* (MCP_TOOL, RUN_COMMAND, etc)
    - step.metadata.toolCall = {name, argumentsJson, ...}
    
    Args:
        trajectory: Trajectory data with steps list
        
    Returns:
        List of {tool, args_hash, result_type} dicts
    """
    sequence = []
    
    steps = trajectory.get("steps", [])
    if isinstance(steps, dict):
        steps = list(steps.values())
    
    for step in steps:
        if not isinstance(step, dict):
            continue
        
        step_type = step.get("type", "")
        metadata = step.get("metadata", {})
        
        # Extract tool name from step type or toolCall
        tool_name = None
        args_str = ""
        
        # Handle MCP tool calls (metadata.toolCall.name)
        tool_call = metadata.get("toolCall", {})
        if tool_call:
            tool_name = tool_call.get("name", "")
            args_str = tool_call.get("argumentsJson", "") or ""
        
        # Fallback: extract from step type (CORTEX_STEP_TYPE_RUN_COMMAND -> run_command)
        if not tool_name and step_type.startswith("CORTEX_STEP_TYPE_"):
            tool_name = step_type.replace("CORTEX_STEP_TYPE_", "").lower()
        
        if not tool_name:
            continue
        
        # Hash args to detect patterns without storing full content
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
        
        # Categorize result based on step status
        status = step.get("status", "")
        if "ERROR" in status or "FAILED" in status:
            result_type = "error"
        elif "DONE" in status or "CLEARED" in status:
            result_type = "success"
        else:
            result_type = "pending"
        
        sequence.append({
            "tool": tool_name,
            "args_hash": args_hash,
            "result_type": result_type
        })
    
    return sequence



def fetch_similar_trajectories(
    goal_embedding: List[float],
    conn,
    outcome_filter: Optional[str] = None,
    limit: int = 3
) -> List[Dict]:
    """
    Query trajectory_patterns by embedding similarity.
    
    Args:
        goal_embedding: Vector embedding of current goal
        conn: Database connection
        outcome_filter: Optional filter ('success', 'failure', etc)
        limit: Maximum results
        
    Returns:
        List of matching trajectory patterns
    """
    try:
        cur = conn.cursor()
        
        query = """
            SELECT cascade_id, summary, outcome, tool_sequence, step_count,
                   1 - (summary_embedding <=> %s::vector) as similarity
            FROM trajectory_patterns
            WHERE summary_embedding IS NOT NULL
        """
        params = [goal_embedding]
        
        if outcome_filter:
            query += " AND outcome = %s"
            params.append(outcome_filter)
        
        query += """
            ORDER BY summary_embedding <=> %s::vector
            LIMIT %s
        """
        params.extend([goal_embedding, limit])
        
        cur.execute(query, params)
        
        results = []
        for row in cur.fetchall():
            results.append({
                "cascade_id": row[0],
                "summary": row[1],
                "outcome": row[2],
                "tool_sequence": row[3] if row[3] else [],
                "step_count": row[4],
                "similarity": round(float(row[5]), 3) if row[5] else 0
            })
        
        return results
        
    except Exception as e:
        logger.warning(f"Trajectory similarity search failed: {e}")
        return []
