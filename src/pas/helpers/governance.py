"""
PAS Governance Helpers (Phase 6)

Query APIs for Vision → Roadmap → Plans hierarchy.
Implements versioned artifact storage with prompt linkage.

PAS Session: 18e98d43-9bd9-4c56-a8c5-036e5e9c8fd1 | Score: 0.927
"""

from typing import Any, Optional
import logging

from pas.utils import get_db_connection, get_embedding, safe_close_connection

logger = logging.getLogger(__name__)


def get_or_create_project_vision(
    project_id: str,
    mission: Optional[str] = None,
    user_needs: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Get or create project vision record.
    
    Uses ON CONFLICT for upsert - vision can exist before project_registry sync.
    
    Args:
        project_id: Project identifier
        mission: Core mission statement
        user_needs: List of user needs served
        
    Returns:
        Vision record dict
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Upsert vision
            cur.execute("""
                INSERT INTO project_vision (project_id, mission, user_needs)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id) DO UPDATE SET
                    mission = COALESCE(EXCLUDED.mission, project_vision.mission),
                    user_needs = COALESCE(EXCLUDED.user_needs, project_vision.user_needs),
                    updated_at = NOW()
                RETURNING id, project_id, mission, user_needs, created_at
            """, (project_id, mission, user_needs))
            
            row = cur.fetchone()
            conn.commit()
            
            return {
                "id": str(row["id"]),
                "project_id": row["project_id"],
                "mission": row["mission"],
                "user_needs": row["user_needs"],
                "created_at": str(row["created_at"])
            }
    finally:
        safe_close_connection(conn)


def get_roadmap_phases(project_id: str) -> list[dict[str, Any]]:
    """
    Get all roadmap phases for a project, ordered by sequence.
    
    Args:
        project_id: Project identifier
        
    Returns:
        List of phase dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, phase_name, description, status, sequence, created_at
                FROM roadmap_phases
                WHERE project_id = %s
                ORDER BY sequence
            """, (project_id,))
            
            return [
                {
                    "id": str(row["id"]),
                    "phase_name": row["phase_name"],
                    "description": row["description"],
                    "status": row["status"],
                    "sequence": row["sequence"],
                    "created_at": str(row["created_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def create_roadmap_phase(
    project_id: str,
    phase_name: str,
    description: Optional[str] = None,
    status: str = "planned"
) -> dict[str, Any]:
    """
    Create a new roadmap phase.
    
    Auto-assigns sequence as max+1 for the project.
    
    Args:
        project_id: Project identifier
        phase_name: Name of the phase
        description: Phase description
        status: planned, active, complete, blocked
        
    Returns:
        Created phase dict
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Get next sequence
            cur.execute("""
                SELECT COALESCE(MAX(sequence), 0) + 1 as next_seq
                FROM roadmap_phases
                WHERE project_id = %s
            """, (project_id,))
            next_seq = cur.fetchone()["next_seq"]
            
            cur.execute("""
                INSERT INTO roadmap_phases (project_id, phase_name, description, status, sequence)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, phase_name, sequence, created_at
            """, (project_id, phase_name, description, status, next_seq))
            
            row = cur.fetchone()
            conn.commit()
            
            return {
                "id": str(row["id"]),
                "project_id": project_id,
                "phase_name": row["phase_name"],
                "sequence": row["sequence"],
                "status": status,
                "created_at": str(row["created_at"])
            }
    finally:
        safe_close_connection(conn)


# ============================================================================
# v91: Phase Execution Helpers
# ============================================================================

def activate_phase(phase_id: str) -> dict[str, Any]:
    """
    Activate a phase for execution.
    
    Changes status from 'planned' to 'active'.
    Returns phase context for execution.
    
    Args:
        phase_id: Phase UUID
        
    Returns:
        Phase context including goal, dependencies, success criteria
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Update status
            cur.execute("""
                UPDATE roadmap_phases
                SET status = 'active', updated_at = NOW()
                WHERE id = %s AND status = 'planned'
                RETURNING id, phase_name, description, project_id
            """, (phase_id,))
            
            row = cur.fetchone()
            if not row:
                # Check if already active or doesn't exist
                cur.execute("SELECT status FROM roadmap_phases WHERE id = %s", (phase_id,))
                existing = cur.fetchone()
                if existing:
                    return {"error": f"Phase already has status: {existing['status']}"}
                return {"error": "Phase not found"}
            
            conn.commit()
            project_id = row["project_id"]
            
            # Get dependencies
            cur.execute("""
                SELECT rp.phase_name
                FROM phase_dependencies pd
                JOIN roadmap_phases rp ON rp.id = pd.depends_on_phase_id
                WHERE pd.phase_id = %s
            """, (phase_id,))
            dependencies = [r["phase_name"] for r in cur.fetchall()]
            
            # Get success criteria
            cur.execute("""
                SELECT criterion, done
                FROM phase_success_criteria
                WHERE phase_id = %s
            """, (phase_id,))
            criteria = [
                {"criterion": r["criterion"], "done": r["done"]}
                for r in cur.fetchall()
            ]
            
            return {
                "success": True,
                "phase_id": str(row["id"]),
                "phase_name": row["phase_name"],
                "description": row["description"],
                "status": "active",
                "dependencies": dependencies,
                "success_criteria": criteria,
                "next_step": f"Start PAS session: start_reasoning_session(user_goal='Implement {row['phase_name']}')"
            }
    finally:
        safe_close_connection(conn)


def complete_phase(phase_id: str, notes: Optional[str] = None) -> dict[str, Any]:
    """
    Complete a phase after execution.
    
    Validates all success criteria are checked, then marks complete.
    
    Args:
        phase_id: Phase UUID
        notes: Optional completion notes
        
    Returns:
        Completion confirmation or validation errors
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check uncompleted criteria
            cur.execute("""
                SELECT criterion FROM phase_success_criteria
                WHERE phase_id = %s AND done = FALSE
            """, (phase_id,))
            unchecked = [r["criterion"] for r in cur.fetchall()]
            
            if unchecked:
                return {
                    "error": "Cannot complete phase - unchecked success criteria",
                    "unchecked_criteria": unchecked,
                    "hint": "Use update_success_criterion to mark each as checked"
                }
            
            # Update status
            cur.execute("""
                UPDATE roadmap_phases
                SET status = 'complete', updated_at = NOW()
                WHERE id = %s AND status = 'active'
                RETURNING phase_name
            """, (phase_id,))
            
            row = cur.fetchone()
            if not row:
                cur.execute("SELECT status FROM roadmap_phases WHERE id = %s", (phase_id,))
                existing = cur.fetchone()
                if existing:
                    return {"error": f"Phase has status: {existing['status']} (expected 'active')"}
                return {"error": "Phase not found"}
            
            conn.commit()
            
            return {
                "success": True,
                "phase_name": row["phase_name"],
                "status": "complete",
                "notes": notes
            }
    finally:
        safe_close_connection(conn)


def get_phase_context(phase_id: str) -> dict[str, Any]:
    """
    Get full execution context for a phase.
    
    Returns everything needed to execute a phase:
    - Goal and description
    - Dependencies and their status
    - Success criteria
    - Linked implementation plans
    
    Args:
        phase_id: Phase UUID
        
    Returns:
        Complete phase execution context
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Get phase details
            cur.execute("""
                SELECT id, phase_name, description, status, project_id, sequence
                FROM roadmap_phases
                WHERE id = %s
            """, (phase_id,))
            
            row = cur.fetchone()
            if not row:
                return {"error": "Phase not found"}
            
            project_id = row["project_id"]
            
            # Get dependencies with status
            cur.execute("""
                SELECT rp.phase_name, rp.status
                FROM phase_dependencies pd
                JOIN roadmap_phases rp ON rp.id = pd.depends_on_phase_id
                WHERE pd.phase_id = %s
            """, (phase_id,))
            dependencies = [
                {
                    "phase": r["phase_name"], 
                    "status": r["status"],
                    "ready": r["status"] == "complete"
                }
                for r in cur.fetchall()
            ]
            
            # Check if all dependencies are met
            deps_met = all(d["ready"] for d in dependencies) if dependencies else True
            
            # Get success criteria
            cur.execute("""
                SELECT id, criterion, done
                FROM phase_success_criteria
                WHERE phase_id = %s
            """, (phase_id,))
            criteria = [
                {"id": str(r["id"]), "criterion": r["criterion"], "done": r["done"]}
                for r in cur.fetchall()
            ]
            
            # Get linked artifacts
            cur.execute("""
                SELECT id, name, artifact_type, version
                FROM artifacts
                WHERE roadmap_phase_id = %s
                ORDER BY version DESC
            """, (phase_id,))
            artifacts = [
                {"id": str(r["id"]), "name": r["name"], "type": r["artifact_type"], "version": r["version"]}
                for r in cur.fetchall()
            ]
            
            return {
                "phase_id": str(row["id"]),
                "phase_name": row["phase_name"],
                "description": row["description"],
                "status": row["status"],
                "sequence": row["sequence"],
                "project_id": project_id,
                "dependencies": dependencies,
                "dependencies_met": deps_met,
                "success_criteria": criteria,
                "artifacts": artifacts,
                "can_activate": row["status"] == "planned" and deps_met
            }
    finally:
        safe_close_connection(conn)


# ============================================================================
# v91: Structured Gaps Helpers
# ============================================================================

def store_structured_gap(
    session_id: str,
    project_id: str,
    gap_layer: str,
    gap_description: str,
    severity: str = "medium"
) -> str:
    """
    Store a gap identified from sequential analysis.
    
    Args:
        session_id: PAS session UUID
        project_id: Project identifier
        gap_layer: One of: code_structure, dependencies, data_flow, interfaces, workflows
        gap_description: Description of the gap
        severity: low, medium, high, critical
        
    Returns:
        Created gap ID
    """
    conn = get_db_connection()
    try:
        embedding = get_embedding(gap_description[:4000])
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO structured_gaps 
                    (session_id, project_id, gap_layer, gap_description, severity, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (session_id, project_id, gap_layer, gap_description, severity, embedding))
            
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def get_unaddressed_gaps(
    project_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Get unaddressed gaps for a project or session.
    
    Args:
        project_id: Filter by project
        session_id: Filter by session
        
    Returns:
        List of gap dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if session_id:
                cur.execute("""
                    SELECT id, gap_layer, gap_description, severity, created_at
                    FROM structured_gaps
                    WHERE session_id = %s AND addressed = FALSE
                    ORDER BY 
                        CASE severity 
                            WHEN 'critical' THEN 1 
                            WHEN 'high' THEN 2 
                            WHEN 'medium' THEN 3 
                            ELSE 4 
                        END
                """, (session_id,))
            elif project_id:
                cur.execute("""
                    SELECT id, session_id, gap_layer, gap_description, severity, created_at
                    FROM structured_gaps
                    WHERE project_id = %s AND addressed = FALSE
                    ORDER BY 
                        CASE severity 
                            WHEN 'critical' THEN 1 
                            WHEN 'high' THEN 2 
                            WHEN 'medium' THEN 3 
                            ELSE 4 
                        END
                """, (project_id,))
            else:
                return []
            
            return [
                {
                    "id": str(r["id"]),
                    "session_id": str(r.get("session_id", "")),
                    "gap_layer": r["gap_layer"],
                    "gap_description": r["gap_description"],
                    "severity": r["severity"],
                    "created_at": str(r["created_at"])
                }
                for r in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def mark_gap_addressed(gap_id: str, addressed_by_node_id: Optional[str] = None) -> bool:
    """
    Mark a gap as addressed.
    
    Args:
        gap_id: Gap UUID
        addressed_by_node_id: Optional thought node that addressed it
        
    Returns:
        True if update succeeded
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE structured_gaps
                SET addressed = TRUE, addressed_by = %s
                WHERE id = %s
            """, (addressed_by_node_id, gap_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        safe_close_connection(conn)



def store_artifact(
    project_id: str,
    name: str,
    content: str,
    artifact_type: str = "implementation_plan",
    session_id: Optional[str] = None,
    roadmap_phase_id: Optional[str] = None,
    source_verbatim_log_id: Optional[str] = None,
    tags: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Store a new artifact version.
    
    Versioning: Uses advisory lock + MAX(version)+1 in transaction for atomic increment.
    
    Args:
        project_id: Project identifier
        name: Artifact name (used for versioning)
        content: Full artifact content
        artifact_type: roadmap, implementation_plan, walkthrough, vision, other
        session_id: Optional PAS session that generated this
        roadmap_phase_id: Optional link to roadmap phase
        source_verbatim_log_id: Optional link to verbatim prompt
        tags: List of tags for filtering
        
    Returns:
        Created artifact dict with version
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Use advisory lock based on hash of (project_id, name) to prevent concurrent inserts
            lock_key = hash((project_id, name)) & 0x7FFFFFFF  # Ensure positive int32
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))
            
            # Get next version
            cur.execute("""
                SELECT COALESCE(MAX(version), 0) + 1 as next_version
                FROM artifacts
                WHERE project_id = %s AND name = %s
            """, (project_id, name))
            next_version = cur.fetchone()["next_version"]
            
            # Embed content (truncate for embedding)
            content_embedding = get_embedding(content[:4000])
            
            # Insert artifact
            cur.execute("""
                INSERT INTO artifacts (
                    project_id, name, content, artifact_type, version,
                    session_id, roadmap_phase_id, source_verbatim_log_id,
                    tags, content_embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, version, created_at
            """, (
                project_id, name, content, artifact_type, next_version,
                session_id, roadmap_phase_id, source_verbatim_log_id,
                tags or [], content_embedding
            ))
            
            row = cur.fetchone()
            conn.commit()
            
            return {
                "id": str(row["id"]),
                "project_id": project_id,
                "name": name,
                "artifact_type": artifact_type,
                "version": row["version"],
                "created_at": str(row["created_at"])
            }
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to store artifact: {e}")
        raise e
    finally:
        safe_close_connection(conn)



def get_artifact_versions(
    project_id: str,
    name: str
) -> list[dict[str, Any]]:
    """
    Get all versions of an artifact.
    
    Args:
        project_id: Project identifier
        name: Artifact name
        
    Returns:
        List of version dicts, newest first
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, version, artifact_type, tags, session_id, created_at
                FROM artifacts
                WHERE project_id = %s AND name = %s
                ORDER BY version DESC
            """, (project_id, name))
            
            return [
                {
                    "id": str(row["id"]),
                    "version": row["version"],
                    "artifact_type": row["artifact_type"],
                    "tags": row["tags"],
                    "session_id": str(row["session_id"]) if row["session_id"] else None,
                    "created_at": str(row["created_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def get_latest_artifact(
    project_id: str,
    name: str
) -> Optional[dict[str, Any]]:
    """
    Get the latest version of an artifact with content.
    
    Args:
        project_id: Project identifier
        name: Artifact name
        
    Returns:
        Artifact dict with content, or None if not found
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, version, artifact_type, content, tags, session_id, created_at
                FROM artifacts
                WHERE project_id = %s AND name = %s
                ORDER BY version DESC
                LIMIT 1
            """, (project_id, name))
            
            row = cur.fetchone()
            if not row:
                return None
                
            return {
                "id": str(row["id"]),
                "version": row["version"],
                "artifact_type": row["artifact_type"],
                "content": row["content"],
                "tags": row["tags"],
                "session_id": str(row["session_id"]) if row["session_id"] else None,
                "created_at": str(row["created_at"])
            }
    finally:
        safe_close_connection(conn)


def search_artifacts_by_tag(
    project_id: str,
    tags: list[str],
    limit: int = 10
) -> list[dict[str, Any]]:
    """
    Search artifacts by tag overlap.
    
    Args:
        project_id: Project identifier
        tags: Tags to match (any overlap)
        limit: Max results
        
    Returns:
        Matching artifact summaries
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, artifact_type, version, tags, created_at
                FROM artifacts
                WHERE project_id = %s
                  AND tags && %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (project_id, tags, limit))
            
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "artifact_type": row["artifact_type"],
                    "version": row["version"],
                    "tags": row["tags"],
                    "created_at": str(row["created_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def search_artifacts_semantic(
    project_id: str,
    query: str,
    limit: int = 5
) -> list[dict[str, Any]]:
    """
    Semantic search over artifact content.
    
    Args:
        project_id: Project identifier
        query: Natural language query
        limit: Max results
        
    Returns:
        Matching artifacts with similarity scores
    """
    conn = get_db_connection()
    try:
        query_embedding = get_embedding(query)
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, artifact_type, version, tags,
                       1 - (content_embedding <=> %s::vector) as similarity
                FROM artifacts
                WHERE project_id = %s
                  AND content_embedding IS NOT NULL
                ORDER BY content_embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, project_id, query_embedding, limit))
            
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "artifact_type": row["artifact_type"],
                    "version": row["version"],
                    "tags": row["tags"],
                    "similarity": round(row["similarity"], 4)
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def get_governance_hierarchy(project_id: str) -> dict[str, Any]:
    """
    Get full governance hierarchy: Vision → Phases → Artifacts.
    
    Args:
        project_id: Project identifier
        
    Returns:
        Nested hierarchy dict
    """
    vision = get_or_create_project_vision(project_id)
    phases = get_roadmap_phases(project_id)
    
    # Attach artifacts to each phase
    conn = get_db_connection()
    try:
        enriched_phases = []
        with conn.cursor() as cur:
            for phase in phases:
                cur.execute("""
                    SELECT id, name, artifact_type, version, tags, created_at
                    FROM artifacts
                    WHERE roadmap_phase_id = %s
                    ORDER BY created_at
                """, (phase["id"],))
                
                phase_artifacts = [
                    {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "artifact_type": row["artifact_type"],
                        "version": row["version"],
                        "tags": row["tags"],
                        "created_at": str(row["created_at"])
                    }
                    for row in cur.fetchall()
                ]
                
                enriched_phases.append({
                    **phase,
                    "artifacts": phase_artifacts,
                    "artifact_count": len(phase_artifacts)
                })
        
        return {
            "project_id": project_id,
            "vision": vision,
            "phases": enriched_phases,
            "phase_count": len(enriched_phases)
        }
    finally:
        safe_close_connection(conn)


# ============================================================================
# v88 Roadmap Functions
# PAS Research Session: b995dca2-6f1e-47f7-8e01-a0e5452026ce
# PAS Implementation Session: 25456d82-d9ee-49b7-b61f-4f3e82068d41
# ============================================================================


def create_roadmap(
    project_id: str,
    title: str,
    version_tag: Optional[str] = None,
    priority_taxonomy: Optional[dict] = None,
    architecture_content: Optional[str] = None
) -> str:
    """
    Create a new roadmap for a project.
    
    Args:
        project_id: Project identifier
        title: Roadmap title
        version_tag: Version string (e.g., "v1.0")
        priority_taxonomy: Dict mapping priorities to definitions
        architecture_content: Architecture diagram/notes
        
    Returns:
        Roadmap UUID
    """
    import json
    
    conn = get_db_connection()
    try:
        # Generate embedding from title and content
        embed_text = f"{title} {architecture_content or ''}"
        embedding = get_embedding(embed_text[:4000]) if embed_text.strip() else None
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO roadmaps (
                    project_id, title, version_tag, 
                    priority_taxonomy, architecture_content, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                project_id, title, version_tag,
                json.dumps(priority_taxonomy or {}),
                architecture_content, embedding
            ))
            
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def get_roadmaps(
    project_id: str,
    status: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Get all roadmaps for a project, optionally filtered by status.
    
    Args:
        project_id: Project identifier
        status: Optional filter (draft, active, archived)
        
    Returns:
        List of roadmap dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute("""
                    SELECT id, title, version_tag, status, priority_taxonomy,
                           created_at, updated_at
                    FROM roadmaps 
                    WHERE project_id = %s AND status = %s
                    ORDER BY created_at DESC
                """, (project_id, status))
            else:
                cur.execute("""
                    SELECT id, title, version_tag, status, priority_taxonomy,
                           created_at, updated_at
                    FROM roadmaps 
                    WHERE project_id = %s
                    ORDER BY created_at DESC
                """, (project_id,))
            
            return [
                {
                    "id": str(row["id"]),
                    "title": row["title"],
                    "version_tag": row["version_tag"],
                    "status": row["status"],
                    "priority_taxonomy": row["priority_taxonomy"],
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def update_roadmap(roadmap_id: str, **fields) -> bool:
    """
    Update roadmap fields.
    
    Args:
        roadmap_id: Roadmap UUID
        **fields: Fields to update (title, version_tag, status, 
                  priority_taxonomy, architecture_content)
        
    Returns:
        True if updated, False if no valid fields or roadmap not found
    """
    import json
    
    allowed = {'title', 'version_tag', 'status', 'priority_taxonomy', 
               'architecture_content'}
    updates = {k: v for k, v in fields.items() if k in allowed}
    
    if not updates:
        return False
    
    # JSON-encode priority_taxonomy if present
    if 'priority_taxonomy' in updates and isinstance(updates['priority_taxonomy'], dict):
        updates['priority_taxonomy'] = json.dumps(updates['priority_taxonomy'])
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            set_parts = [f"{k} = %s" for k in updates.keys()]
            set_clause = ", ".join(set_parts)
            values = list(updates.values()) + [roadmap_id]
            
            cur.execute(f"""
                UPDATE roadmaps 
                SET {set_clause}, updated_at = NOW()
                WHERE id = %s
            """, values)
            
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        safe_close_connection(conn)


# ============================================================================
# v88 Phase Success Criteria
# ============================================================================


def create_phase_success_criterion(
    phase_id: str,
    criterion: str,
    sequence: int
) -> str:
    """
    Create a success criterion for a phase.
    
    Args:
        phase_id: Phase UUID
        criterion: Success criterion text
        sequence: Order in list
        
    Returns:
        Criterion UUID
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO phase_success_criteria (phase_id, criterion, sequence)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (phase_id, criterion, sequence))
            
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def update_success_criterion(
    criterion_id: str,
    done: bool,
    verified_by_session_id: Optional[str] = None
) -> bool:
    """
    Mark a criterion as done/not done.
    
    Args:
        criterion_id: Criterion UUID
        done: Whether criterion is complete
        verified_by_session_id: Optional PAS session that verified
        
    Returns:
        True if updated
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE phase_success_criteria 
                SET done = %s, 
                    verified_at = CASE WHEN %s THEN NOW() ELSE NULL END,
                    verified_by_session_id = %s
                WHERE id = %s
            """, (done, done, verified_by_session_id, criterion_id))
            
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        safe_close_connection(conn)


def get_phase_success_criteria(phase_id: str) -> list[dict[str, Any]]:
    """
    Get all success criteria for a phase.
    
    Args:
        phase_id: Phase UUID
        
    Returns:
        List of criteria dicts, ordered by sequence
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, criterion, done, verified_at, sequence
                FROM phase_success_criteria 
                WHERE phase_id = %s
                ORDER BY sequence
            """, (phase_id,))
            
            return [
                {
                    "id": str(row["id"]),
                    "criterion": row["criterion"],
                    "done": row["done"],
                    "verified_at": str(row["verified_at"]) if row["verified_at"] else None,
                    "sequence": row["sequence"]
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


# ============================================================================
# v88 Phase Dependencies
# ============================================================================


def create_phase_dependency(phase_id: str, depends_on_phase_id: str) -> bool:
    """
    Create a dependency between phases.
    
    Args:
        phase_id: Phase that has the dependency
        depends_on_phase_id: Phase that must complete first
        
    Returns:
        True if created, False if already exists or invalid
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO phase_dependencies (phase_id, depends_on_phase_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (phase_id, depends_on_phase_id))
                
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                return False
    finally:
        safe_close_connection(conn)


def get_phase_dependencies(phase_id: str) -> list[dict[str, Any]]:
    """
    Get phases that this phase depends on.
    
    Args:
        phase_id: Phase UUID
        
    Returns:
        List of dependency phase dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT rp.id, rp.phase_name, rp.status
                FROM phase_dependencies pd
                JOIN roadmap_phases rp ON rp.id = pd.depends_on_phase_id
                WHERE pd.phase_id = %s
            """, (phase_id,))
            
            return [
                {
                    "id": str(row["id"]),
                    "phase_name": row["phase_name"],
                    "status": row["status"]
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


# ============================================================================
# v88 Phase Critiques
# ============================================================================


def create_phase_critique(
    phase_id: str,
    critique_text: str,
    status: str = 'open'
) -> str:
    """
    Create a critique for a phase.
    
    Args:
        phase_id: Phase UUID
        critique_text: Critique description
        status: open, warning, addressed
        
    Returns:
        Critique UUID
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO phase_critiques (phase_id, critique_text, status)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (phase_id, critique_text, status))
            
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def update_phase_critique(
    critique_id: str,
    status: str,
    addressed_in_session_id: Optional[str] = None
) -> bool:
    """
    Update critique status.
    
    Args:
        critique_id: Critique UUID
        status: New status (open, warning, addressed)
        addressed_in_session_id: Optional PAS session that addressed it
        
    Returns:
        True if updated
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE phase_critiques 
                SET status = %s, addressed_in_session_id = %s
                WHERE id = %s
            """, (status, addressed_in_session_id, critique_id))
            
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        safe_close_connection(conn)


def get_phase_critiques(
    phase_id: str,
    status: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Get critiques for a phase.
    
    Args:
        phase_id: Phase UUID
        status: Optional filter (open, warning, addressed)
        
    Returns:
        List of critique dicts
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute("""
                    SELECT id, critique_text, status, addressed_in_session_id, created_at
                    FROM phase_critiques 
                    WHERE phase_id = %s AND status = %s
                    ORDER BY created_at
                """, (phase_id, status))
            else:
                cur.execute("""
                    SELECT id, critique_text, status, addressed_in_session_id, created_at
                    FROM phase_critiques 
                    WHERE phase_id = %s
                    ORDER BY created_at
                """, (phase_id,))
            
            return [
                {
                    "id": str(row["id"]),
                    "critique_text": row["critique_text"],
                    "status": row["status"],
                    "addressed_in_session_id": str(row["addressed_in_session_id"]) 
                        if row["addressed_in_session_id"] else None,
                    "created_at": str(row["created_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


# ============================================================================
# v89 Cross-Phase Decisions & Research Findings
# PAS Session: 8a0e1440-84e5-4d3a-968c-dc0ca0062151
# ============================================================================

def create_cross_phase_decision(
    project_id: str,
    decision_summary: str,
    options_considered: Optional[list] = None,
    chosen_option: Optional[str] = None,
    rationale: Optional[str] = None,
    pas_node_id: Optional[str] = None,
    phase_ids: Optional[list] = None
) -> str:
    """
    Create a cross-phase decision record.
    
    Args:
        project_id: Project identifier
        decision_summary: Brief description of the decision
        options_considered: List of options that were evaluated
        chosen_option: The selected option
        rationale: Why this option was chosen
        pas_node_id: Link to PAS thought node if applicable
        phase_ids: List of phase UUIDs this decision affects
        
    Returns:
        Decision UUID
    """
    import json
    
    conn = get_db_connection()
    try:
        embed_text = f"{decision_summary} {rationale or ''}"
        embedding = get_embedding(embed_text[:4000]) if embed_text.strip() else None
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cross_phase_decisions (
                    project_id, decision_summary, options_considered,
                    chosen_option, rationale, pas_node_id, phase_ids, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                project_id, decision_summary, 
                json.dumps(options_considered or []),
                chosen_option, rationale, pas_node_id,
                phase_ids or [], embedding
            ))
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def get_cross_phase_decisions(project_id: str) -> list:
    """
    Get all cross-phase decisions for a project.
    
    Args:
        project_id: Project identifier
        
    Returns:
        List of decision records
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, decision_summary, options_considered,
                       chosen_option, rationale, pas_node_id, phase_ids,
                       created_at
                FROM cross_phase_decisions
                WHERE project_id = %s
                ORDER BY created_at DESC
            """, (project_id,))
            return [
                {
                    "id": str(row["id"]),
                    "decision_summary": row["decision_summary"],
                    "options_considered": row["options_considered"],
                    "chosen_option": row["chosen_option"],
                    "rationale": row["rationale"],
                    "pas_node_id": str(row["pas_node_id"]) if row["pas_node_id"] else None,
                    "phase_ids": [str(p) for p in row["phase_ids"]] if row["phase_ids"] else [],
                    "created_at": str(row["created_at"])
                }
                for row in cur.fetchall()
            ]
    finally:
        safe_close_connection(conn)


def update_phase_dual_recommendation(
    phase_id: str,
    balanced: dict,
    aspirational: Optional[dict] = None
) -> bool:
    """
    Update dual_recommendation for a roadmap phase.
    
    Args:
        phase_id: Phase UUID
        balanced: The balanced recommendation option
        aspirational: Optional aspirational recommendation option
        
    Returns:
        True if update succeeded
    """
    import json
    
    conn = get_db_connection()
    try:
        dual_rec = {
            "balanced": balanced,
            "aspirational": aspirational
        }
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE roadmap_phases
                SET dual_recommendation = %s, updated_at = NOW()
                WHERE id = %s
            """, (json.dumps(dual_rec), phase_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        safe_close_connection(conn)


def store_research_findings(
    artifact_id: str,
    findings: list,
    confidence_level: str = "medium"
) -> bool:
    """
    Store structured findings for a research artifact.
    
    Args:
        artifact_id: Artifact UUID (must be type='research')
        findings: List of finding dicts with source, type, text
        confidence_level: Overall confidence (high/medium/low)
        
    Returns:
        True if update succeeded
    """
    import json
    
    conn = get_db_connection()
    try:
        findings_data = {
            "findings": findings,
            "confidence_level": confidence_level
        }
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE artifacts
                SET findings_data = %s
                WHERE id = %s
            """, (json.dumps(findings_data), artifact_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        safe_close_connection(conn)


# ============================================================================
# v90: Implementation Plan Functions
# ============================================================================

def add_plan_step(
    artifact_id: str,
    content: str,
    step_order: int,
    pas_node_id: Optional[str] = None,
    status: str = "pending"
) -> str:
    """
    Add a step to an implementation plan.
    
    Args:
        artifact_id: UUID of the implementation_plan artifact
        content: Step description
        step_order: Order in the plan (1-indexed)
        pas_node_id: Optional link to PAS thought node
        status: pending/in_progress/done/skipped
        
    Returns:
        Created step ID
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO plan_steps (artifact_id, content, step_order, pas_node_id, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (artifact_id, content, step_order, pas_node_id, status))
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def update_step_status(step_id: str, status: str) -> bool:
    """
    Update the status of a plan step.
    
    Args:
        step_id: UUID of the step
        status: pending/in_progress/done/skipped
        
    Returns:
        True if update succeeded
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE plan_steps
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, step_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        safe_close_connection(conn)


def get_plan_steps(artifact_id: str) -> list:
    """
    Get all steps for an implementation plan.
    
    Args:
        artifact_id: UUID of the implementation_plan artifact
        
    Returns:
        List of steps ordered by step_order
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, step_order, content, status, pas_node_id, created_at, updated_at
                FROM plan_steps
                WHERE artifact_id = %s
                ORDER BY step_order
            """, (artifact_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        safe_close_connection(conn)


def add_plan_scope(
    artifact_id: str,
    file_path: str,
    change_type: str,
    lsp_refs: Optional[list] = None
) -> str:
    """
    Add a scope declaration to an implementation plan.
    
    Args:
        artifact_id: UUID of the implementation_plan artifact
        file_path: Path to the file
        change_type: modify/create/delete
        lsp_refs: Optional LSP reference data [{symbol, locations, count}]
        
    Returns:
        Created scope ID
    """
    import json
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO plan_scope (artifact_id, file_path, change_type, lsp_refs)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (artifact_id, file_path, change_type, json.dumps(lsp_refs or [])))
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def get_plan_scope(artifact_id: str) -> list:
    """
    Get all scope declarations for an implementation plan.
    
    Args:
        artifact_id: UUID of the implementation_plan artifact
        
    Returns:
        List of scope declarations
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, change_type, lsp_refs, created_at
                FROM plan_scope
                WHERE artifact_id = %s
                ORDER BY change_type, file_path
            """, (artifact_id,))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        safe_close_connection(conn)


def link_step_to_critique(
    step_id: str,
    thought_node_id: str,
    resolution_note: Optional[str] = None
) -> str:
    """
    Link a plan step to a PAS critique it addresses.
    
    Args:
        step_id: UUID of the plan step
        thought_node_id: UUID of the thought node (critique)
        resolution_note: Optional note on how critique is resolved
        
    Returns:
        Created link ID
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO plan_step_critiques (step_id, thought_node_id, resolution_note)
                VALUES (%s, %s, %s)
                ON CONFLICT (step_id, thought_node_id) DO UPDATE
                SET resolution_note = EXCLUDED.resolution_note
                RETURNING id
            """, (step_id, thought_node_id, resolution_note))
            row = cur.fetchone()
            conn.commit()
            return str(row["id"])
    finally:
        safe_close_connection(conn)


def get_unaddressed_critiques(session_id: str, artifact_id: str) -> list:
    """
    Find critiques from session not linked to any plan step.
    
    Args:
        session_id: PAS session UUID
        artifact_id: Implementation plan artifact UUID
        
    Returns:
        List of unaddressed critique nodes
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tn.id, tn.content, tn.path
                FROM thought_nodes tn
                WHERE tn.session_id = %s
                AND tn.critique_data IS NOT NULL
                AND tn.id NOT IN (
                    SELECT psc.thought_node_id
                    FROM plan_step_critiques psc
                    JOIN plan_steps ps ON psc.step_id = ps.id
                    WHERE ps.artifact_id = %s
                )
            """, (session_id, artifact_id))
            rows = cur.fetchall()
            return [dict(row) for row in rows]
    finally:
        safe_close_connection(conn)


def update_plan_checklist(artifact_id: str, checklist: list) -> bool:
    """
    Update the pre-submission checklist for an implementation plan.
    
    Args:
        artifact_id: UUID of the implementation_plan artifact
        checklist: List of checklist items [{item, checked, note}]
        
    Returns:
        True if update succeeded
    """
    import json
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE artifacts
                SET checklist_data = %s
                WHERE id = %s
            """, (json.dumps(checklist), artifact_id))
            conn.commit()
            return cur.rowcount > 0
    finally:
        safe_close_connection(conn)

