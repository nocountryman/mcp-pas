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
