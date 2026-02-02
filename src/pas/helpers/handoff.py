"""
Session Handoff Helpers - Phase 12

Functions for creating, retrieving, and managing session handoffs.
"""

from typing import Any, Optional
import json


def create_handoff_record(
    conn,
    session_id: str,
    summary: str,
    project_id: Optional[str] = None,
    next_task: Optional[str] = None,
    context: Optional[dict] = None,
    linked_artifacts: Optional[list] = None,
    linked_sessions: Optional[list] = None
) -> dict[str, Any]:
    """Create a new handoff record with embedded summary."""
    from pas.utils import get_embedding
    
    # Validate session exists
    cur = conn.cursor()
    cur.execute(
        "SELECT id, project_id FROM reasoning_sessions WHERE id = %s",
        (session_id,)
    )
    session = cur.fetchone()
    if not session:
        return {"success": False, "error": f"Session {session_id} not found"}
    
    # Use session's project_id if not provided
    if not project_id:
        project_id = session.get("project_id") if hasattr(session, 'get') else session[1]
    
    # Generate embedding for semantic search
    summary_embedding = get_embedding(summary)
    
    # Insert handoff record
    cur.execute("""
        INSERT INTO session_handoffs 
        (session_id, project_id, summary, summary_embedding, next_task, 
         context, linked_artifacts, linked_sessions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, created_at
    """, (
        session_id,
        project_id,
        summary,
        summary_embedding,
        next_task,
        json.dumps(context or {}),
        linked_artifacts or [],
        linked_sessions or []
    ))
    
    result = cur.fetchone()
    conn.commit()
    
    return {
        "success": True,
        "handoff_id": str(result[0]),
        "created_at": result[1].isoformat(),
        "project_id": project_id
    }


def list_active_handoffs(
    conn,
    project_id: Optional[str] = None,
    limit: int = 5
) -> list[dict]:
    """List active (unprocessed) handoffs, optionally filtered by project."""
    cur = conn.cursor()
    
    if project_id:
        cur.execute("""
            SELECT id, session_id, project_id, summary, next_task, 
                   linked_artifacts, created_at
            FROM session_handoffs
            WHERE status = 'active' AND project_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (project_id, limit))
    else:
        cur.execute("""
            SELECT id, session_id, project_id, summary, next_task,
                   linked_artifacts, created_at
            FROM session_handoffs
            WHERE status = 'active'
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
    
    rows = cur.fetchall()
    return [
        {
            "handoff_id": str(row[0]),
            "session_id": str(row[1]) if row[1] else None,
            "project_id": row[2],
            "summary": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
            "next_task": row[4],
            "linked_artifacts": row[5],
            "created_at": row[6].isoformat()
        }
        for row in rows
    ]


def search_handoffs(
    conn,
    query: str,
    project_id: Optional[str] = None,
    limit: int = 5
) -> list[dict]:
    """Semantic search for handoffs by topic."""
    from pas.utils import get_embedding
    
    query_embedding = get_embedding(query)
    cur = conn.cursor()
    
    if project_id:
        cur.execute("""
            SELECT id, session_id, project_id, summary, next_task,
                   linked_artifacts, context, created_at, status,
                   1 - (summary_embedding <=> %s::vector) as similarity
            FROM session_handoffs
            WHERE project_id = %s
            ORDER BY summary_embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, project_id, query_embedding, limit))
    else:
        cur.execute("""
            SELECT id, session_id, project_id, summary, next_task,
                   linked_artifacts, context, created_at, status,
                   1 - (summary_embedding <=> %s::vector) as similarity
            FROM session_handoffs
            ORDER BY summary_embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))
    
    rows = cur.fetchall()
    return [
        {
            "handoff_id": str(row[0]),
            "session_id": str(row[1]) if row[1] else None,
            "project_id": row[2],
            "summary": row[3],
            "next_task": row[4],
            "linked_artifacts": row[5],
            "context": row[6],
            "created_at": row[7].isoformat(),
            "status": row[8],
            "similarity": round(row[9], 4)
        }
        for row in rows
    ]


def get_handoff_by_id(conn, handoff_id: str) -> Optional[dict]:
    """Get a specific handoff by ID."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, session_id, project_id, summary, next_task,
               linked_artifacts, linked_sessions, context, created_at, 
               status, processed_at
        FROM session_handoffs
        WHERE id = %s
    """, (handoff_id,))
    
    row = cur.fetchone()
    if not row:
        return None
    
    return {
        "handoff_id": str(row[0]),
        "session_id": str(row[1]) if row[1] else None,
        "project_id": row[2],
        "summary": row[3],
        "next_task": row[4],
        "linked_artifacts": row[5],
        "linked_sessions": [str(s) for s in row[6]] if row[6] else [],
        "context": row[7],
        "created_at": row[8].isoformat(),
        "status": row[9],
        "processed_at": row[10].isoformat() if row[10] else None
    }


def mark_handoff_processed(conn, handoff_id: str) -> dict:
    """Mark a handoff as processed after onboarding."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE session_handoffs
        SET status = 'processed', processed_at = NOW()
        WHERE id = %s
        RETURNING id, status, processed_at
    """, (handoff_id,))
    
    result = cur.fetchone()
    if not result:
        return {"success": False, "error": f"Handoff {handoff_id} not found"}
    
    conn.commit()
    return {
        "success": True,
        "handoff_id": str(result[0]),
        "status": result[1],
        "processed_at": result[2].isoformat()
    }
