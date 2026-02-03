"""
Session Handoff Helpers - Phase 12

Functions for creating, retrieving, and managing session handoffs.

Modes:
- new: Create new handoff, archive any existing active ones for same session
- update: Update most recent active handoff for session (upsert)
- list: List active handoffs
- restore: Get handoff by ID, auto-mark processed if different session
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
    """Create a new handoff record, archiving any existing active ones for this session."""
    from pas.utils import get_embedding
    
    # Validate session exists
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM reasoning_sessions WHERE id = %s",
        (session_id,)
    )
    session = cur.fetchone()
    if not session:
        return {"success": False, "error": f"Session {session_id} not found"}
    
    # Archive any existing active handoffs for this PROJECT (singleton per project)
    cur.execute("""
        UPDATE session_handoffs
        SET status = 'archived'
        WHERE project_id = %s AND status = 'active'
        RETURNING id
    """, (project_id,))
    archived = cur.fetchall()
    archived_count = len(archived)
    
    # Generate embedding for semantic search
    summary_embedding = get_embedding(summary)
    
    # Insert new handoff record
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
        "handoff_id": str(result["id"]),
        "created_at": result["created_at"].isoformat(),
        "project_id": project_id,
        "archived_previous": archived_count
    }


def update_handoff_record(
    conn,
    session_id: str,
    summary: Optional[str] = None,
    project_id: Optional[str] = None,
    next_task: Optional[str] = None,
    context: Optional[dict] = None,
    linked_artifacts: Optional[list] = None,
    linked_sessions: Optional[list] = None
) -> dict[str, Any]:
    """Update the most recent active handoff for this session (upsert behavior)."""
    from pas.utils import get_embedding
    
    cur = conn.cursor()
    
    # Find most recent active handoff for this session
    cur.execute("""
        SELECT id, summary, project_id, next_task, context, 
               linked_artifacts, linked_sessions
        FROM session_handoffs
        WHERE session_id = %s AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
    """, (session_id,))
    
    existing = cur.fetchone()
    
    if not existing:
        # No existing handoff - create new one (requires summary)
        if not summary:
            return {"success": False, "error": "No active handoff for this session. Use 'new' mode with summary."}
        return create_handoff_record(
            conn, session_id, summary, project_id, next_task, 
            context, linked_artifacts, linked_sessions
        )
    
    # Build update fields (only update what's provided)
    updates = []
    params = []
    
    if summary is not None:
        updates.append("summary = %s")
        params.append(summary)
        updates.append("summary_embedding = %s")
        params.append(get_embedding(summary))
    
    if project_id is not None:
        updates.append("project_id = %s")
        params.append(project_id)
    
    if next_task is not None:
        updates.append("next_task = %s")
        params.append(next_task)
    
    if context is not None:
        updates.append("context = %s")
        params.append(json.dumps(context))
    
    if linked_artifacts is not None:
        updates.append("linked_artifacts = %s")
        params.append(linked_artifacts)
    
    if linked_sessions is not None:
        updates.append("linked_sessions = %s")
        params.append(linked_sessions)
    
    if not updates:
        return {"success": False, "error": "No fields to update"}
    
    # Execute update
    params.append(existing["id"])
    cur.execute(f"""
        UPDATE session_handoffs
        SET {", ".join(updates)}
        WHERE id = %s
        RETURNING id, created_at
    """, tuple(params))
    
    result = cur.fetchone()
    conn.commit()
    
    return {
        "success": True,
        "handoff_id": str(result["id"]),
        "updated_at": result["created_at"].isoformat(),
        "mode": "updated"
    }


def list_active_handoffs(
    conn,
    project_id: Optional[str] = None,
    limit: int = 10
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
            "handoff_id": str(row["id"]),
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "project_id": row["project_id"],
            "summary": row["summary"][:200] + "..." if len(row["summary"]) > 200 else row["summary"],
            "next_task": row["next_task"],
            "linked_artifacts": row["linked_artifacts"],
            "created_at": row["created_at"].isoformat()
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
            "handoff_id": str(row["id"]),
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "project_id": row["project_id"],
            "summary": row["summary"],
            "next_task": row["next_task"],
            "linked_artifacts": row["linked_artifacts"],
            "context": row["context"],
            "created_at": row["created_at"].isoformat(),
            "status": row["status"],
            "similarity": round(row["similarity"], 4)
        }
        for row in rows
    ]


def get_active_handoff_for_project(conn, project_id: str) -> Optional[dict]:
    """Get THE active handoff for a project (singleton)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, session_id, project_id, summary, next_task,
               linked_artifacts, linked_sessions, context, created_at, 
               status, processed_at
        FROM session_handoffs
        WHERE project_id = %s AND status = 'active'
        ORDER BY created_at DESC
        LIMIT 1
    """, (project_id,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "handoff_id": str(row["id"]),
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "project_id": row["project_id"],
        "summary": row["summary"],
        "next_task": row["next_task"],
        "linked_artifacts": row["linked_artifacts"],
        "linked_sessions": [str(s) for s in row["linked_sessions"]] if row["linked_sessions"] else [],
        "context": row["context"],
        "created_at": row["created_at"].isoformat(),
        "status": row["status"],
        "processed_at": row["processed_at"].isoformat() if row["processed_at"] else None
    }


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
        "handoff_id": str(row["id"]),
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "project_id": row["project_id"],
        "summary": row["summary"],
        "next_task": row["next_task"],
        "linked_artifacts": row["linked_artifacts"],
        "linked_sessions": [str(s) for s in row["linked_sessions"]] if row["linked_sessions"] else [],
        "context": row["context"],
        "created_at": row["created_at"].isoformat(),
        "status": row["status"],
        "processed_at": row["processed_at"].isoformat() if row["processed_at"] else None
    }


def restore_handoff(
    conn, 
    handoff_id: str, 
    current_session_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Restore a handoff. If current_session_id differs from handoff's session,
    auto-mark as processed (consumed).
    """
    handoff = get_handoff_by_id(conn, handoff_id)
    if not handoff:
        return {"success": False, "error": f"Handoff {handoff_id} not found"}
    
    # Check if we should auto-mark as processed
    auto_processed = False
    if current_session_id and handoff["session_id"] != current_session_id:
        if handoff["status"] == "active":
            mark_handoff_processed(conn, handoff_id)
            handoff["status"] = "processed"
            auto_processed = True
    
    return {
        "success": True,
        "handoff": handoff,
        "auto_processed": auto_processed,
        "message": "Handoff restored and marked as processed" if auto_processed else "Handoff restored"
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
        "handoff_id": str(result["id"]),
        "status": result["status"],
        "processed_at": result["processed_at"].isoformat()
    }
