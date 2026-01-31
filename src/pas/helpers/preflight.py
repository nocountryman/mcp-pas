# ============================================================================
# Preflight Enforcement Helpers (v41)
# Session call tracking and preflight verification for implementation planning
# ============================================================================

import logging
import re
from typing import Optional

logger = logging.getLogger("pas-server")

# SQL keywords that suggest database operations
SQL_KEYWORDS = {
    "CREATE", "ALTER", "DROP", "SELECT", "INSERT", "UPDATE", "DELETE",
    "TABLE", "COLUMN", "INDEX", "CONSTRAINT", "FOREIGN KEY", "PRIMARY KEY",
    "SCHEMA", "DATABASE", "TRIGGER", "FUNCTION", "VIEW"
}


def ensure_session_call_log_table(cur) -> None:
    """Lazy migration: create session_call_log table if not exists."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_call_log (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id      UUID REFERENCES reasoning_sessions(id) ON DELETE CASCADE,
            tool_name       TEXT NOT NULL,
            call_metadata   JSONB DEFAULT '{}',
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_call_log_session
            ON session_call_log(session_id, tool_name)
    """)


def log_tool_call(cur, session_id: str, tool_name: str, metadata: Optional[dict] = None) -> None:
    """Log a tool call for preflight tracking."""
    import json
    ensure_session_call_log_table(cur)
    cur.execute(
        """
        INSERT INTO session_call_log (session_id, tool_name, call_metadata)
        VALUES (%s, %s, %s)
        """,
        (session_id, tool_name, json.dumps(metadata or {}))
    )


def get_session_calls(cur, session_id: str) -> list[str]:
    """Get list of tool names called in this session."""
    ensure_session_call_log_table(cur)
    cur.execute(
        """
        SELECT DISTINCT tool_name FROM session_call_log
        WHERE session_id = %s
        """,
        (session_id,)
    )
    return [row["tool_name"] for row in cur.fetchall()]


def detect_sql_operations(text: str) -> bool:
    """Check if text contains SQL-related keywords suggesting DB operations."""
    if not text:
        return False
    upper_text = text.upper()
    return any(keyword in upper_text for keyword in SQL_KEYWORDS)


def check_preflight_conditions(
    cur,
    session_id: str,
    has_suggested_lookups: bool = False,  # Changed: boolean flag instead of list
    schema_check_required: bool = False,
    has_failure_warnings: bool = False,  # Changed: boolean flag instead of list
    has_project_id: bool = False  # v42a: Was project_id provided to prepare_expansion?
) -> list[dict]:
    """
    Check if required preflight steps were performed.
    Returns list of warnings for any missing steps.
    """
    warnings = []
    session_calls = get_session_calls(cur, session_id)
    
    # Check 1: If suggested_lookups were present, did they call find_references?
    if has_suggested_lookups and "find_references" not in session_calls:
        warnings.append({
            "type": "missing_find_references",
            "message": "suggested_lookups present but find_references not called",
            "suggestion": "Call find_references(project_id='...', symbol_name='...') for suggested symbols"
        })
    
    # Check 2: If schema_check_required, did they verify schema?
    if schema_check_required:
        schema_tools = {"get_self_awareness", "query_codebase"}
        if not any(tool in session_calls for tool in schema_tools):
            warnings.append({
                "type": "missing_schema_check",
                "message": "SQL operations detected but no schema verification performed",
                "suggestion": "Call get_self_awareness() or verify schema before writing SQL queries"
            })
    
    # Check 3: If past_failure_warnings were present, did they acknowledge?
    if has_failure_warnings and "log_conversation" not in session_calls:
        warnings.append({
            "type": "unacknowledged_warnings",
            "message": "Past failure warnings present but not acknowledged",
            "suggestion": "Call log_conversation() to acknowledge past failure warnings"
        })
    
    # Check 4 (v42a): If project_id provided, did they research existing codebase?
    if has_project_id:
        research_tools = {"query_codebase", "find_references", "infer_file_purpose", "infer_module_purpose"}
        if not any(tool in session_calls for tool in research_tools):
            warnings.append({
                "type": "missing_codebase_research",
                "message": "project_id provided but no codebase research performed",
                "suggestion": "Call query_codebase(query='<goal keywords>', project_id='...') to find existing related functionality"
            })
    
    return warnings


# ============================================================================
# Raw Input Logging Enforcement (v44)
# Ensures verbatim user input is captured for psychological analysis
# ============================================================================

# Keywords indicating user-initiated session (vs LLM-initiated)
USER_INITIATED_KEYWORDS = {
    'user wants', 'user asked', 'user requested', 'requested', 
    'build', 'implement', 'design', 'create', 'add', 'fix',
    'user feedback', 'user said', 'user:', 'the user'
}


def check_raw_input_required(user_goal: str, raw_input: Optional[str]) -> Optional[dict]:
    """
    Check if raw_input should be present but isn't.
    
    Returns warning dict if session appears user-initiated but raw_input missing.
    Returns None if check passes.
    """
    if raw_input:
        return None  # Raw input provided, all good
    
    if not user_goal:
        return None  # Empty goal, nothing to check
    
    goal_lower = user_goal.lower()
    
    # Check for user-initiated keywords
    for keyword in USER_INITIATED_KEYWORDS:
        if keyword in goal_lower:
            return {
                "type": "missing_raw_input",
                "message": f"Session appears user-initiated ('{keyword}' in goal) but raw_input not provided",
                "suggestion": "Pass raw_input='<verbatim user prompt>' or skip_raw_input_check=True if LLM-initiated",
                "detected_keyword": keyword
            }
    
    return None


