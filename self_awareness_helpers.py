"""
PAS Self-Awareness Helpers (v40 Phase 4)

Enables PAS to understand its own architecture:
- Schema introspection via information_schema
- Tool capability registry via FastMCP internals
- Session statistics from outcome_records
- Static architecture map of data flows
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Static Architecture Map
# =============================================================================

ARCHITECTURE_MAP = {
    "reasoning_flow": {
        "description": "Core reasoning process",
        "stages": [
            {"name": "session", "tool": "start_reasoning_session"},
            {"name": "expansion", "tools": ["prepare_expansion", "store_expansion"]},
            {"name": "critique", "tools": ["prepare_critique", "store_critique"]},
            {"name": "finalize", "tool": "finalize_session"},
            {"name": "outcome", "tool": "record_outcome"}
        ]
    },
    "learning_flow": {
        "description": "Self-learning from outcomes",
        "stages": [
            {"name": "record", "table": "outcome_records"},
            {"name": "calibrate", "table": "calibration_records"},
            {"name": "refresh", "tool": "refresh_law_weights"},
            {"name": "boost", "table": "scientific_laws.success_count"}
        ]
    },
    "codebase_flow": {
        "description": "Code navigation and understanding",
        "stages": [
            {"name": "sync", "tool": "sync_project"},
            {"name": "index", "tables": ["file_registry", "symbol_index"]},
            {"name": "query", "tool": "query_codebase"},
            {"name": "navigate", "tools": ["find_references", "go_to_definition"]}
        ]
    },
    "metacognitive_flow": {
        "description": "5-stage metacognitive prompting (v40)",
        "stages": [
            {"stage": 1, "name": "Understanding"},
            {"stage": 2, "name": "Preliminary Judgment"},
            {"stage": 3, "name": "Critical Evaluation"},
            {"stage": 4, "name": "Final Decision"},
            {"stage": 5, "name": "Confidence Expression"}
        ]
    }
}


# =============================================================================
# Schema Introspection
# =============================================================================

def get_schema_info(conn) -> Dict[str, Any]:
    """
    Query PostgreSQL information_schema for PAS tables.
    
    Args:
        conn: Database connection
        
    Returns:
        Dict with tables, columns, and relationships
    """
    cur = conn.cursor()
    
    try:
        # Get all PAS tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = [row["table_name"] for row in cur.fetchall()]
        
        # Get columns for each table
        table_info = {}
        for table in tables:
            cur.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table,))
            table_info[table] = [
                {"column": r["column_name"], "type": r["data_type"], "nullable": r["is_nullable"] == "YES"}
                for r in cur.fetchall()
            ]
        
        # Get foreign key relationships
        cur.execute("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu 
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)
        relationships = [
            {
                "from_table": r["table_name"],
                "from_column": r["column_name"],
                "to_table": r["foreign_table"],
                "to_column": r["foreign_column"]
            }
            for r in cur.fetchall()
        ]
        
        return {
            "table_count": len(tables),
            "tables": table_info,
            "relationships": relationships
        }
        
    except Exception as e:
        logger.error(f"Schema introspection failed: {e}")
        return {"error": str(e)}


# =============================================================================
# Tool Registry Introspection
# =============================================================================

def get_tool_registry(mcp_instance) -> List[Dict[str, Any]]:
    """
    Extract registered tools from FastMCP instance.
    
    Args:
        mcp_instance: The FastMCP server instance
        
    Returns:
        List of tool info dicts with name, description, parameters
    """
    tools = []
    
    try:
        # Access FastMCP's tool manager via list_tools()
        if hasattr(mcp_instance, '_tool_manager'):
            tool_manager = mcp_instance._tool_manager
            if hasattr(tool_manager, 'list_tools'):
                for tool in tool_manager.list_tools():
                    # Extract category from docstring/description if present
                    docstring = tool.description or ""
                    category = "unknown"
                    if "reasoning" in docstring.lower() or "session" in docstring.lower():
                        category = "reasoning"
                    elif "codebase" in docstring.lower() or "project" in docstring.lower():
                        category = "codebase"
                    elif "calibration" in docstring.lower():
                        category = "calibration"
                    elif "learning" in docstring.lower() or "outcome" in docstring.lower():
                        category = "learning"
                    elif "interview" in docstring.lower() or "question" in docstring.lower():
                        category = "interview"
                    elif "purpose" in docstring.lower():
                        category = "purpose"
                    elif "metacognitive" in docstring.lower() or "stage" in docstring.lower():
                        category = "metacognitive"
                    
                    tools.append({
                        "name": tool.name,
                        "description": docstring.split("\n")[0].strip() if docstring else None,
                        "category": category
                    })
        
        return sorted(tools, key=lambda t: t["name"])
        
    except Exception as e:
        logger.error(f"Tool registry introspection failed: {e}")
        return [{"error": str(e)}]


# =============================================================================
# Session Statistics
# =============================================================================

def get_session_statistics(conn, days: int = 30) -> Dict[str, Any]:
    """
    Compute aggregate statistics about past sessions.
    
    Args:
        conn: Database connection
        days: Number of days to look back
        
    Returns:
        Dict with success rate, patterns, law effectiveness
    """
    cur = conn.cursor()
    
    try:
        # Total sessions
        cur.execute("SELECT COUNT(*) FROM reasoning_sessions")
        total_sessions = cur.fetchone()["count"]
        
        # Recent outcomes
        cur.execute("""
            SELECT outcome, COUNT(*) 
            FROM outcome_records 
            WHERE created_at > NOW() - INTERVAL '%s days'
            GROUP BY outcome
        """, (days,))
        outcomes = {row["outcome"]: row["count"] for row in cur.fetchall()}
        
        total_outcomes = sum(outcomes.values())
        success_rate = outcomes.get("success", 0) / total_outcomes if total_outcomes > 0 else 0
        
        # Average nodes per session
        cur.execute("""
            SELECT AVG(node_count)
            FROM (
                SELECT session_id, COUNT(*) as node_count 
                FROM thought_nodes 
                GROUP BY session_id
            ) sub
        """)
        avg_nodes = cur.fetchone()["avg"] or 0
        
        # Top failure reasons (semantic patterns)
        cur.execute("""
            SELECT failure_reason, COUNT(*) as cnt
            FROM outcome_records
            WHERE outcome = 'failure' AND failure_reason IS NOT NULL
            GROUP BY failure_reason
            ORDER BY cnt DESC
            LIMIT 5
        """)
        top_failures = [{"reason": r["failure_reason"], "count": r["cnt"]} for r in cur.fetchall()]
        
        # Law effectiveness (success-weighted)
        cur.execute("""
            SELECT sl.law_name, sl.success_count, sl.selection_count
            FROM scientific_laws sl
            WHERE sl.selection_count > 0
            ORDER BY (CAST(sl.success_count AS FLOAT) / sl.selection_count) DESC
            LIMIT 10
        """)
        law_effectiveness = [
            {
                "law": r["law_name"],
                "success_count": r["success_count"],
                "selection_count": r["selection_count"],
                "rate": round(r["success_count"] / r["selection_count"], 3) if r["selection_count"] > 0 else 0
            }
            for r in cur.fetchall()
        ]
        
        return {
            "total_sessions": total_sessions,
            "recent_period_days": days,
            "outcomes": outcomes,
            "success_rate": round(success_rate, 3),
            "avg_nodes_per_session": round(float(avg_nodes), 2),
            "top_failure_patterns": top_failures,
            "law_effectiveness": law_effectiveness
        }
        
    except Exception as e:
        logger.error(f"Session statistics failed: {e}")
        return {"error": str(e)}
