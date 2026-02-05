"""
Cross-Project Learning: Pattern Extraction

Extracts generalized, project-agnostic patterns from project-specific failures.
Enables lessons learned in one project to apply across all projects.
"""
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger("pas-server")

# Heuristic pattern rules: keyword -> (pattern_type, generalized_description)
PATTERN_RULES = {
    # API/Network patterns
    "timeout": ("api_timeout", "API timeout for large responses - look for paginated or streaming endpoint variant"),
    "times out": ("api_timeout", "API timeout for large responses - look for paginated or streaming endpoint variant"),
    "connection refused": ("connection_refused", "Service connection refused - verify service is running and port is correct"),
    "connection error": ("connection_stale", "Connection state cached but stale - add reconnection logic on failure"),
    "port": ("dynamic_port", "Dynamic port discovery - service port changes on restart, discover via process inspection"),

    
    # Import/Module patterns  
    "import": ("import_error", "Import path incorrect - verify module location with grep_search before importing"),
    "circular import": ("circular_import", "Circular import detected - restructure to avoid bidirectional dependencies"),
    "module not found": ("module_missing", "Module not found - check virtual environment activation and package installation"),
    
    # Database/Schema patterns
    "column": ("schema_mismatch", "Column name assumed incorrectly - verify schema with psql or information_schema before querying"),
    "table": ("schema_mismatch", "Table structure assumed incorrectly - query schema first, don't assume from memory"),
    "constraint": ("constraint_violation", "Database constraint violated - check NOT NULL, UNIQUE, CHECK constraints"),
    "RealDictRow": ("dict_access", "RealDictRow from psycopg2 uses dict-style access ['column'], not index [0]"),
    
    # Type/Data patterns
    "type error": ("type_mismatch", "Type mismatch - verify expected types before operations"),
    "none": ("null_handling", "NoneType error - add null checks for optional values"),
    "key error": ("dict_key_missing", "Dictionary key missing - use .get() with default or check key existence"),
    
    # Async/Threading patterns
    "async": ("async_boundary", "Async/sync boundary issue - ensure proper await or sync wrapper"),
    "coroutine": ("async_boundary", "Coroutine not awaited - add await or use asyncio.run()"),
    
    # Singleton/State patterns
    "singleton": ("stale_singleton", "Singleton holds stale state - add invalidation or refresh mechanism"),
    "cached": ("stale_cache", "Cached value stale - add cache invalidation or TTL"),
}


def extract_generalized_pattern(
    failure_reason: str,
    notes: Optional[str] = None
) -> Dict[str, str]:
    """
    Extract a generalized, project-agnostic pattern from a specific failure.
    
    Args:
        failure_reason: The project-specific failure description
        notes: Optional additional context
        
    Returns:
        {
            "pattern_type": "api_timeout",
            "generalized_pattern": "API timeout for large responses..."
        }
    """
    if not failure_reason:
        return {"pattern_type": "unknown", "generalized_pattern": ""}
    
    text_to_search = f"{failure_reason} {notes or ''}".lower()
    
    # Check each pattern rule
    for keyword, (pattern_type, generalized) in PATTERN_RULES.items():
        if keyword.lower() in text_to_search:
            logger.debug(f"Pattern matched: {keyword} -> {pattern_type}")
            return {
                "pattern_type": pattern_type,
                "generalized_pattern": generalized
            }
    
    # No match - return unknown with the original as description
    # This preserves the failure for semantic search even without classification
    return {
        "pattern_type": "unclassified",
        "generalized_pattern": _extract_action_pattern(failure_reason)
    }


def _extract_action_pattern(text: str) -> str:
    """
    Try to extract an actionable pattern from unclassified text.
    Looks for common patterns like "X -> Y" or "use X instead of Y".
    """
    # Look for explicit action patterns
    action_patterns = [
        r"use\s+(\w+)\s+instead\s+of\s+(\w+)",  # "use X instead of Y"
        r"(\w+)\s*->\s*(\w+)",                   # "X -> Y"
        r"should\s+be\s+(.+?)(?:\.|$)",          # "should be X"
        r"must\s+(.+?)(?:\.|$)",                 # "must X"
    ]
    
    for pattern in action_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text  # Keep the original as it contains actionable info
    
    # Truncate long descriptions
    if len(text) > 200:
        return text[:200] + "..."
    
    return text


def get_pattern_types() -> list[str]:
    """Return all known pattern types for categorization."""
    return sorted(set(pt for pt, _ in PATTERN_RULES.values()))
