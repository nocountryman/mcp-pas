"""
PAS Critique Helper Functions

Pure functions for critique preparation, including node/law fetching,
prompt construction, and past failure/critique searching.
"""

import logging
from typing import Any, Optional

from pas.utils import get_embedding, get_db_connection, safe_close_connection

logger = logging.getLogger(__name__)


# =============================================================================
# Node & Law Fetching
# =============================================================================

def fetch_node_with_laws(cur, node_id: str) -> tuple[Optional[dict], list[dict]]:
    """
    Fetch a thought node and its supporting laws.
    
    Args:
        cur: Database cursor
        node_id: The thought node UUID
        
    Returns:
        Tuple of (node_dict, laws_list) or (None, []) if not found
    """
    cur.execute(
        """
        SELECT t.id, t.session_id, t.path, t.content, t.prior_score, 
               t.likelihood, t.posterior_score, t.supporting_laws, s.goal
        FROM thought_nodes t 
        JOIN reasoning_sessions s ON t.session_id = s.id 
        WHERE t.id = %s
        """,
        (node_id,)
    )
    node = cur.fetchone()
    if not node:
        return None, []
    
    # Get supporting laws
    laws_text = []
    if node["supporting_laws"]:
        cur.execute(
            "SELECT law_name, definition, failure_modes FROM scientific_laws WHERE id = ANY(%s)",
            (node["supporting_laws"],)
        )
        for law in cur.fetchall():
            laws_text.append({
                "law_name": law["law_name"],
                "definition": law["definition"],
                "failure_modes": law["failure_modes"] or []
            })
    
    return node, laws_text


# =============================================================================
# Critique Prompt Building
# =============================================================================

def build_critique_prompt(
    node_content: str,
    session_goal: str,
    laws_text: list[dict],
    critique_mode: str = "standard"
) -> tuple[str, str, str]:
    """
    Build critique prompt based on critique mode.
    
    Args:
        node_content: The hypothesis text to critique
        session_goal: The session's goal
        laws_text: List of supporting law dicts
        critique_mode: 'standard' or 'negative_space'
        
    Returns:
        Tuple of (prompt, system_message, expected_format)
    """
    if critique_mode == "negative_space":
        prompt = f"""Analyze what this hypothesis does NOT address:

Hypothesis: {node_content}

Session Goal: {session_goal}

For EACH component of the goal, answer:
1. Is this component fully addressed by the hypothesis?
2. What aspects are MISSING or assumed?
3. What adjacent concerns are NOT considered?

Return ONLY a valid JSON object:
{{
  "gaps": [
    {{"component": "...", "addressed": true/false, "missing": "what's not covered"}},
    ...
  ],
  "blind_spots": ["things taken for granted"],
  "boundary_issues": ["where the hypothesis ends but the goal continues"],
  "overall_coverage": 0.8
}}

Focus on OMISSIONS, not flaws in what IS proposed."""
        system = "You are a gap analyst. Find what's MISSING, not what's wrong. Return only valid JSON."
        expected_format = "JSON object with gaps/blind_spots/boundary_issues"
    else:
        # Standard critique mode
        law_names = ', '.join(l.get('law_name', '') for l in laws_text) if laws_text else 'None'
        prompt = f"""Critique this hypothesis:
{node_content}

Session Goal: {session_goal}

Consider these scientific laws: {law_names}

Return ONLY a valid JSON object:
{{
  "counterargument": "Main counter-argument that challenges this hypothesis",
  "severity": 0.5,
  "major_flaws": ["flaw1", "flaw2"],
  "minor_flaws": ["minor issue"],
  "edge_cases": ["edge case 1"]
}}

Be specific and constructive. Focus on what could go wrong."""
        system = "You are a hypothesis critic. Return only valid JSON."
        expected_format = "JSON object with counterargument/severity/flaws"
    
    return prompt, system, expected_format


# =============================================================================
# Past Failure/Critique Search
# =============================================================================

def _query_keyword_failure_patterns(text: str) -> list[dict]:
    """
    Query failure_patterns table for keyword matches.
    
    Args:
        text: Text to match against keywords
        
    Returns:
        List of matching pattern warnings
    """
    warnings = []
    text_lower = text.lower()
    seen_patterns: set[str] = set()
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT pattern_name, keywords, warning_text
            FROM failure_patterns
            WHERE enabled = true
        """)
        
        for row in cur.fetchall():
            pattern_name = row['pattern_name']
            keywords = row['keywords']  # TEXT[] array
            warning_text = row['warning_text']
            
            # Check if any keyword matches
            for keyword in keywords:
                if keyword.lower() in text_lower and pattern_name not in seen_patterns:
                    warnings.append({
                        'pattern': pattern_name,
                        'source': 'keyword',
                        'warning': warning_text,
                        'triggered_by': keyword
                    })
                    seen_patterns.add(pattern_name)
                    break
        
        safe_close_connection(conn)
    except Exception as e:
        logger.warning(f"Keyword pattern lookup failed: {e}")
    
    return warnings


def search_past_failures(
    cur,
    conn,
    node_content: str,
    node_id: str
) -> list[dict]:
    """
    Search for past failures matching this hypothesis (keyword + semantic).
    
    Args:
        cur: Database cursor
        conn: Database connection (for rollback on failure)
        node_content: The hypothesis text
        node_id: The node ID (for logging)
        
    Returns:
        List of failure warning dicts
    """
    past_failures = []
    
    # Keyword pattern matching (fast, deterministic)
    keyword_warnings = _query_keyword_failure_patterns(node_content)
    for w in keyword_warnings:
        past_failures.append({
            "pattern": w.get('pattern'),
            "warning": w.get('warning'),
            "source": "keyword",
            "triggered_by": w.get('triggered_by')
        })
    
    # Semantic matching via embedding
    try:
        node_embedding = get_embedding(node_content[:1000])
        cur.execute("""
            SELECT o.failure_reason, s.goal,
                   1 - (o.failure_reason_embedding <=> %s::vector) as similarity
            FROM outcome_records o
            JOIN reasoning_sessions s ON o.session_id = s.id
            WHERE o.outcome != 'success'
            AND o.failure_reason_embedding IS NOT NULL
            AND 1 - (o.failure_reason_embedding <=> %s::vector) > 0.5
            ORDER BY similarity DESC
            LIMIT 3
        """, (node_embedding, node_embedding))
        
        for r in cur.fetchall():
            # Avoid duplicates from keyword matches
            if not any(r["failure_reason"][:50] in str(pf.get('warning', '')) for pf in past_failures):
                past_failures.append({
                    "failure_reason": r["failure_reason"][:200],
                    "session_goal": r["goal"][:100],
                    "similarity": round(float(r["similarity"]), 3),
                    "source": "semantic"
                })
        
        if past_failures:
            logger.info(f"Found {len(past_failures)} failure warning(s) for node {node_id}")
    except Exception as e:
        logger.warning(f"Past failures semantic lookup failed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    
    return past_failures


def search_past_critiques(
    cur,
    node_content: str,
    node_id: str
) -> list[dict]:
    """
    Search for past critiques of similar hypotheses (semantic).
    
    Args:
        cur: Database cursor
        node_content: The hypothesis text
        node_id: The node ID (to exclude from results)
        
    Returns:
        List of past critique dicts
    """
    past_critiques = []
    
    try:
        node_embedding = get_embedding(node_content[:1000])
        cur.execute("""
            SELECT t.content AS hypothesis, t.critique, t.critique_severity,
                   1 - (t.embedding <=> %s::vector) as similarity,
                   s.goal
            FROM thought_nodes t
            JOIN reasoning_sessions s ON t.session_id = s.id
            WHERE t.critique IS NOT NULL
              AND t.embedding IS NOT NULL
              AND t.id != %s
              AND 1 - (t.embedding <=> %s::vector) > 0.65
            ORDER BY similarity DESC
            LIMIT 3
        """, (node_embedding, node_id, node_embedding))
        
        for r in cur.fetchall():
            past_critiques.append({
                "similar_hypothesis": r["hypothesis"][:150] + "..." if len(r["hypothesis"]) > 150 else r["hypothesis"],
                "critique": r["critique"],
                "severity": float(r["critique_severity"]) if r["critique_severity"] else None,
                "similarity": round(float(r["similarity"]), 3),
                "session_goal": r["goal"][:100]
            })
        
        if past_critiques:
            logger.info(f"Found {len(past_critiques)} past critique(s) for similar hypotheses")
    except Exception as e:
        logger.warning(f"Past critiques lookup failed: {e}")
    
    return past_critiques


def build_assumption_extraction_prompt(node_content: str) -> str:
    """
    Build prompt for extracting implicit assumptions from a hypothesis.
    
    Args:
        node_content: The hypothesis text
        
    Returns:
        Prompt string for LLM to process
    """
    return f"""Analyze this hypothesis and extract 2-3 IMPLICIT ASSUMPTIONS that must be true for it to work:

Hypothesis: {node_content}

For each assumption, answer:
1. What is the assumption? (something taken for granted)
2. When could this assumption be FALSE? (challenge it)
3. What happens if this assumption fails? (the risk)

Focus on hidden dependencies, preconditions, and things taken for granted.
Format your response as a numbered list."""
