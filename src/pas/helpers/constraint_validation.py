"""Constraint validation for hypothesis and recommendation checking.

Phase 7c: Environment Constraints
PAS Session: fea58ef5-3773-46ac-b2d2-359f2283ba29
"""

import re
from typing import Optional, Any

from pas.utils import get_db_connection


# Patterns that indicate MVP/v1/simplified approaches
MVP_PATTERNS = [
    r"\bv1\b",
    r"\bmvp\b",
    r"\bminimum\s+viable\b",
    r"\bbasic\s+version\b",
    r"\bsimplified\b",
    r"\binitial\s+version\b",
    r"\bfirst\s+iteration\b",
    r"\bstripped[\s-]?down\b",
    r"\bbare[\s-]?bones\b",
    r"\bproof\s+of\s+concept\b",
    r"\bpoc\b",
    r"\bfor\s+now\b",
    r"\bquick\s+and\s+dirty\b",
]


def get_active_constraints(
    project_id: str, 
    constraint_type: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Fetch active (non-expired) constraints for a project.
    
    Args:
        project_id: Project identifier string (e.g., 'mcp-pas')
        constraint_type: Optional filter by type ('philosophy', 'environment', 'quality')
    
    Returns:
        List of constraint dicts with key, data, enforcement_level, type
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get project UUID from project_id string
    cur.execute("SELECT id FROM project_registry WHERE project_id = %s", (project_id,))
    row = cur.fetchone()
    if not row:
        return []
    
    project_uuid = row["id"]
    
    query = """
        SELECT constraint_key, constraint_data, enforcement_level, constraint_type, source
        FROM project_constraints
        WHERE project_id = %s AND valid_to IS NULL
    """
    params: list[Any] = [project_uuid]
    
    if constraint_type:
        query += " AND constraint_type = %s"
        params.append(constraint_type)
    
    query += " ORDER BY constraint_key"
    
    cur.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def validate_hypothesis(hypothesis: str, project_id: str) -> dict[str, Any]:
    """
    Check hypothesis text against blocking constraints.
    
    Args:
        hypothesis: The hypothesis text to validate
        project_id: Project identifier
    
    Returns:
        Validation result with:
        - passed: bool - True if no blocking violations
        - violations: list of blocking constraint violations
        - warnings: list of warn-level constraint issues
        - blocked: bool - True if hypothesis should be rejected
    """
    constraints = get_active_constraints(project_id, "philosophy")
    violations = []
    warnings = []
    
    # Check no_mvp constraint
    no_mvp = next(
        (c for c in constraints if c["constraint_key"] == "no_mvp"), 
        None
    )
    
    if no_mvp and no_mvp["constraint_data"] in (True, "true", "True"):
        for pattern in MVP_PATTERNS:
            match = re.search(pattern, hypothesis, re.IGNORECASE)
            if match:
                violation = {
                    "constraint": "no_mvp",
                    "pattern_matched": pattern,
                    "matched_text": match.group(0),
                    "enforcement": no_mvp["enforcement_level"],
                    "message": f"Hypothesis contains MVP language: '{match.group(0)}'"
                }
                
                if no_mvp["enforcement_level"] == "block":
                    violations.append(violation)
                else:
                    warnings.append(violation)
                break  # One match is enough
    
    # Check code_quality constraint
    code_quality = next(
        (c for c in constraints if c["constraint_key"] == "code_quality"),
        None
    )
    
    if code_quality and code_quality["constraint_data"] == "production_grade":
        # Check for prototype/draft language
        prototype_patterns = [r"\bprototype\b", r"\bdraft\b", r"\bexperimental\b"]
        for pattern in prototype_patterns:
            match = re.search(pattern, hypothesis, re.IGNORECASE)
            if match:
                warning = {
                    "constraint": "code_quality",
                    "pattern_matched": pattern,
                    "matched_text": match.group(0),
                    "enforcement": code_quality["enforcement_level"],
                    "message": f"Hypothesis uses non-production language: '{match.group(0)}'"
                }
                warnings.append(warning)
                break
    
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "blocked": any(v["enforcement"] == "block" for v in violations),
        "constraints_checked": len(constraints)
    }


def validate_recommendation(recommendation: str, project_id: str) -> dict[str, Any]:
    """
    Validate final recommendation against all blocking constraints.
    
    This is called during finalize_session to ensure the winning
    hypothesis doesn't violate any project constraints.
    """
    return validate_hypothesis(recommendation, project_id)


def get_constraint_summary(project_id: str) -> dict[str, Any]:
    """
    Get a summary of active constraints for display in prepare_expansion.
    
    Returns constraints formatted with enforcement icons for surfacing.
    """
    constraints = get_active_constraints(project_id)
    
    ENFORCEMENT_ICONS = {
        "block": "🚫",
        "warn": "⚠️",
        "hidden": "👁️"
    }
    
    visible_constraints = [
        {
            "key": c["constraint_key"],
            "value": c["constraint_data"],
            "enforcement": c["enforcement_level"],
            "icon": ENFORCEMENT_ICONS.get(c["enforcement_level"], ""),
            "type": c["constraint_type"]
        }
        for c in constraints
        if c["enforcement_level"] != "hidden"
    ]
    
    # Build constraint prompt for hypothesis generation
    if visible_constraints:
        lines = ["🔒 ACTIVE CONSTRAINTS:"]
        for c in visible_constraints:
            lines.append(f"  {c['icon']} {c['key']}: {c['value']} ({c['enforcement'].upper()})")
        constraint_prompt = "\n".join(lines)
    else:
        constraint_prompt = None
    
    return {
        "constraints": visible_constraints,
        "constraint_count": len(visible_constraints),
        "constraint_prompt": constraint_prompt,
        "blocking_count": sum(1 for c in visible_constraints if c["enforcement"] == "block")
    }


def validate_lsp_section(plan_text: str) -> dict[str, Any]:
    """
    Validate that implementation plan has proper LSP Impact Analysis section.
    
    Phase 9: LSP Enforcement in Planning
    
    Checks for:
    - Section heading exists
    - Either has symbol content OR explicit skip reasoning
    
    Args:
        plan_text: Full implementation plan text
        
    Returns:
        {
            "passed": bool,
            "has_section": bool,
            "has_content": bool,
            "has_skip_reasoning": bool,
            "message": str
        }
    """
    # Check for LSP section heading
    lsp_pattern = r"##\s*LSP\s*Impact\s*Analysis"
    has_section = bool(re.search(lsp_pattern, plan_text, re.IGNORECASE))
    
    if not has_section:
        return {
            "passed": False,
            "has_section": False,
            "has_content": False,
            "has_skip_reasoning": False,
            "message": "Missing '## LSP Impact Analysis' section"
        }
    
    # Extract section content (from heading to next ## or end)
    section_match = re.search(
        r"##\s*LSP\s*Impact\s*Analysis.*?(?=\n##|\Z)",
        plan_text,
        re.IGNORECASE | re.DOTALL
    )
    
    if not section_match:
        return {
            "passed": False,
            "has_section": True,
            "has_content": False,
            "has_skip_reasoning": False,
            "message": "LSP section found but empty"
        }
    
    section_content = section_match.group(0)
    
    # Check for symbol content (tables with backticks, function names)
    has_symbols = bool(re.search(r"`[a-zA-Z_][a-zA-Z0-9_]*`", section_content))
    has_table = bool(re.search(r"\|.*\|.*\|", section_content))
    has_content = has_symbols or has_table
    
    # Check for skip reasoning
    skip_patterns = [
        r"skip\s*reason",
        r"not\s*(needed|required|applicable)",
        r"N/?A",
        r"no\s*LSP\s*(analysis\s*)?(needed|required)",
        r"trivial\s*(change|fix)",
        r"single\s*file"
    ]
    has_skip_reasoning = any(
        re.search(p, section_content, re.IGNORECASE) 
        for p in skip_patterns
    )
    
    passed = has_content or has_skip_reasoning
    
    if passed:
        message = "LSP section valid"
    else:
        message = "LSP section missing content. Add symbols OR skip reasoning."
    
    return {
        "passed": passed,
        "has_section": True,
        "has_content": has_content,
        "has_skip_reasoning": has_skip_reasoning,
        "message": message
    }

