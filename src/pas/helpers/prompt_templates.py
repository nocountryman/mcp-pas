"""
PAS Prompt Template Helpers (Phase 37)

Centralized prompt building with injection points for:
- Constraint tags with enforcement levels
- Few-shot JSON examples
- Psychological weight display
"""

from typing import Optional


# =============================================================================
# Few-Shot Example Constants
# =============================================================================

HYPOTHESIS_EXAMPLE = """{
  "hypotheses": [{
    "text": "Refactor auth.py to use dependency injection",
    "confidence": 0.85,
    "scope": "auth.py, test_auth.py",
    "supporting_evidence": ["Reduces coupling", "Easier testing"],
    "disconfirming_evidence": ["Requires changes to 5 callers", "Runtime overhead"]
  }]
}"""

CRITIQUE_EXAMPLE = """{
  "major_flaws": ["Race condition in concurrent access", "No rollback on failure"],
  "minor_flaws": ["Verbose logging"],
  "severity": 0.65,
  "edge_cases": ["Empty input array", "Network timeout mid-transaction"]
}"""

GAPS_EXAMPLE = """{
  "gaps": [
    {"layer": "CODE_STRUCTURE", "covered": "Main logic changes", "missing": "Helper utilities"},
    {"layer": "DEPENDENCIES", "covered": "Core packages", "missing": "Dev dependencies"}
  ],
  "critical_gaps": ["No error handling for network failures"],
  "overall_coverage": 0.75
}"""

PLAN_EXAMPLE = """{
  "phases": [
    {"name": "Schema Migration", "files": ["migrations/001.sql"], "verification": "psql check"},
    {"name": "Implementation", "files": ["src/main.py"], "verification": "pytest"}
  ],
  "total_files": 2,
  "estimated_complexity": "medium"
}"""


# =============================================================================
# Constraint Tag Formatting
# =============================================================================

def format_constraint_tag(
    constraint: dict,
    include_weight: bool = True
) -> str:
    """Format a constraint as a tagged line.
    
    Format: [ENFORCEMENT:key] value
            ↳ *enforcement_reason* (if psychological_weight present)
    
    Args:
        constraint: Constraint dict with enforcement_level, constraint_key, constraint_data
        include_weight: Whether to include psychological weight context
        
    Returns:
        Formatted constraint tag string
    """
    enforcement = constraint.get("enforcement_level", "warn").upper()
    key = constraint.get("constraint_key", constraint.get("key", "unknown"))
    value = constraint.get("constraint_data", constraint.get("value", {}))
    
    # Extract display value
    if isinstance(value, dict):
        display = value.get("value", value.get("description", str(value)[:100]))
    else:
        display = str(value)[:100]
    
    # Map enforcement to icon
    icon = "🚫" if enforcement == "BLOCK" else "⚠️"
    tag_line = f"{icon} [{enforcement}:{key}] {display}"
    
    # Add psychological weight context if present
    if include_weight:
        weight = constraint.get("psychological_weight", {})
        if weight and weight.get("enforcement_reason"):
            tag_line += f"\n  ↳ *{weight['enforcement_reason']}*"
    
    return tag_line


def build_constraint_section(
    constraints: list[dict],
    title: str = "Active Constraints"
) -> str:
    """Build a markdown constraint section for prompt injection.
    
    Args:
        constraints: List of constraint dicts
        title: Section title
        
    Returns:
        Markdown formatted constraint section
    """
    if not constraints:
        return ""
    
    lines = [f"## {title}\n"]
    
    # Separate blocking vs warning constraints
    blocking = [c for c in constraints if c.get("enforcement_level", c.get("enforcement", "warn")).lower() == "block"]
    warning = [c for c in constraints if c.get("enforcement_level", c.get("enforcement", "warn")).lower() != "block"]
    
    if blocking:
        lines.append("> [!CAUTION]")
        lines.append(f"> {len(blocking)} BLOCKING constraint(s) - violations will be rejected.\n")
    
    for c in constraints:
        lines.append(format_constraint_tag(c))
    
    return "\n".join(lines)


def inject_few_shot_example(
    prompt_type: str,
    example_json: Optional[str] = None
) -> str:
    """Wrap a few-shot example in markdown.
    
    Args:
        prompt_type: One of 'hypothesis', 'critique', 'gaps', 'plan'
        example_json: Custom example or None to use default
        
    Returns:
        Markdown formatted example section
    """
    # Use default examples if not provided
    defaults = {
        "hypothesis": HYPOTHESIS_EXAMPLE,
        "critique": CRITIQUE_EXAMPLE,
        "gaps": GAPS_EXAMPLE,
        "plan": PLAN_EXAMPLE,
    }
    
    example = example_json or defaults.get(prompt_type, "{}")
    
    return f"""## Example Output

```json
{example}
```
"""


# =============================================================================
# Psychological Weight Helpers
# =============================================================================

def detect_psychological_markers(text: str) -> dict:
    """Detect psychological patterns in user text.
    
    Looks for:
    - Loss-framing: "prevent", "avoid", "don't want", "stop"
    - Gain-framing: "enable", "improve", "want", "achieve"
    - Hedging: "might", "maybe", "could", "perhaps"
    
    Args:
        text: User input text
        
    Returns:
        Dict with detected patterns and priority boost
    """
    text_lower = text.lower()
    
    loss_markers = ["prevent", "avoid", "don't want", "stop", "never", "block", "fail"]
    gain_markers = ["enable", "improve", "want", "achieve", "great", "love", "amazing"]
    hedge_markers = ["might", "maybe", "could", "perhaps", "probably", "possibly"]
    
    detected_loss = [m for m in loss_markers if m in text_lower]
    detected_gain = [m for m in gain_markers if m in text_lower]
    detected_hedge = [m for m in hedge_markers if m in text_lower]
    
    # Loss aversion: losses weighted ~2x gains
    priority_boost = 0.0
    if detected_loss:
        priority_boost += 0.2  # Loss-framing = higher priority
    if detected_gain and not detected_loss:
        priority_boost += 0.1  # Gain-framing = moderate priority
    if detected_hedge:
        priority_boost -= 0.1  # Hedging = lower confidence
    
    source_pattern = None
    if detected_loss:
        source_pattern = "loss_aversion"
    elif detected_gain:
        source_pattern = "gain_framing"
    elif detected_hedge:
        source_pattern = "uncertainty_hedging"
    
    return {
        "source_pattern": source_pattern,
        "user_markers": detected_loss + detected_gain + detected_hedge,
        "priority_boost": priority_boost,
        "loss_markers": detected_loss,
        "gain_markers": detected_gain,
        "hedge_markers": detected_hedge,
    }


def build_psychological_weight(
    text: str,
    source_answer: Optional[str] = None,
    source_session: Optional[str] = None
) -> dict:
    """Build psychological_weight JSONB for constraint storage.
    
    Args:
        text: User input text to analyze
        source_answer: Optional interview answer ID
        source_session: Optional session ID
        
    Returns:
        Dict suitable for psychological_weight JSONB column
    """
    markers = detect_psychological_markers(text)
    
    if not markers["source_pattern"]:
        return {}
    
    enforcement_reason = None
    if markers["loss_markers"]:
        enforcement_reason = f"User used loss-framing language: {', '.join(markers['loss_markers'][:3])}"
    elif markers["gain_markers"]:
        enforcement_reason = f"User expressed enthusiasm: {', '.join(markers['gain_markers'][:3])}"
    elif markers["hedge_markers"]:
        enforcement_reason = f"User hedged with uncertainty: {', '.join(markers['hedge_markers'][:3])}"
    
    return {
        "source_pattern": markers["source_pattern"],
        "user_markers": markers["user_markers"][:5],  # Limit to 5 markers
        "priority_boost": markers["priority_boost"],
        "enforcement_reason": enforcement_reason,
        "source_answer": source_answer,
        "source_session": source_session,
    }
