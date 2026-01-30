"""
PAS Hybrid Synthesis Helpers (v40 Phase 0)

Pure functions for complementarity detection and hypothesis synthesis.
Enables PAS to detect when hypotheses address different goals and should
be combined rather than selected winner-takes-all.
"""

from typing import List, Tuple, Dict, Any
import re


# Goal extraction patterns - maps keywords to goal domains
GOAL_PATTERNS: Dict[str, List[str]] = {
    "reasoning": [
        "metacognitive", "5-stage", "critique", "reasoning", "tot", 
        "tree of thought", "bayesian", "hypothesis", "inference"
    ],
    "learning": [
        "rlvr", "rlaif", "calibration", "outcome", "prediction", 
        "self-learning", "reward", "feedback", "training"
    ],
    "code_awareness": [
        "purpose", "semantic", "intent", "symbol", "codebase", 
        "syntax", "comprehension", "indexing", "lsif"
    ],
    "self_awareness": [
        "analyze_self", "introspection", "self-aware", "meta-analysis",
        "capability", "limitation", "identity"
    ],
}


def extract_addressed_goals(content: str, scope: str = "") -> List[str]:
    """
    Extract goals/domains addressed by a hypothesis.
    
    Analyzes content and scope to identify which problem domains
    the hypothesis targets using keyword matching.
    
    Args:
        content: The hypothesis text content
        scope: Optional scope declaration (file paths, layers)
        
    Returns:
        List of goal tags like ["reasoning", "learning", "code_awareness"]
    """
    goals = []
    text = f"{content} {scope}".lower()
    
    for goal, patterns in GOAL_PATTERNS.items():
        if any(p in text for p in patterns):
            goals.append(goal)
    
    return goals


def compute_goal_overlap(goals_a: List[str], goals_b: List[str]) -> float:
    """
    Compute Jaccard similarity between two goal sets.
    
    Args:
        goals_a: First list of goal tags
        goals_b: Second list of goal tags
        
    Returns:
        Similarity score: 0.0 for completely disjoint, 1.0 for identical
    """
    if not goals_a and not goals_b:
        return 1.0  # Both empty = same
    
    set_a, set_b = set(goals_a), set(goals_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def detect_complementarity(
    candidates: List[Dict[str, Any]], 
    threshold: float = 0.5
) -> Tuple[bool, List[str], float]:
    """
    Detect if top candidates address different goals (are complementary).
    
    Complementary hypotheses address distinct problem domains and should
    be synthesized rather than selected winner-takes-all.
    
    Args:
        candidates: List of dicts with 'content' and optional 'scope' keys
        threshold: Max average pairwise overlap to consider complementary
        
    Returns:
        Tuple of:
        - is_complementary: True if candidates are complementary
        - covered_goals: Union of all goals addressed
        - avg_overlap: Average pairwise goal overlap
    """
    if len(candidates) < 2:
        return False, [], 1.0
    
    # Extract goals for each candidate
    candidate_goals = []
    all_goals = set()
    
    for c in candidates:
        goals = extract_addressed_goals(
            c.get("content", ""), 
            c.get("scope", "")
        )
        candidate_goals.append(set(goals))
        all_goals.update(goals)
    
    # Calculate average pairwise overlap
    total_overlap = 0.0
    pairs = 0
    
    for i, g1 in enumerate(candidate_goals):
        for g2 in candidate_goals[i+1:]:
            total_overlap += compute_goal_overlap(list(g1), list(g2))
            pairs += 1
    
    avg_overlap = total_overlap / pairs if pairs > 0 else 1.0
    is_complementary = avg_overlap < threshold
    
    return is_complementary, sorted(all_goals), avg_overlap


def synthesize_hypothesis_text(
    candidates: List[Dict[str, Any]],
    include_details: bool = True
) -> str:
    """
    Generate a merged hypothesis description from complementary candidates.
    
    Creates a unified hypothesis that combines the core ideas from
    multiple complementary hypotheses.
    
    Args:
        candidates: List of hypothesis dicts with 'content' and optional 'scope'
        include_details: If True, include individual component descriptions
        
    Returns:
        Synthesized hypothesis text
    """
    parts = []
    details = []
    
    for i, c in enumerate(candidates, 1):
        content = c.get("content", "")
        
        # Extract bold title if present
        match = re.search(r"\*\*([^*]+)\*\*", content)
        if match:
            title = match.group(1).strip()
            parts.append(f"({i}) {title}")
        else:
            # Fallback to first sentence
            first_sentence = content.split('.')[0][:50]
            parts.append(f"({i}) {first_sentence}")
        
        if include_details:
            # Extract goals addressed
            goals = extract_addressed_goals(content, c.get("scope", ""))
            if goals:
                details.append(f"  - Component {i} addresses: {', '.join(goals)}")
    
    synthesis = f"**Unified Approach**: Combine {' + '.join(parts)}"
    
    if details:
        synthesis += "\n\n**Components**:\n" + "\n".join(details)
    
    return synthesis


def merge_scopes(candidates: List[Dict[str, Any]]) -> str:
    """
    Merge scope declarations from multiple candidates.
    
    Args:
        candidates: List of hypothesis dicts with optional 'scope'
        
    Returns:
        Combined scope string with deduplication
    """
    all_files = set()
    
    for c in candidates:
        scope = c.get("scope", "")
        # Extract file references (anything that looks like a path or .py/.sql)
        files = re.findall(r'[\w/]+\.(?:py|sql|md|yaml|json)', scope)
        all_files.update(files)
    
    return ", ".join(sorted(all_files)) if all_files else ""
