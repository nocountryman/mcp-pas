"""
PAS Reasoning Helper Functions

Pure utility functions for reasoning tree operations.
These have no side effects and can be easily tested.
"""

import logging
import math
import random
from typing import Any, Optional

from pas.utils import detect_negation, get_embedding, get_db_connection, safe_close_connection

logger = logging.getLogger(__name__)

# =============================================================================
# Keyword Failure Patterns (fallback if DB lookup fails)
# =============================================================================

KEYWORD_FAILURE_PATTERNS: dict[str, tuple[str, str]] = {
    "embedding": ("EMBEDDING_SHAPE_MISMATCH", "Embedding operations often fail due to dimension mismatches. Verify vector dimensions match schema (1024 for nomic)."),
    "vector": ("EMBEDDING_SHAPE_MISMATCH", "Vector operations need dimension checks."),
    "import": ("SCOPE_BOUNDARY_CROSSING", "Imports across file boundaries risk missing implicit context."),
    "config": ("SCOPE_BOUNDARY_CROSSING", "Config references need explicit qualification."),
    "schema": ("SCHEMA_EVOLUTION", "Schema changes need migration planning."),
    "column": ("SCHEMA_EVOLUTION", "Column changes may break existing queries."),
}

# =============================================================================
# Constants
# =============================================================================

# Heuristic penalties for finalize_session
HEURISTIC_PENALTIES = {
    "unchallenged": 0.05,      # Hypothesis was never critiqued
    "shallow_alternatives": 0.03,  # <2 alternatives at same level
    "monoculture": 0.05        # All siblings match same law
}

# Depth bonus for deeper exploration
DEPTH_BONUS_PER_LEVEL = 0.02

# Rollout Monte Carlo weight
ROLLOUT_WEIGHT = 0.2

# UCT exploration parameters
UCT_THRESHOLD = 0.05           # Gap below which to apply UCT
UCT_EXPLORATION_C = 1.0        # Exploration constant


# =============================================================================
# Failure Search Functions
# =============================================================================

def search_relevant_failures(
    text: str,
    semantic_threshold: float = 0.55,
    limit: int = 3,
    context_type: str = "goal",  # v42b: "goal", "scope", "critique"
    schema_check_required: bool = False,  # v42b: boost schema failures
    exclude_ids: set = None  # v42b: deduplication
) -> list[dict]:
    """
    v32/v42b: Search for past failures relevant to the given text.
    
    Combines:
    1. Keyword pattern matching from failure_patterns table (or fallback dict)
    2. Semantic search on failure_reason_embedding
    
    v42b enhancements:
    - context_type: adjusts threshold (0.45 for scope, 0.55 for goal)
    - schema_check_required: boosts schema-related failures
    - exclude_ids: skip already-surfaced warnings for deduplication
    
    Returns list of warnings with source and details.
    """
    # v42b: Adjust threshold based on context
    if context_type == "scope":
        semantic_threshold = 0.45  # Lower threshold for scope matching
    
    # v42b: Boost schema failures when relevant
    search_text = text
    if schema_check_required and "schema" not in text.lower():
        search_text = f"{text} schema table migration CREATE ALTER"
    
    warnings = []
    text_lower = search_text.lower()
    seen_patterns: set[str] = set()
    exclude_ids = exclude_ids or set()

    
    # Part 1: Keyword pattern matching from DB (v32) or fallback dict
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # v32: Query failure_patterns table
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
        logger.warning(f"v32 DB pattern lookup failed, using fallback: {e}")
        # Fallback to hardcoded dict
        for keyword, (pattern, warning_text) in KEYWORD_FAILURE_PATTERNS.items():
            if keyword in text_lower and pattern not in seen_patterns:
                warnings.append({
                    'pattern': pattern,
                    'source': 'keyword',
                    'warning': warning_text,
                    'triggered_by': keyword
                })
                seen_patterns.add(pattern)
    
    # Part 2: Semantic search (broader, probabilistic)
    if len(warnings) < limit:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Generate embedding for the text
            embedding = get_embedding(text[:1000])
            
            cur.execute(
                """
                SELECT failure_reason, 
                       1 - (failure_reason_embedding <=> %s::vector) as similarity,
                       notes
                FROM outcome_records
                WHERE outcome = 'failure' 
                  AND failure_reason_embedding IS NOT NULL
                  AND 1 - (failure_reason_embedding <=> %s::vector) > %s
                ORDER BY failure_reason_embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, semantic_threshold, embedding, limit - len(warnings))
            )
            
            for row in cur.fetchall():
                # Avoid duplicates if semantic matches keyword pattern
                if not any(w.get('triggered_by') and w['triggered_by'] in row['failure_reason'].lower() for w in warnings):
                    warnings.append({
                        'pattern': 'SEMANTIC_MATCH',
                        'source': 'semantic',
                        'warning': row['failure_reason'],
                        'similarity': round(float(row['similarity']), 3)
                    })
            
            safe_close_connection(conn)
        except Exception as e:
            logger.warning(f"Semantic failure search failed: {e}")
    
    return warnings[:limit]


# =============================================================================
# Domain Pattern Matching
# =============================================================================

DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("database", ["db", "schema", "table", "sql", "postgres", "column"]),
    ("api", ["api", "endpoint", "route", "request", "response"]),
    ("backend", ["server", "backend", "service", "logic"]),
    ("frontend", ["ui", "ux", "frontend", "component", "page"]),
    ("testing", ["test", "pytest", "verify", "validation"]),
    ("bugfix", ["bug", "fix", "error", "failure"]),
    ("feature", ["feature", "implement", "add", "new"]),
    ("refactor", ["refactor", "cleanup", "improve", "optimize"]),
]


# =============================================================================
# Pure Functions
# =============================================================================

def apply_heuristic_penalties(
    node: dict[str, Any],
    sibling_counts: dict[str, int],
    law_diversity: dict[str, tuple[int, int]]
) -> tuple[float, list[str]]:
    """
    Apply penalties to candidate score.
    
    Args:
        node: Candidate node from DB (needs posterior_score, likelihood, depth, path)
        sibling_counts: Dict of parent_path -> count of siblings at that level
        law_diversity: Dict of parent_path -> (unique_laws, total_siblings)
    
    Returns:
        Tuple of (adjusted_score, list_of_penalties_applied)
    """
    original_score = float(node["posterior_score"])
    adjusted_score = original_score
    penalties: list[str] = []
    
    # Unchallenged penalty: if likelihood is suspiciously round
    likelihood = float(node["likelihood"])
    if str(likelihood).endswith(('0', '5')) and likelihood in [0.8, 0.85, 0.9, 0.95, 0.88, 0.92]:
        adjusted_score -= HEURISTIC_PENALTIES["unchallenged"]
        penalties.append("unchallenged_penalty")
    
    # Depth bonus (deeper = more refined)
    depth = int(node["depth"])
    if depth > 2:
        bonus = (depth - 2) * DEPTH_BONUS_PER_LEVEL
        adjusted_score += bonus
        if bonus > 0:
            penalties.append(f"depth_bonus_+{bonus:.2f}")
    
    # Shallow alternatives penalty: if <2 sibling hypotheses at same level
    node_path = str(node["path"])
    parent_path = ".".join(node_path.split(".")[:-1]) if "." in node_path else ""
    sibling_count = sibling_counts.get(parent_path, 1)
    if sibling_count < 2:
        adjusted_score -= HEURISTIC_PENALTIES["shallow_alternatives"]
        penalties.append("shallow_alternatives_penalty")
    
    # v13b: Monoculture penalty - if all siblings match same law, penalize
    if parent_path in law_diversity:
        unique_laws, total_siblings = law_diversity[parent_path]
        if total_siblings >= 2 and unique_laws == 1:
            adjusted_score -= HEURISTIC_PENALTIES["monoculture"]
            penalties.append("monoculture_penalty")
    
    # Ensure score stays in valid range
    adjusted_score = max(0.1, min(1.0, adjusted_score))
    
    return adjusted_score, penalties


def get_outcome_multiplier(outcome: str) -> float:
    """
    Get multiplier for trait persistence based on outcome.
    
    Args:
        outcome: 'success', 'partial', or 'failure'
        
    Returns:
        Multiplier factor (1.2 for success, 0.8 for failure, 1.0 otherwise)
    """
    if outcome == "success":
        return 1.2
    elif outcome == "failure":
        return 0.8
    return 1.0


def compute_critique_accuracy(
    node_path: str,
    winning_path: str,
    outcome: str
) -> Optional[bool]:
    """
    Determine if critique was accurate based on outcome.
    
    Args:
        node_path: Path of the critiqued node
        winning_path: Path of the winning hypothesis
        outcome: Session outcome ('success', 'partial', 'failure')
    
    Returns:
        True if critique was accurate, False if not, None if indeterminate
    """
    is_winner = str(node_path).startswith(str(winning_path)) or str(winning_path).startswith(str(node_path))
    
    if is_winner:
        # If winner was critiqued and outcome is failure, critique was accurate
        return outcome == 'failure'
    else:
        # If non-winner was critiqued and outcome is success, critique may have been accurate
        return outcome == 'success'


def compute_law_effective_weight(
    base_weight: float,
    selection_count: int,
    success_count: int,
    negation_asymmetry: bool
) -> tuple[float, float]:
    """
    Compute effective weight using Thompson sampling.
    
    Args:
        base_weight: Scientific weight from law
        selection_count: Times this law was selected
        success_count: Times it led to success
        negation_asymmetry: If hypothesis negation mismatches law
    
    Returns:
        Tuple of (effective_weight, thompson_sample)
    """
    alpha = success_count + 1
    beta_param = selection_count - success_count + 1
    thompson_sample = random.betavariate(alpha, beta_param)
    
    effective_weight = 0.7 * base_weight + 0.3 * thompson_sample
    negation_penalty = 0.15 if negation_asymmetry else 0.0
    effective_weight = max(0.1, effective_weight - negation_penalty)
    
    return effective_weight, thompson_sample


def compute_ensemble_prior(
    matching_laws: list[dict],
    hypothesis_text: str
) -> tuple[float, list[int], str | None]:
    """
    Compute ensemble prior from multiple matching laws.
    
    Args:
        matching_laws: List of law dicts with similarity, scientific_weight, etc
        hypothesis_text: The hypothesis text (for negation check)
    
    Returns:
        Tuple of (prior, supporting_law_ids, primary_law_name)
    """
    if not matching_laws:
        return 0.5, [], None
    
    total_weighted = 0.0
    total_similarity = 0.0
    supporting_law_ids = []
    law_names = []
    
    for law in matching_laws:
        # Negation detection
        hyp_negations = detect_negation(hypothesis_text)
        law_negations = detect_negation(law["definition"]) if law.get("definition") else set()
        negation_asymmetry = bool(hyp_negations) != bool(law_negations)
        
        # Compute effective weight
        effective_weight, _ = compute_law_effective_weight(
            float(law["scientific_weight"]),
            law.get("selection_count", 0) or 0,
            law.get("success_count", 0) or 0,
            negation_asymmetry
        )
        
        similarity = float(law["similarity"])
        total_weighted += effective_weight * similarity
        total_similarity += similarity
        
        supporting_law_ids.append(law["id"])
        law_names.append(law["law_name"])
    
    # Ensemble prior = Σ(weight × similarity) / Σ(similarity)
    prior = (total_weighted / total_similarity * 0.6) + (matching_laws[0]["similarity"] * 0.4)
    prior = max(0.1, min(0.95, prior))
    
    return prior, supporting_law_ids, law_names[0] if law_names else None


def generate_suggested_tags(goal: str, winning_content: str, max_tags: int = 5) -> list[str]:
    """
    Generate suggested tags based on goal and winning content.
    
    Args:
        goal: Session goal text
        winning_content: Content of the winning hypothesis
        max_tags: Maximum number of tags to return
        
    Returns:
        List of suggested tag strings
    """
    combined_text = f"{goal} {winning_content}".lower()
    tags = []
    for tag, keywords in DOMAIN_PATTERNS:
        if any(kw in combined_text for kw in keywords):
            tags.append(tag)
    return tags[:max_tags]


def compute_decision_quality(
    gap: float,
    rec_conf: float,
    run_conf: float,
    uct_applied: bool = False
) -> tuple[str, str]:
    """
    Compute decision quality and gap analysis message.
    
    Args:
        gap: Score difference between recommendation and runner-up
        rec_conf: Recommendation confidence
        run_conf: Runner-up confidence
        uct_applied: Whether UCT exploration was applied
        
    Returns:
        Tuple of (decision_quality, gap_analysis_message)
    """
    if gap < 0.03:
        decision_quality = "low"
        gap_analysis = f"Very close decision (gap: {gap:.3f}). Consider both options."
    elif gap < 0.10:
        decision_quality = "medium"
        gap_analysis = f"Moderate confidence (gap: {gap:.3f}). Winner is better but runner-up has merit."
    else:
        decision_quality = "high"
        gap_analysis = f"Clear winner (gap: {gap:.3f}). High confidence in recommendation."
    
    # v14a.1: Confidence-weighted gap
    weighted_gap = gap * min(rec_conf, run_conf)
    
    if decision_quality == "high" and weighted_gap < 0.05:
        decision_quality = "medium"
        gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
    elif decision_quality == "medium" and weighted_gap < 0.02:
        decision_quality = "low"
        gap_analysis += f" [v14a.1: Downgraded - low confidence (wgap: {weighted_gap:.3f})]"
    
    if uct_applied:
        gap_analysis += " [v11a: UCT exploration applied]"
    
    return decision_quality, gap_analysis


def apply_uct_tiebreaking(
    rec_score: float,
    rec_depth: int,
    run_score: float,
    run_depth: int
) -> tuple[bool, bool]:
    """
    Apply UCT exploration bonus for close decisions.
    
    Args:
        rec_score: Recommendation score
        rec_depth: Recommendation depth
        run_score: Runner-up score
        run_depth: Runner-up depth
        
    Returns:
        Tuple of (should_swap, uct_was_applied)
    """
    gap = rec_score - run_score
    if gap >= UCT_THRESHOLD:
        return False, False
    
    rec_visits = max(1, rec_depth)
    run_visits = max(1, run_depth)
    total_visits = rec_visits + run_visits
    
    rec_uct = rec_score + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / rec_visits)
    run_uct = run_score + UCT_EXPLORATION_C * math.sqrt(math.log(total_visits) / run_visits)
    
    return run_uct > rec_uct, True


def build_processed_candidate(
    node: dict,
    adjusted_score: float,
    penalties: list[str],
    rollout_score: float
) -> dict[str, Any]:
    """
    Build a processed candidate dict for finalize_session.
    
    Args:
        node: Raw node dict from database
        adjusted_score: Score after heuristic penalties
        penalties: List of penalty names applied
        rollout_score: Monte Carlo rollout score
        
    Returns:
        Processed candidate dict with all scores
    """
    original_score = float(node["posterior_score"])
    depth = int(node["depth"])
    final_score = (1 - ROLLOUT_WEIGHT) * adjusted_score + ROLLOUT_WEIGHT * rollout_score
    
    return {
        "node_id": str(node["id"]),
        "path": node["path"],
        "content": node["content"],
        "depth": depth,
        "prior_score": round(float(node["prior_score"]), 4),
        "likelihood": round(float(node["likelihood"]), 4),
        "confidence": round(float(node["prior_score"]) * float(node["likelihood"]), 4),
        "original_score": round(original_score, 4),
        "adjusted_score": round(adjusted_score, 4),
        "rollout_score": round(rollout_score, 4),
        "final_score": round(final_score, 4),
        "penalties_applied": penalties
    }


# =============================================================================
# Trait Inference (v35)
# =============================================================================

def infer_traits_from_hidden_values(
    hidden_value_counts: dict[str, int]
) -> list[dict]:
    """
    Infer latent traits from hidden value patterns.
    
    Args:
        hidden_value_counts: Dict of hidden_value -> occurrence count
    
    Returns:
        List of trait dicts with trait, confidence, evidence_count
    """
    # Hidden value to trait mapping
    TRAIT_MAPPING = {
        # Risk traits
        "RISK_TOLERANCE": ("RISK_TOLERANT", "RISK_AVERSE"),
        "RISK_AVERSION": ("RISK_AVERSE",),
        
        # Control traits
        "CONTROL_PREFERENCE": ("CONTROL_ORIENTED",),
        "DELEGATION_COMFORT": ("AUTOMATION_TRUSTING",),
        
        # Approach traits
        "SIMPLICITY_PREFERENCE": ("MINIMALIST",),
        "SAFETY_PREFERENCE": ("SAFETY_CONSCIOUS",),
        "SPEED_PREFERENCE": ("SPEED_FOCUSED",),
        "AUTONOMY_PREFERENCE": ("AUTONOMY_FOCUSED",),
    }
    
    traits = []
    for hidden_value, count in hidden_value_counts.items():
        if hidden_value in TRAIT_MAPPING:
            for trait in TRAIT_MAPPING[hidden_value]:
                # Confidence scales with evidence count (capped at 1.0)
                confidence = min(1.0, 0.5 + count * 0.1)
                traits.append({
                    "trait": trait,
                    "confidence": confidence,
                    "evidence_count": count,
                    "source": hidden_value
                })
    
    return traits
