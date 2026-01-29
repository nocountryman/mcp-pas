"""
PAS Reasoning Helper Functions

Pure utility functions for reasoning tree operations.
These have no side effects and can be easily tested.
"""

import math
import random
from typing import Any, Optional

from utils import detect_negation

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
