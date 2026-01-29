"""
PAS Session Helper Functions

Pure functions for session lifecycle management,
trait handling, and session context processing.
"""

import hashlib
from typing import Any, Optional
from datetime import datetime, timezone


# =============================================================================
# Session Configuration
# =============================================================================

# Half-life for trait decay in days
TRAIT_HALF_LIFE_DAYS = 30

# Minimum score threshold for trait inclusion
TRAIT_INCLUSION_THRESHOLD = 0.3

# Maximum traits to load per session
MAX_PERSISTENT_TRAITS = 5


# =============================================================================
# User/Project Identity
# =============================================================================

def derive_user_id_from_goal(goal: str, length: int = 32) -> str:
    """
    Derive a pseudo user_id from the goal text.
    
    Uses SHA-256 hash of goal prefix as a proxy for project identity.
    In production, this could be enhanced with actual user/project tracking.
    
    Args:
        goal: Session goal text
        length: Length of returned hash (default 32)
        
    Returns:
        Hex string user ID
    """
    goal_prefix = goal.strip()[:100]
    return hashlib.sha256(goal_prefix.encode()).hexdigest()[:length]


# =============================================================================
# Trait Decay
# =============================================================================

def compute_decayed_trait_score(
    cumulative_score: float,
    days_since_reinforced: int,
    half_life_days: int = TRAIT_HALF_LIFE_DAYS
) -> float:
    """
    Apply exponential decay to a trait score.
    
    Args:
        cumulative_score: Raw cumulative score
        days_since_reinforced: Days since last reinforcement
        half_life_days: Decay half-life in days
        
    Returns:
        Decayed score (0.0 to 1.0)
    """
    decay_factor = 0.5 ** (days_since_reinforced / half_life_days)
    return min(cumulative_score * decay_factor, 1.0)


def should_include_trait(
    decayed_score: float,
    threshold: float = TRAIT_INCLUSION_THRESHOLD
) -> bool:
    """
    Determine if a trait should be included in session context.
    
    Args:
        decayed_score: Decayed trait score
        threshold: Minimum score for inclusion
        
    Returns:
        True if trait should be included
    """
    return decayed_score >= threshold


def build_trait_entry(
    trait_name: str,
    decayed_score: float,
    days_since_reinforced: int
) -> dict:
    """
    Build a trait entry for session context.
    
    Args:
        trait_name: Name of the trait
        decayed_score: Decayed score value
        days_since_reinforced: Days since reinforced
        
    Returns:
        Trait dict for session context
    """
    return {
        "trait": trait_name,
        "confidence": min(decayed_score, 1.0),
        "source": "persistent",
        "days_since_reinforced": days_since_reinforced
    }


# =============================================================================
# Session Context
# =============================================================================

def build_initial_context(source: str = "mcp_tool") -> dict:
    """
    Build initial session context.
    
    Args:
        source: Context source identifier
        
    Returns:
        Initial context dict
    """
    return {"source": source}


def merge_traits_into_context(
    context: dict,
    traits: list[dict],
    user_id: Optional[str] = None
) -> dict:
    """
    Merge persistent traits into session context.
    
    Args:
        context: Existing context dict
        traits: List of trait entries
        user_id: Optional user ID
        
    Returns:
        Updated context dict
    """
    updates = {}
    
    if traits:
        updates["persistent_traits"] = traits
    
    if user_id:
        updates["user_id"] = user_id
    
    return {**context, **updates}


# =============================================================================
# Session State
# =============================================================================

VALID_SESSION_STATES = {"active", "paused", "completed"}

def validate_session_state(state: str) -> bool:
    """
    Check if a session state is valid.
    
    Args:
        state: State string to validate
        
    Returns:
        True if valid state
    """
    return state in VALID_SESSION_STATES


def compute_session_duration(
    created_at: datetime,
    completed_at: Optional[datetime] = None
) -> float:
    """
    Compute session duration in seconds.
    
    Args:
        created_at: Session creation time
        completed_at: Completion time (or now if None)
        
    Returns:
        Duration in seconds
    """
    if completed_at is None:
        completed_at = datetime.now(timezone.utc)
    
    delta = completed_at - created_at
    return delta.total_seconds()


# =============================================================================
# Session Summary
# =============================================================================

def summarize_session_for_response(
    session_id: str,
    goal: str,
    state: str,
    created_at: datetime,
    thought_count: int = 0,
    persistent_traits: Optional[list] = None
) -> dict:
    """
    Build a session summary for API responses.
    
    Args:
        session_id: Session UUID
        goal: Session goal
        state: Current state
        created_at: Creation timestamp
        thought_count: Number of thoughts
        persistent_traits: Optional traits loaded
        
    Returns:
        Summary dict for response
    """
    response = {
        "success": True,
        "session_id": session_id,
        "goal": goal,
        "state": state,
        "created_at": created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at),
        "thought_count": thought_count
    }
    
    if persistent_traits:
        response["persistent_traits_loaded"] = len(persistent_traits)
        response["v22_feature"] = "persistent_traits"
    
    return response


# =============================================================================
# Continuation Sessions
# =============================================================================

def build_continuation_context(
    original_session_id: str,
    original_goal: str,
    original_context: Optional[dict] = None
) -> dict:
    """
    Build context for a continuation session.
    
    Args:
        original_session_id: Parent session ID
        original_goal: Original session goal
        original_context: Original session context
        
    Returns:
        Context dict for continuation
    """
    context = {
        "source": "continuation",
        "continues_session": original_session_id,
        "original_goal": original_goal
    }
    
    # Inherit persistent traits if present
    if original_context:
        if "persistent_traits" in original_context:
            context["persistent_traits"] = original_context["persistent_traits"]
        if "user_id" in original_context:
            context["user_id"] = original_context["user_id"]
    
    return context
