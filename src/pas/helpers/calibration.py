"""
PAS Calibration Helpers (v40 Phase 3)

Calibrated Self-Rewarding (NeurIPS 2024 CSR) implementation:
- Brier score computation for calibration measurement
- Overconfidence detection and warnings
- Outcome mapping for partial results
"""

from typing import Dict, Any, List, Optional


def _get_calibration_config() -> dict:
    """Load calibration config from PAS_CONFIG or use defaults."""
    try:
        from pas.server import PAS_CONFIG
        return PAS_CONFIG.get("calibration", {})
    except ImportError:
        # Fallback for standalone testing
        return {}


# Outcome mapping for nuanced calibration
OUTCOME_MAPPING = {
    "success": 1.0,
    "partial": 0.5,
    "failure": 0.0
}

# Calibration thresholds (with config fallback)
MIN_SAMPLES_FOR_CALIBRATION = 10
BRIER_WARNING_THRESHOLD = 0.25

# v56-v58: Decay and warmup - these are lazy-loaded from config
def _get_decay_config():
    """Get decay configuration values."""
    cfg = _get_calibration_config()
    return {
        "min_samples": cfg.get("min_samples_for_decay", 20),
        "decay_rate": cfg.get("decay_rate", 0.5),
        "bias_threshold": cfg.get("bias_threshold", 0.1),
    }

# Convenience accessors for backward compatibility
MIN_SAMPLES_FOR_DECAY = 20  # Default, overridden by config at runtime
DECAY_RATE = 0.5  # Default, overridden by config at runtime
BIAS_THRESHOLD_FOR_DECAY = 0.1  # Default, overridden by config at runtime


def map_outcome_to_numeric(outcome: str) -> float:
    """
    Map string outcome to numeric value for calibration.
    
    Args:
        outcome: 'success', 'partial', or 'failure'
        
    Returns:
        Numeric outcome: 1.0, 0.5, or 0.0
    """
    return OUTCOME_MAPPING.get(outcome.lower(), 0.0)


def compute_brier_score(records: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute Brier score (mean squared error) for calibration.
    
    Lower is better: 0.0 = perfect calibration, 1.0 = worst possible.
    
    Args:
        records: List of {predicted_confidence, actual_outcome}
        
    Returns:
        Brier score or None if insufficient samples
    """
    if len(records) < MIN_SAMPLES_FOR_CALIBRATION:
        return None
    
    total_squared_error = 0.0
    for record in records:
        predicted = record.get("predicted_confidence", 0.5)
        actual = record.get("actual_outcome", 0.0)
        
        squared_error = (predicted - actual) ** 2
        total_squared_error += squared_error
    
    return total_squared_error / len(records)


def compute_overconfidence_bias(records: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute overconfidence bias (mean predicted - mean actual).
    
    Positive = overconfident, Negative = underconfident.
    
    Args:
        records: List of {predicted_confidence, actual_outcome}
        
    Returns:
        Bias value or None if insufficient samples
    """
    if len(records) < MIN_SAMPLES_FOR_CALIBRATION:
        return None
    
    total_predicted = sum(r.get("predicted_confidence", 0.5) for r in records)
    total_actual = sum(r.get("actual_outcome", 0.0) for r in records)
    
    mean_predicted = total_predicted / len(records)
    mean_actual = total_actual / len(records)
    
    return mean_predicted - mean_actual


def compute_confidence_decay(
    stated_confidence: float,
    overconfidence_bias: float,
    decay_rate: Optional[float] = None
) -> tuple[float, Dict[str, Any]]:
    """
    Apply calibration decay to stated confidence.
    
    v56: When overconfident, reduce stated confidence proportionally.
    
    Args:
        stated_confidence: Original confidence (0.0-1.0)
        overconfidence_bias: Current bias (+ve = overconfident)
        decay_rate: Override decay multiplier (default from config)
        
    Returns:
        Tuple of (adjusted_confidence, decay_info)
    """
    cfg = _get_decay_config()
    rate = decay_rate if decay_rate is not None else cfg["decay_rate"]
    threshold = cfg["bias_threshold"]
    
    if overconfidence_bias <= threshold:
        return stated_confidence, {"applied": False, "reason": "bias below threshold"}
    
    # decay = confidence * (1 - bias * decay_rate)
    adjustment = overconfidence_bias * rate
    adjusted = stated_confidence * (1 - adjustment)
    
    # Clamp to valid range
    adjusted = max(0.1, min(adjusted, stated_confidence))
    
    return adjusted, {
        "applied": True,
        "original": stated_confidence,
        "adjusted": round(adjusted, 4),
        "decay_factor": round(adjustment, 4),
        "bias": round(overconfidence_bias, 4)
    }



def compute_calibration_stats(
    records: List[Dict[str, Any]],
    domain: Optional[str] = None  # v57: Domain stratification
) -> Dict[str, Any]:
    """
    Compute comprehensive calibration statistics.
    
    Args:
        records: List of calibration records
        domain: Optional domain filter for stratification (v57)
        
    Returns:
        Dict with brier_score, overconfidence_bias, sample_count, warning, domain
    """
    # Filter by domain if specified
    if domain:
        records = [r for r in records if r.get("domain_id") == domain]
    
    sample_count = len(records)
    
    brier_score = compute_brier_score(records)
    bias = compute_overconfidence_bias(records)
    
    # Determine if warning should be issued
    warning = False
    warning_message = None
    
    if brier_score is not None and brier_score > BRIER_WARNING_THRESHOLD:
        warning = True
        warning_message = f"Calibration warning: Brier score {brier_score:.3f} exceeds threshold {BRIER_WARNING_THRESHOLD}"
    
    if bias is not None and bias > 0.15:
        warning = True
        bias_msg = f"Overconfidence detected: bias = +{bias:.3f}"
        warning_message = f"{warning_message}. {bias_msg}" if warning_message else bias_msg
    
    return {
        "brier_score": round(brier_score, 4) if brier_score is not None else None,
        "overconfidence_bias": round(bias, 4) if bias is not None else None,
        "sample_count": sample_count,
        "sufficient_samples": sample_count >= MIN_SAMPLES_FOR_CALIBRATION,
        "sufficient_for_decay": sample_count >= MIN_SAMPLES_FOR_DECAY,  # v58
        "domain": domain,  # v57
        "warning": warning,
        "warning_message": warning_message
    }


def should_warn_calibration(stats: Dict[str, Any]) -> bool:
    """
    Check if calibration warning should be issued.
    
    Args:
        stats: Output from compute_calibration_stats
        
    Returns:
        True if warning should be shown
    """
    return stats.get("warning", False)


def format_calibration_for_response(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format calibration stats for inclusion in API response.
    
    Args:
        stats: Output from compute_calibration_stats
        
    Returns:
        Formatted dict suitable for API response
    """
    result = {
        "sample_count": stats["sample_count"],
        "sufficient_samples": stats["sufficient_samples"]
    }
    
    if stats["sufficient_samples"]:
        result["brier_score"] = stats["brier_score"]
        result["overconfidence_bias"] = stats["overconfidence_bias"]
        if stats["warning"]:
            result["warning"] = stats["warning_message"]
    
    return result


# =============================================================================
# v53 Auto-Deflation: Critique Depth Penalty
# =============================================================================

# Penalty thresholds
CRITIQUE_DEPTH_THRESHOLD = 2  # Minimum depth before penalty
CRITIQUE_DEPTH_PENALTY_PER_LEVEL = 0.1  # Penalty per missing level
MAX_HEURISTIC_PENALTY = 0.4  # Cap total penalties


def compute_critique_depth(conn, node_id: str) -> int:
    """
    Compute the critique tree depth for a node.
    
    Depth = number of critique nodes in the ancestry chain.
    
    Args:
        conn: Database connection
        node_id: The thought node UUID
        
    Returns:
        Critique depth (0 if no critiques)
    """
    cur = conn.cursor()
    
    # Count critique records for this node
    cur.execute(
        """
        SELECT COUNT(*) FROM critique_records 
        WHERE node_id = %s
        """,
        (node_id,)
    )
    row = cur.fetchone()
    return row[0] if row else 0


def compute_critique_depth_penalty(conn, node_id: str) -> Dict[str, Any]:
    """
    Compute penalty for shallow critique depth.
    
    v53 Auto-Deflation: Nodes with depth < 2 are penalized.
    
    Args:
        conn: Database connection
        node_id: The thought node UUID
        
    Returns:
        Dict with depth, penalty, and explanation
    """
    depth = compute_critique_depth(conn, node_id)
    
    if depth >= CRITIQUE_DEPTH_THRESHOLD:
        return {
            "depth": depth,
            "penalty": 0.0,
            "explanation": None
        }
    
    missing_levels = CRITIQUE_DEPTH_THRESHOLD - depth
    raw_penalty = missing_levels * CRITIQUE_DEPTH_PENALTY_PER_LEVEL
    penalty = min(raw_penalty, MAX_HEURISTIC_PENALTY)
    
    return {
        "depth": depth,
        "penalty": penalty,
        "explanation": f"Shallow critique depth ({depth} < {CRITIQUE_DEPTH_THRESHOLD}): -{penalty:.2f}"
    }


# =============================================================================
# v55 Confidence Nudge: Evidence Ratio
# =============================================================================

EVIDENCE_RATIO_THRESHOLD = 0.5  # Minimum ratio before warning
HIGH_CONFIDENCE_THRESHOLD = 0.8  # Confidence level that triggers check


def compute_evidence_ratio(supporting_laws_count: int, confidence: float) -> float:
    """
    Compute evidence ratio for confidence calibration.
    
    Higher ratio = more grounded confidence.
    
    Args:
        supporting_laws_count: Number of supporting laws/critiques
        confidence: Stated confidence (0.0-1.0)
        
    Returns:
        Evidence ratio (supporting_laws / confidence), capped at 2.0
    """
    if confidence <= 0:
        return 2.0  # Max ratio if confidence is 0
    
    ratio = supporting_laws_count / confidence
    return min(ratio, 2.0)


def check_confidence_nudge(supporting_laws_count: int, confidence: float) -> Optional[Dict[str, Any]]:
    """
    Check if confidence nudge warning should be issued.
    
    v55: Warns when confidence > 0.8 but evidence ratio < 0.5
    
    Args:
        supporting_laws_count: Number of supporting evidence items
        confidence: Stated confidence (0.0-1.0)
        
    Returns:
        Warning dict if nudge needed, None otherwise
    """
    if confidence <= HIGH_CONFIDENCE_THRESHOLD:
        return None
    
    ratio = compute_evidence_ratio(supporting_laws_count, confidence)
    
    if ratio >= EVIDENCE_RATIO_THRESHOLD:
        return None
    
    return {
        "type": "confidence_nudge",
        "message": f"High confidence ({confidence:.2f}) with low evidence (ratio={ratio:.2f}). Consider lowering confidence.",
        "evidence_ratio": round(ratio, 3),
        "confidence": confidence,
        "supporting_count": supporting_laws_count
    }
