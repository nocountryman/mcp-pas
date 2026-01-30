"""
PAS Calibration Helpers (v40 Phase 3)

Calibrated Self-Rewarding (NeurIPS 2024 CSR) implementation:
- Brier score computation for calibration measurement
- Overconfidence detection and warnings
- Outcome mapping for partial results
"""

from typing import Dict, Any, List, Optional


# Outcome mapping for nuanced calibration
OUTCOME_MAPPING = {
    "success": 1.0,
    "partial": 0.5,
    "failure": 0.0
}

# Calibration thresholds
MIN_SAMPLES_FOR_CALIBRATION = 10
BRIER_WARNING_THRESHOLD = 0.25


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


def compute_calibration_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive calibration statistics.
    
    Args:
        records: List of calibration records
        
    Returns:
        Dict with brier_score, overconfidence_bias, sample_count, warning
    """
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
