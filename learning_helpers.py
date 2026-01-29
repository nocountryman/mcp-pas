"""
PAS Learning Helper Functions

Pure functions for RLVR (Reinforcement Learning from Verifiable Results),
terminal output parsing, and outcome processing.
"""

import re
from typing import Any, Optional

# =============================================================================
# Terminal Output Patterns (v17a RLVR)
# =============================================================================

# Domain-agnostic patterns for success/failure detection
SUCCESS_PATTERNS = [
    r'\bPASS(?:ED)?\b',
    r'\bOK\b',
    r'\bSUCCESS(?:FUL)?\b',
    r'✓',
    r'\bAll tests passed\b',
    r'\bBuild succeeded\b',
    r'\bexit code 0\b',
    r'\b0 failed\b',
    r'\bno errors\b',
    r'\bcompleted successfully\b',
]

FAILURE_PATTERNS = [
    r'\bFAIL(?:ED|URE)?\b',
    r'\bERROR\b',
    r'\bException\b',
    r'✗',
    r'\bAssertionError\b',
    r'\bBuild failed\b',
    r'exit code [1-9]\d*',
    r'\bTraceback\b',
    r'\bSyntaxError\b',
    r'\bTypeError\b',
    r'\bValueError\b',
    r'\bAttributeError\b',
    r'\bImportError\b',
    r'\bRuntimeError\b',
    r'\bCRITICAL\b',
    r'\bFATAL\b',
]

# v17b.2: Patterns to extract failure reasons for semantic learning
FAILURE_REASON_PATTERNS = [
    # Python exceptions with message: "ValueError: invalid literal for int()"
    r'(?P<type>\w+Error):\s*(?P<reason>.+?)(?:\n|$)',
    # Python exceptions: "Exception: something went wrong"
    r'(?P<type>Exception):\s*(?P<reason>.+?)(?:\n|$)',
    # Assertion failures: "AssertionError: expected X but got Y"
    r'AssertionError:\s*(?P<reason>.+?)(?:\n|$)',
    # pytest/jest assertion: "assert x == y" or "Expected X to equal Y"
    r'(?:assert|Assert)\s+(?P<reason>.+?)(?:\n|$)',
    r'Expected\s+(?P<reason>.+?)(?:\n|$)',
    # Build errors: "error: cannot find module 'X'"
    r'error:\s*(?P<reason>.+?)(?:\n|$)',
    # Compilation errors: "fatal error: file not found"
    r'fatal error:\s*(?P<reason>.+?)(?:\n|$)',
    # npm/node errors: "Error: Cannot find module"
    r'Error:\s*(?P<reason>.+?)(?:\n|$)',
    # Generic test failure with name: "FAILED test_foo - reason"
    r'FAILED\s+\S+\s*[-:]\s*(?P<reason>.+?)(?:\n|$)',
]


# =============================================================================
# Terminal Output Parsing (Pure Functions)
# =============================================================================

def parse_terminal_signals(terminal_text: str) -> dict[str, Any]:
    """
    Parse terminal output for success/failure signals.
    
    Uses domain-agnostic regex patterns to detect test pass/fail,
    build success/error, and runtime crashes.
    
    Args:
        terminal_text: Raw terminal output to analyze
        
    Returns:
        Dict with signal, confidence, matches, and failure_reason
    """
    if not terminal_text or not terminal_text.strip():
        return {
            "signal": "unknown",
            "confidence": 0.0,
            "matches": [],
            "failure_reason": None
        }
    
    # Case-insensitive matching
    success_matches = []
    failure_matches = []
    
    for pattern in SUCCESS_PATTERNS:
        matches = re.findall(pattern, terminal_text, re.IGNORECASE)
        if matches:
            success_matches.extend(matches)
    
    for pattern in FAILURE_PATTERNS:
        matches = re.findall(pattern, terminal_text, re.IGNORECASE)
        if matches:
            failure_matches.extend(matches)
    
    # v17a.2: Filter out false-positive failures from success contexts
    # "0 failed", "passed", etc. should not count as failure signals
    false_positive_context = re.search(r'\b0\s+failed\b', terminal_text, re.IGNORECASE)
    if false_positive_context:
        # Remove one 'failed' match for each '0 failed' context found
        zero_failed_count = len(re.findall(r'\b0\s+failed\b', terminal_text, re.IGNORECASE))
        for _ in range(zero_failed_count):
            for i, m in enumerate(failure_matches):
                if m.lower() == 'failed':
                    failure_matches.pop(i)
                    break
    
    # Determine signal and confidence
    success_count = len(success_matches)
    failure_count = len(failure_matches)
    total = success_count + failure_count
    
    if total == 0:
        signal = "unknown"
        confidence = 0.0
        all_matches = []
    elif failure_count > 0 and success_count == 0:
        signal = "failure"
        confidence = min(0.95, 0.7 + (failure_count * 0.05))
        all_matches = failure_matches
    elif success_count > 0 and failure_count == 0:
        signal = "success"
        confidence = min(0.95, 0.7 + (success_count * 0.05))
        all_matches = success_matches
    else:
        # Mixed signals - failure takes precedence
        if failure_count >= success_count:
            signal = "failure"
            confidence = 0.6
        else:
            signal = "success"
            confidence = 0.5
        all_matches = failure_matches + success_matches
    
    # v17b.2: Extract failure reason for semantic learning
    failure_reason = None
    if signal == "failure":
        failure_reason = extract_failure_reason(terminal_text)
    
    return {
        "signal": signal,
        "confidence": confidence,
        "matches": all_matches[:10],  # Limit to avoid huge responses
        "failure_reason": failure_reason
    }


def extract_failure_reason(terminal_text: str) -> Optional[str]:
    """
    Extract the most specific failure reason from terminal output.
    
    Args:
        terminal_text: Raw terminal output containing error
        
    Returns:
        Extracted failure reason string, or None
    """
    for pattern in FAILURE_REASON_PATTERNS:
        match = re.search(pattern, terminal_text, re.IGNORECASE)
        if match:
            try:
                reason = match.group('reason')
                if reason and len(reason) > 5:
                    # Clean up the reason
                    reason = reason.strip()[:200]
                    return reason
            except (IndexError, AttributeError):
                continue
    return None


def signal_to_outcome(signal: str) -> str:
    """
    Convert a terminal signal to an outcome value.
    
    Args:
        signal: 'success', 'failure', or 'unknown'
        
    Returns:
        Outcome string for record_outcome
    """
    if signal == "success":
        return "success"
    elif signal == "failure":
        return "failure"
    return "partial"


# =============================================================================
# Outcome Processing Helpers
# =============================================================================

def compute_trait_reinforcement(
    trait_name: str,
    current_score: float,
    outcome: str,
    decay_factor: float = 0.9
) -> float:
    """
    Compute new trait score based on outcome.
    
    Args:
        trait_name: Name of the trait
        current_score: Current cumulative score
        outcome: Session outcome ('success', 'partial', 'failure')
        decay_factor: Score decay multiplier per session
        
    Returns:
        New cumulative score
    """
    # Start with decay
    new_score = current_score * decay_factor
    
    # Apply outcome boost/penalty
    if outcome == "success":
        new_score += 0.2
    elif outcome == "failure":
        new_score = max(0, new_score - 0.1)
    # 'partial' keeps decayed score
    
    # Cap at 1.0
    return min(1.0, new_score)


def compute_law_success_rate(
    selection_count: int,
    success_count: int
) -> float:
    """
    Compute success rate for a law with Bayesian smoothing.
    
    Args:
        selection_count: Times law was selected
        success_count: Times it led to success
        
    Returns:
        Smoothed success rate (0.0-1.0)
    """
    # Bayesian smoothing with prior of 0.5
    alpha = success_count + 1
    beta = selection_count - success_count + 1
    return alpha / (alpha + beta)


def should_refresh_law_weights(
    outcome_count: int,
    min_samples: int = 5
) -> bool:
    """
    Determine if law weights should be refreshed.
    
    Args:
        outcome_count: Number of outcomes since last refresh
        min_samples: Minimum samples needed
        
    Returns:
        True if refresh is warranted
    """
    return outcome_count >= min_samples
