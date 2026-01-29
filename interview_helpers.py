"""
PAS Interview Helper Functions

Pure functions for interview flow management, question handling,
and domain extraction.
"""

from typing import Any, Optional

# =============================================================================
# Interview Configuration
# =============================================================================

DEFAULT_INTERVIEW_CONFIG = {
    "max_questions": 15,
    "max_depth": 3,
    "questions_answered": 0,
    "questions_remaining": 0
}

# v20: Adaptive Depth Quality Thresholds
DEFAULT_QUALITY_THRESHOLDS = {
    "gap_score": 0.10,           # Winner must be ≥10% better than runner-up
    "critique_coverage": 0.66,   # ≥66% of top candidates must be critiqued
    "min_depth": 2,              # Must explore at least 2 levels deep
    "max_confidence_variance": 0.25,  # Variance ≤0.25 for stability
    "max_iterations": 5          # Safeguard: max expansion cycles
}

# Domain extraction keywords
DOMAIN_KEYWORDS = {
    "ui": "ui_design", 
    "ux": "ui_design", 
    "dashboard": "ui_design", 
    "widget": "ui_design",
    "cache": "architecture", 
    "api": "architecture", 
    "database": "architecture", 
    "schema": "architecture",
    "debug": "debugging", 
    "fix": "debugging", 
    "error": "debugging", 
    "bug": "debugging",
    "test": "testing", 
    "verify": "testing", 
    "validate": "testing",
    "design": "design", 
    "architect": "architecture", 
    "pattern": "patterns"
}


# =============================================================================
# Pure Helper Functions
# =============================================================================

def get_interview_context(session_context: dict) -> dict:
    """Extract or initialize interview context from session."""
    if not session_context:
        session_context = {}
    
    if "interview" not in session_context:
        session_context["interview"] = {
            "config": DEFAULT_INTERVIEW_CONFIG.copy(),
            "pending_questions": [],
            "answer_history": []
        }
    return session_context["interview"]


def extract_domain_from_goal(goal: str) -> str:
    """
    Extract domain category from goal text.
    
    Args:
        goal: The session goal text
        
    Returns:
        Domain string like 'ui_design', 'architecture', etc.
    """
    goal_lower = goal.lower()
    
    for keyword, domain in DOMAIN_KEYWORDS.items():
        if keyword in goal_lower:
            return domain
    
    return "general"


def process_interview_answer(
    question: dict,
    answer: str
) -> dict:
    """
    Process an interview answer and determine follow-up actions.
    
    Args:
        question: The question dict with choices and rules
        answer: The selected answer label (e.g., 'A', 'B', 'C')
        
    Returns:
        Dict with answer_text, follow_up_triggered, and any injected questions
    """
    # Find answer text from choices
    answer_text = None
    if "choices" in question:
        for choice in question["choices"]:
            if choice.get("label") == answer:
                answer_text = choice.get("text", "")
                break
    
    # Check for follow-up triggers
    follow_up_triggered = False
    inject_questions = []
    
    for rule in question.get("follow_up_rules", []):
        if rule.get("when_answer") == answer:
            follow_up_triggered = True
            if "inject_question" in rule:
                inject_questions.append(rule["inject_question"])
            break
    
    return {
        "answer_text": answer_text,
        "follow_up_triggered": follow_up_triggered,
        "inject_questions": inject_questions
    }


def format_question_for_display(question: dict, progress: str = "") -> dict:
    """
    Format a question for user display.
    
    Args:
        question: Raw question dict
        progress: Progress indicator string (e.g., "Question 2/5")
        
    Returns:
        Formatted question dict with display-ready text
    """
    formatted_choices = []
    
    for choice in question.get("choices", []):
        choice_text = f"{choice['label']}) {choice['text']}"
        
        # Add pros/cons if present
        if choice.get("pros"):
            choice_text += f"\n   ✓ Pros: {choice['pros']}"
        if choice.get("cons"):
            choice_text += f"\n   ✗ Cons: {choice['cons']}"
        
        formatted_choices.append(choice_text)
    
    return {
        "id": question.get("id"),
        "question": question.get("question_text"),
        "choices_formatted": "\n".join(formatted_choices),
        "category": question.get("category", "general"),
        "progress": progress
    }


def compute_interview_progress(interview: dict) -> dict:
    """
    Compute progress statistics for an interview.
    
    Args:
        interview: Interview context dict
        
    Returns:
        Dict with answered, remaining, total, is_complete
    """
    pending = interview.get("pending_questions", [])
    answered = sum(1 for q in pending if q.get("answered"))
    remaining = len([q for q in pending if not q.get("answered")])
    total = len(pending)
    
    config = interview.get("config", DEFAULT_INTERVIEW_CONFIG)
    max_questions = config.get("max_questions", 15)
    
    return {
        "answered": answered,
        "remaining": remaining,
        "total": total,
        "max_allowed": max_questions,
        "is_complete": remaining == 0 or answered >= max_questions
    }


def extract_hidden_context_from_interview(interview: dict) -> dict:
    """
    Extract hidden context values from answered interview questions.
    
    Args:
        interview: Interview context dict
        
    Returns:
        Dict of context_key -> value mappings
    """
    hidden_context = {}
    
    for question in interview.get("pending_questions", []):
        if not question.get("answered"):
            continue
        
        answer = question.get("answer")
        
        # Look for hidden_context in the matching choice
        for choice in question.get("choices", []):
            if choice.get("label") == answer:
                if "hidden_context" in choice:
                    hidden_context.update(choice["hidden_context"])
                break
    
    return hidden_context


def should_skip_interview(session_context: dict) -> bool:
    """
    Determine if interview can be skipped based on context.
    
    Args:
        session_context: Full session context
        
    Returns:
        True if interview is not needed
    """
    # Skip if interview already complete
    interview = session_context.get("interview", {})
    if interview.get("pending_questions"):
        progress = compute_interview_progress(interview)
        if progress["is_complete"]:
            return True
    
    # Skip if explicit skip flag
    if session_context.get("skip_interview"):
        return True
    
    return False
