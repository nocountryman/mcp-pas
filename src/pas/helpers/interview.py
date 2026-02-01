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
# v21/v35: Trait Inference Rules (moved from server.py for Phase 5)
# =============================================================================

# Pattern: (hidden_value_substring, min_occurrences, trait_name, confidence)
TRAIT_RULES = [
    ("SAFETY", 2, "RISK_AVERSE", 0.8),
    ("SIMPLICITY", 2, "MINIMALIST", 0.75),
    ("POWER_USER", 2, "CONTROL_ORIENTED", 0.8),
    ("AI_", 2, "AUTOMATION_TRUSTING", 0.7),
    ("BEGINNER", 2, "ACCESSIBILITY_FOCUSED", 0.75),
    ("BALANCE", 3, "PRAGMATIST", 0.7),
    ("EXPLICIT", 2, "EXPLICIT_CONTROL", 0.8),
    ("PERFORMANCE", 2, "SPEED_FOCUSED", 0.75),
    ("ERROR_PREVENTION", 2, "SAFETY_CONSCIOUS", 0.8),
    ("FREEDOM", 2, "AUTONOMY_FOCUSED", 0.75),
    # Legacy v21 patterns (kept for compatibility)
    ("THOROUGH", 2, "DETAIL_ORIENTED", 0.80),
    ("QUICK", 2, "EFFICIENCY_FOCUSED", 0.75),
    ("SAFE", 2, "RISK_AVERSE", 0.80),
    ("BOLD", 2, "RISK_TOLERANT", 0.75),
]


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


# =============================================================================
# Phase 5: check_interview_complete Helper Functions
# =============================================================================

def infer_traits_from_hidden_values(
    hidden_value_counts: dict[str, int]
) -> list[dict[str, Any]]:
    """
    v35: Infer latent traits from hidden value patterns.
    
    Args:
        hidden_value_counts: Dict of hidden_value -> occurrence count
    
    Returns:
        List of trait dicts with trait, confidence, evidence_count
    """
    latent_traits = []
    
    for pattern, min_count, trait_name, confidence in TRAIT_RULES:
        matching_count = sum(
            count for hv, count in hidden_value_counts.items() 
            if pattern in hv.upper()
        )
        if matching_count >= min_count:
            latent_traits.append({
                "trait": trait_name,
                "confidence": confidence,
                "evidence_count": matching_count
            })
    
    return latent_traits


def build_context_summary(answered_questions: list[dict]) -> dict:
    """
    Build Q/A summary from answered questions. Pure function.
    
    Args:
        answered_questions: List of answered question dicts
        
    Returns:
        Dict mapping question_id -> {question, answer, selected}
    """
    context_summary = {}
    for q in answered_questions:
        selected = None
        for c in q.get("choices", []):
            if c["label"] == q.get("answer"):
                selected = c.get("description") or c.get("text")
                break
        context_summary[q["id"]] = {
            "question": q.get("question_text"),
            "answer": q.get("answer"),
            "selected": selected
        }
    return context_summary


def aggregate_hidden_values(answer_history: list[dict]) -> dict[str, int]:
    """
    Count hidden_value occurrences from history. Pure function.
    
    Args:
        answer_history: List of answer history entries
        
    Returns:
        Dict mapping hidden_value -> occurrence count
    """
    counts: dict[str, int] = {}
    for entry in answer_history:
        hv = entry.get("hidden_value")
        if hv:
            counts[hv] = counts.get(hv, 0) + 1
    return counts


def run_semantic_trait_matching(
    cur,
    unmatched_descriptions: list[str],
    get_embedding_fn,
    existing_traits: list[dict],
    threshold: float = 0.65
) -> tuple[list[dict], list[dict]]:
    """
    v22: Semantic fallback for unmatched descriptions.
    
    Args:
        cur: Database cursor
        unmatched_descriptions: Text descriptions that didn't match TRAIT_RULES
        get_embedding_fn: Embedding function (passed explicitly for purity)
        existing_traits: Traits already inferred from hidden values
        threshold: Similarity threshold for matching
        
    Returns:
        Tuple of (semantic_matches, updated_latent_traits)
    """
    import logging
    logger = logging.getLogger(__name__)
    semantic_matches = []
    latent_traits = list(existing_traits)  # Copy to avoid mutation
    
    try:
        cur.execute("SELECT COUNT(*) FROM trait_exemplars WHERE embedding IS NOT NULL")
        row = cur.fetchone()
        count = row[0] if isinstance(row, tuple) else row.get("count", 0)
        if count == 0:
            return semantic_matches, latent_traits
        
        for desc_text in unmatched_descriptions[:5]:  # Limit to 5
            desc_embedding = get_embedding_fn(desc_text)
            cur.execute("""
                SELECT trait_name, 1 - (embedding <=> %s::vector) as similarity
                FROM trait_exemplars WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT 1
            """, (desc_embedding, desc_embedding))
            
            result = cur.fetchone()
            if result:
                similarity = result["similarity"] if isinstance(result, dict) else result[1]
                trait_name = result["trait_name"] if isinstance(result, dict) else result[0]
                
                if similarity > threshold:
                    semantic_matches.append({
                        "trait": trait_name,
                        "similarity": round(float(similarity), 3),
                        "source_text": desc_text[:50]
                    })
                    existing = next((t for t in latent_traits if t["trait"] == trait_name), None)
                    if existing:
                        existing["evidence_count"] = existing.get("evidence_count", 1) + 1
                        existing["semantic_boost"] = True
                    else:
                        latent_traits.append({
                            "trait": trait_name, "confidence": 0.6,
                            "evidence_count": 1, "source": "semantic"
                        })
                    logger.info(f"v22 Semantic: '{desc_text[:30]}' → {trait_name}")
    except Exception as e:
        logger.warning(f"v22 Hybrid inference failed: {e}")
    
    return semantic_matches, latent_traits


def collect_unmatched_descriptions(
    answer_history: list[dict],
    trait_rules: list[tuple]
) -> list[str]:
    """
    v22: Collect descriptions that didn't match any trait rule pattern.
    
    Args:
        answer_history: Answer history entries
        trait_rules: TRAIT_RULES list for pattern matching
        
    Returns:
        List of unmatched description strings
    """
    unmatched = []
    for entry in answer_history:
        desc = entry.get("answer_description")
        hv = entry.get("hidden_value")
        if not desc:
            continue
        if not hv or not any(p in (hv or "").upper() for p, _, _, _ in trait_rules):
            unmatched.append(desc)
    return unmatched


def run_trait_inference(
    cur,
    interview: dict,
    get_embedding_fn
) -> tuple[list[dict], dict[str, int]]:
    """
    Phase 5: Orchestrate full trait inference pipeline.
    
    Combines: aggregate_hidden_values + infer_traits + collect_unmatched + semantic matching.
    
    Args:
        cur: Database cursor
        interview: Interview context dict
        get_embedding_fn: Embedding function for semantic matching
        
    Returns:
        Tuple of (latent_traits, hidden_value_counts)
    """
    answer_history = interview.get("answer_history", [])
    
    # Step 1: Aggregate hidden values
    hidden_value_counts = aggregate_hidden_values(answer_history)
    
    # Step 2: Rule-based trait inference
    latent_traits = infer_traits_from_hidden_values(hidden_value_counts)
    
    # Step 3: Semantic fallback for unmatched descriptions
    unmatched = collect_unmatched_descriptions(answer_history, TRAIT_RULES)
    if unmatched:
        _, latent_traits = run_semantic_trait_matching(
            cur, unmatched, get_embedding_fn, latent_traits
        )
    
    return latent_traits, hidden_value_counts


def persist_interview_context(
    cur,
    conn,
    session_id: str,
    goal: str,
    context: dict,
    interview: dict,
    latent_traits: list[dict],
    hidden_value_counts: dict[str, int],
    archive_fn
) -> bool:
    """
    Phase 5: Persist trait context and archive interview if complete.
    
    Consolidates the persistence and archival logic.
    
    Args:
        cur: Database cursor
        conn: Database connection  
        session_id: Session UUID
        goal: Session goal text
        context: Full session context dict
        interview: Interview context dict
        latent_traits: Inferred traits
        hidden_value_counts: Hidden value counts
        archive_fn: The archive_interview_to_history function
        
    Returns:
        True if archived, False otherwise
    """
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Store traits in context
    if latent_traits or hidden_value_counts:
        context["latent_traits"] = latent_traits
        context["hidden_value_counts"] = hidden_value_counts
        logger.info(f"v21: Inferred {len(latent_traits)} traits from {sum(hidden_value_counts.values())} hidden values")
    
    # Archive if not already archived
    if not interview.get("archived"):
        try:
            archive_fn(cur, session_id, goal, interview)
            interview["archived"] = True
            cur.execute(
                "UPDATE reasoning_sessions SET context = %s WHERE id = %s",
                (json.dumps(context), session_id)
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Failed to archive interview: {e}")
            conn.rollback()
    return interview.get("archived", False)


# =============================================================================
# Phase 7: submit_answer Helpers
# =============================================================================

def find_and_validate_question(
    pending_questions: list[dict],
    question_id: str
) -> tuple[Optional[dict], Optional[str]]:
    """
    Find a question by ID and validate it can be answered.
    
    Args:
        pending_questions: List of pending interview questions
        question_id: The question ID to find
        
    Returns:
        (question, None) if valid and ready to answer
        (None, error_message) if invalid or already answered
    """
    for q in pending_questions:
        if q.get("id") == question_id:
            if q.get("answered"):
                return None, "Question already answered"
            return q, None
    return None, f"Question {question_id} not found"


def record_answer_with_choice(
    question: dict,
    answer: str,
    interview: dict
) -> tuple[Optional[str], Optional[str]]:
    """
    Mark question answered, extract choice details, append to history.
    
    Args:
        question: The question dict to mark answered
        answer: The answer label (e.g., "A", "B", "C")
        interview: The interview context dict (mutated)
        
    Returns:
        (hidden_value, answer_description) from selected choice
    """
    import uuid
    
    question["answered"] = True
    question["answer"] = answer
    
    hidden_value = None
    answer_description = None
    for choice in question.get("choices", []):
        if choice.get("label") == answer:
            hidden_value = choice.get("hidden_value")
            answer_description = choice.get("description")
            break
    
    interview["answer_history"].append({
        "question_id": question.get("id"),
        "question_text": question.get("question_text"),
        "answer": answer,
        "answer_description": answer_description,
        "hidden_value": hidden_value,
        "timestamp": str(uuid.uuid4())[:8]
    })
    
    return hidden_value, answer_description


def process_answer_side_effects(
    cur,
    conn,
    session_id: str,
    question: dict,
    answer: str,
    hidden_value: Optional[str],
    pending: list[dict],
    interview: dict,
    config: dict,
    context: dict,
    logger
) -> tuple[Optional[str], list[str]]:
    """
    Process v19 dimension tracking, follow-up injection, v23 evidence tracking.
    
    Combines three related side effects that share lifecycle and data flow:
    1. v19: Track dimension coverage and persist to DB
    2. Follow-up injection based on answer rules
    3. v23: Evidence tracking for plateau detection
    
    Args:
        cur: Database cursor
        conn: Database connection (for rollback on v19 failure)
        session_id: The reasoning session UUID
        question: The answered question dict
        answer: The answer label
        hidden_value: Extracted hidden value (may be None)
        pending: List of pending questions (mutated for follow-ups)
        interview: Interview context dict (mutated for evidence tracking)
        config: Interview config dict (mutated for remaining count)
        context: Full session context (for dimension coverage)
        logger: Logger instance
        
    Returns:
        (dimension_covered, list_of_injected_question_ids)
    """
    dimension_covered = None
    injected = []
    
    # -------------------------------------------------------------------------
    # v19: Track dimension coverage
    # -------------------------------------------------------------------------
    if question.get("source") == "domain_dimension" and question.get("dimension_id"):
        dim_id = question["dimension_id"]
        dimension_coverage = context.get("dimension_coverage", {})
        
        if dim_id in dimension_coverage:
            dimension_coverage[dim_id]["covered"] = True
            dimension_covered = dimension_coverage[dim_id]["name"]
            logger.info(f"v19: Dimension '{dimension_covered}' covered")
        
        # Persist answer to interview_answers table
        try:
            answer_text = answer
            for choice in question.get("choices", []):
                if choice.get("label") == answer:
                    answer_text = choice.get("description", answer)
                    break
            
            cur.execute("""
                INSERT INTO interview_answers (session_id, dimension_id, question_id, answer_label, answer_text)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (session_id, dimension_id) DO UPDATE SET
                    answer_label = EXCLUDED.answer_label,
                    answer_text = EXCLUDED.answer_text,
                    created_at = NOW()
            """, (session_id, dim_id, question.get("id"), answer, answer_text))
            
            logger.info(f"v19: Persisted answer for dimension {dim_id}")
        except Exception as e:
            conn.rollback()
            logger.warning(f"v19: Failed to persist answer: {e}")
    
    # -------------------------------------------------------------------------
    # Follow-up injection
    # -------------------------------------------------------------------------
    follow_up_rules = question.get("follow_up_rules", [])
    for rule in follow_up_rules:
        if rule.get("when_answer") == answer:
            new_q = rule["inject"].copy()
            
            # Safety checks
            if new_q.get("depth", 1) > config["max_depth"]:
                logger.warning(f"Skipping injection: depth {new_q['depth']} exceeds max {config['max_depth']}")
                continue
            if len(pending) >= config["max_questions"]:
                logger.warning(f"Skipping injection: max questions {config['max_questions']} reached")
                continue
            
            new_q["answered"] = False
            new_q["answer"] = None
            pending.append(new_q)
            injected.append(new_q["id"])
    
    # Update remaining count
    config["questions_remaining"] = len([q for q in pending if not q.get("answered")])
    
    # -------------------------------------------------------------------------
    # v23: Evidence tracking for plateau detection
    # -------------------------------------------------------------------------
    evidence_history = interview.get("evidence_history", [])
    evidence_delta = 1 if hidden_value else 0
    evidence_history.append(evidence_delta)
    interview["evidence_history"] = evidence_history[-5:]  # Keep last 5
    
    # Check for plateau (last 3 answers with no new trait info)
    if len(evidence_history) >= 3:
        recent_gain = sum(evidence_history[-3:])
        if recent_gain == 0 and config.get("questions_answered", 0) >= 3:
            interview["early_termination_suggested"] = True
            interview["early_termination_reason"] = "diminishing_returns"
            logger.info("v23: Early termination suggested - no trait evidence in last 3 answers")
    
    return dimension_covered, injected


# =============================================================================
# Phase 8: identify_gaps helpers
# =============================================================================


def query_historical_failures(cur, conn, session_id: str, logger) -> list[dict]:
    """
    v16d.2: Query similar failed goals via embedding similarity.
    
    Args:
        cur: Database cursor
        conn: Database connection (for rollback on failure)
        session_id: The reasoning session UUID
        logger: Logger instance
        
    Returns:
        List of historical question dicts with source="historical_failure"
    """
    historical_questions: list[dict] = []
    try:
        cur.execute("""
            SELECT s.goal, o.notes, o.failure_reason
            FROM outcome_records o
            JOIN reasoning_sessions s ON o.session_id = s.id
            WHERE o.outcome != 'success'
            AND (o.notes IS NOT NULL OR o.failure_reason IS NOT NULL)
            AND s.goal_embedding IS NOT NULL
            AND 1 - (s.goal_embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)) > 0.6
            LIMIT 3
        """, (session_id,))
        
        for row in cur.fetchall():
            reason = row["failure_reason"] or row["notes"]
            if reason:
                historical_questions.append({
                    "id": f"hist_{len(historical_questions)+1}",
                    "question_text": f"A similar goal '{row['goal'][:50]}...' failed because: {reason[:100]}. How will you address this?",
                    "question_type": "open",
                    "choices": [],
                    "priority": 5,
                    "depth": 1,
                    "depends_on": [],
                    "follow_up_rules": [],
                    "answered": False,
                    "answer": None,
                    "source": "historical_failure"
                })
    except Exception as e:
        conn.rollback()
        logger.warning(f"v16d.2 historical query failed: {e}")
    
    return historical_questions


def detect_domains(cur, session_id: str, logger) -> list[dict]:
    """
    v19: Detect domains via embedding similarity.
    
    Args:
        cur: Database cursor
        session_id: The reasoning session UUID
        logger: Logger instance
        
    Returns:
        List of detected domain dicts: [{"id": "...", "name": "...", "similarity": float}]
    """
    detected_domains: list[dict] = []
    try:
        cur.execute("""
            SELECT id, domain_name, description,
                   1 - (embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)) as similarity
            FROM interview_domains
            WHERE is_active = true
            AND embedding IS NOT NULL
            ORDER BY embedding <=> (SELECT goal_embedding FROM reasoning_sessions WHERE id = %s)
            LIMIT 2
        """, (session_id, session_id))
        
        similar_domains = cur.fetchall()
        
        for row in similar_domains:
            if row["similarity"] > 0.5 or not detected_domains:
                detected_domains.append({
                    "id": str(row["id"]),
                    "name": row["domain_name"],
                    "similarity": row["similarity"]
                })
        
        if detected_domains:
            logger.info(f"v19: Detected domains: {[d['name'] for d in detected_domains]}")
    except Exception as e:
        logger.warning(f"v19 domain detection failed: {e}")
    
    return detected_domains


def load_dimension_questions(cur, domain_ids: list, context: dict, logger) -> tuple[list[dict], dict]:
    """
    v19: Load dimension questions for detected domains.
    
    Args:
        cur: Database cursor
        domain_ids: List of domain UUIDs
        context: Session context dict (not mutated)
        logger: Logger instance
        
    Returns:
        (domain_questions, dimension_coverage) tuple
    """
    domain_questions: list[dict] = []
    dimension_coverage: dict = {}
    
    try:
        cur.execute("""
            SELECT dim.id, dim.dimension_name, dim.description, dim.is_required, dim.priority,
                   d.domain_name,
                   q.question_template, q.question_type, q.choices
            FROM interview_dimensions dim
            JOIN interview_domains d ON dim.domain_id = d.id
            LEFT JOIN interview_questions q ON q.dimension_id = dim.id AND q.is_default = true
            WHERE dim.domain_id = ANY(%s::uuid[])
            ORDER BY dim.priority ASC
        """, (domain_ids,))
        
        dimensions = cur.fetchall()
        
        for dim in dimensions:
            dim_id = str(dim["id"])
            dimension_coverage[dim_id] = {
                "name": dim["dimension_name"],
                "domain": dim["domain_name"],
                "is_required": dim["is_required"],
                "covered": False
            }
            
            if dim["question_template"]:
                choices = dim["choices"] if dim["choices"] else []
                domain_questions.append({
                    "id": f"dim_{dim['dimension_name']}",
                    "dimension_id": dim_id,
                    "question_text": dim["question_template"],
                    "question_type": dim["question_type"],
                    "choices": choices,
                    "priority": dim["priority"],
                    "depth": 1,
                    "depends_on": [],
                    "follow_up_rules": [],
                    "answered": False,
                    "answer": None,
                    "source": "domain_dimension",
                    "domain": dim["domain_name"],
                    "dimension": dim["dimension_name"],
                    "is_required": dim["is_required"]
                })
        
        logger.info(f"v19: Generated {len(domain_questions)} dimension questions")
    except Exception as e:
        logger.warning(f"v19 load dimensions failed: {e}")
    
    return domain_questions, dimension_coverage


def build_goal_question_prompt(goal: str) -> dict:
    """
    v21: Build LLM prompt for goal-derived questions.
    
    Args:
        goal: The session goal text
        
    Returns:
        Dict with prompt, system, and max_questions
    """
    goal_prompt = f"""Analyze this goal and generate clarifying questions if needed:

GOAL: {goal}

DECISION RULES:
1. If goal is SPECIFIC (names files/functions, has clear constraints) → return []
2. If goal is AMBIGUOUS (multiple valid interpretations) → return 1-2 questions max
3. If goal is OPEN-ENDED (design/architecture/planning) → return 2-3 questions max

EXAMPLES OF SPECIFIC GOALS (return []):
- "Fix bug in auth.py line 45"
- "Add logging to UserService.create_user()"
- "Refactor parse_terminal_output to handle '0 failed'"

=== HIDDEN CONTEXT QUESTION DESIGN (v21) ===
Use psychology-based techniques to extract more information per question:

1. CONSEQUENCE FRAMING (Laddering):
   - Frame choices as CONSEQUENCES, not features
   - BAD: "Which database?" → "PostgreSQL / MySQL / MongoDB"
   - GOOD: "When data integrity fails, what's the worst outcome?"
          → "Users lose trust / Debugging becomes hard / Schema breaks"

2. TRADE-OFF BUNDLES (Conjoint):
   - Force implicit priority decisions
   - BAD: "Is performance important?" (everyone says yes)
   - GOOD: "Which trade-off can you live with?"
          → "50ms slower but consistent / Fast but occasional stale reads"

3. SCENARIO COMPLETION (Projective):
   - Present a situation and ask what happens next
   - GOOD: "A critical bug appears at 2am before launch. When you investigate, you find..."
          → Options reveal risk tolerance, debugging philosophy

4. COGNITIVE LOAD (Keep it simple):
   - Options < 15 words each
   - Use concrete scenarios, not abstractions
   - Gut reaction = true preference

WHAT EACH CHOICE REVEALS:
- Every option should reveal an underlying VALUE (risk tolerance, priority, philosophy)
- Include a "hidden_value" field describing what the choice reveals
- Example: {{"label": "A", "description": "Fast but complex", "hidden_value": "performance_priority"}}

Return format:
[{{"question_text": "...", "choices": [{{"label": "A", "description": "...", "hidden_value": "...", "pros": ["..."], "cons": ["..."]}}]}}]

Return [] if goal is specific enough. Return ONLY the JSON array."""
    
    return {
        "prompt": goal_prompt,
        "system": "You are a structured question generator. Return only valid JSON arrays.",
        "max_questions": 3
    }


def prioritize_questions(
    domain_questions: list[dict],
    goal_questions: list[dict],
    historical_questions: list[dict],
    goal: str
) -> list[dict]:
    """
    Combine and prioritize questions, add catchall if empty.
    
    Priority order: historical (highest) > domain > goal > catchall
    
    Args:
        domain_questions: v19 domain-based questions
        goal_questions: v21 LLM-generated questions (may be empty)
        historical_questions: v16d.2 failure-based questions
        goal: Session goal (for catchall generation)
        
    Returns:
        Combined and prioritized question list
    """
    # v19: Prefer domain questions over LLM-generated
    if domain_questions:
        questions = domain_questions
    elif goal_questions:
        questions = goal_questions
    else:
        questions = []
    
    # v18: If no questions at all, offer catch-all
    if not questions and not historical_questions:
        goal_preview = goal[:50] + "..." if len(goal) > 50 else goal
        questions = [{
            "id": "q_catchall",
            "question_text": f"Is there anything specific about '{goal_preview}' I should know before proceeding?",
            "question_type": "open",
            "choices": [],
            "optional": True,
            "priority": 100,
            "depth": 1,
            "depends_on": [],
            "follow_up_rules": [],
            "answered": False,
            "answer": None,
            "source": "catchall"
        }]
    
    # v16d.2: Prepend historical questions (highest priority)
    return historical_questions + questions
