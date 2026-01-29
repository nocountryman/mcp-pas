"""
PAS Shared Utilities

Database connections, embeddings, and common helpers used across all modules.
"""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    """Create a new database connection."""
    return psycopg2.connect(
        dbname=os.getenv("PAS_DB_NAME", "pas"),
        user=os.getenv("PAS_DB_USER", "postgres"),
        password=os.getenv("PAS_DB_PASSWORD", "postgres"),
        host=os.getenv("PAS_DB_HOST", "localhost"),
        port=os.getenv("PAS_DB_PORT", "5432"),
        cursor_factory=RealDictCursor
    )


def safe_close_connection(conn):
    """
    Safely close a database connection with rollback.
    
    Always rolls back before closing to prevent stuck transactions
    caused by uncommitted errors. Commit must be called explicitly.
    """
    if conn:
        try:
            conn.rollback()  # Clean up any uncommitted transaction
        except Exception:
            pass  # Ignore rollback errors
        try:
            conn.close()
        except Exception:
            pass  # Ignore close errors


# =============================================================================
# Embedding Generation
# =============================================================================

# Lazy-loaded embedding model
_embedding_model = None


def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def get_embedding(text: str) -> list[float]:
    """Generate a 384-dim embedding for the given text."""
    model = get_embedding_model()
    return model.encode(text).tolist()


# =============================================================================
# Negation Detection (v10a NLI)
# =============================================================================

NEGATION_PATTERNS = frozenset([
    "not", "no", "never", "neither", "none", "without",
    "cannot", "can't", "won't", "wouldn't", "shouldn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "doesn't", "don't", "didn't", "couldn't", "mustn't"
])


def detect_negation(text: str) -> set:
    """Detect negation patterns in text for v10a NLI MVP."""
    words = set(text.lower().split())
    return words & NEGATION_PATTERNS


# =============================================================================
# Text Utilities
# =============================================================================

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def normalize_tag(tag: str) -> str:
    """Normalize a tag to lowercase, trimmed."""
    return tag.strip().lower()


# =============================================================================
# UUID Validation
# =============================================================================

import re

UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def is_valid_uuid(value: str) -> bool:
    """Check if string is a valid UUID format."""
    return bool(UUID_PATTERN.match(value))


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate and return UUID, raising ValidationError if invalid."""
    from errors import ValidationError
    if not is_valid_uuid(value):
        raise ValidationError(field_name, f"Invalid UUID format: {value}")
    return value


# =============================================================================
# Confidence Validation 
# =============================================================================

def validate_confidence(value: float, field_name: str = "confidence") -> float:
    """Validate confidence is in 0.0-1.0 range."""
    from errors import InvalidConfidenceError
    if not (0.0 <= value <= 1.0):
        raise InvalidConfidenceError(field_name, value)
    return value


# =============================================================================
# Outcome Validation
# =============================================================================

VALID_OUTCOMES = frozenset(["success", "partial", "failure"])


def validate_outcome(outcome: str) -> str:
    """Validate outcome is success/partial/failure."""
    from errors import InvalidOutcomeError
    if outcome not in VALID_OUTCOMES:
        raise InvalidOutcomeError(outcome)
    return outcome
