"""
PAS Error Hierarchy

Standardized error handling for all PAS modules.
All exceptions inherit from PASError for consistent handling.
"""

class PASError(Exception):
    """Base exception for all PAS errors."""
    
    def __init__(self, message: str, code: str = "PAS_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)
    
    def to_dict(self) -> dict:
        """Convert to response dict for MCP tool returns."""
        return {
            "success": False,
            "error": self.message,
            "code": self.code
        }


# =============================================================================
# Session Errors
# =============================================================================

class SessionNotFoundError(PASError):
    """Raised when a session ID doesn't exist."""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session {session_id} not found",
            "SESSION_NOT_FOUND"
        )
        self.session_id = session_id


class SessionNotActiveError(PASError):
    """Raised when trying to modify a non-active session."""
    
    def __init__(self, session_id: str, state: str):
        super().__init__(
            f"Session is {state}, not active",
            "SESSION_NOT_ACTIVE"
        )
        self.session_id = session_id
        self.state = state


class SessionAlreadyExistsError(PASError):
    """Raised when trying to create a duplicate session."""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Session {session_id} already exists",
            "SESSION_EXISTS"
        )
        self.session_id = session_id


# =============================================================================
# Node Errors
# =============================================================================

class NodeNotFoundError(PASError):
    """Raised when a thought node doesn't exist."""
    
    def __init__(self, node_id: str):
        super().__init__(
            f"Node {node_id} not found",
            "NODE_NOT_FOUND"
        )
        self.node_id = node_id


class NodeNotCritiquedError(PASError):
    """Raised when trying to finalize without critiques."""
    
    def __init__(self, node_id: str):
        super().__init__(
            f"Node {node_id} was not critiqued",
            "NODE_NOT_CRITIQUED"
        )
        self.node_id = node_id


# =============================================================================
# Database Errors
# =============================================================================

class DatabaseError(PASError):
    """Raised when a database operation fails."""
    
    def __init__(self, operation: str, detail: str):
        super().__init__(
            f"DB operation '{operation}' failed: {detail}",
            "DB_ERROR"
        )
        self.operation = operation
        self.detail = detail


class ConnectionError(PASError):
    """Raised when database connection fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            f"Database connection failed: {detail}",
            "CONNECTION_ERROR"
        )
        self.detail = detail


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(PASError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            f"Validation error for '{field}': {message}",
            "VALIDATION_ERROR"
        )
        self.field = field


class InvalidConfidenceError(ValidationError):
    """Raised when confidence is outside 0.0-1.0 range."""
    
    def __init__(self, field: str, value: float):
        super().__init__(
            field,
            f"Confidence must be 0.0-1.0, got {value}"
        )
        self.value = value


class InvalidOutcomeError(ValidationError):
    """Raised when outcome is not success/partial/failure."""
    
    def __init__(self, outcome: str):
        super().__init__(
            "outcome",
            f"Must be 'success', 'partial', or 'failure', got '{outcome}'"
        )
        self.outcome = outcome


# =============================================================================
# Quality Gate Errors
# =============================================================================

class QualityGateError(PASError):
    """Raised when quality gate requirements not met."""
    
    def __init__(self, score: float, threshold: float, gap: float, gap_threshold: float):
        super().__init__(
            f"Quality gate failed: score={score:.3f} (need {threshold}), gap={gap:.3f} (need {gap_threshold})",
            "QUALITY_GATE_FAILED"
        )
        self.score = score
        self.threshold = threshold
        self.gap = gap
        self.gap_threshold = gap_threshold


class SequentialAnalysisMissingError(PASError):
    """Raised when finalize called without sequential analysis."""
    
    def __init__(self, session_id: str):
        super().__init__(
            f"Sequential analysis required before finalize. Call prepare_sequential_analysis first.",
            "SEQUENTIAL_ANALYSIS_MISSING"
        )
        self.session_id = session_id


# =============================================================================
# Codebase Errors  
# =============================================================================

class ProjectNotFoundError(PASError):
    """Raised when project_id doesn't exist."""
    
    def __init__(self, project_id: str):
        super().__init__(
            f"Project {project_id} not found. Run sync_project first.",
            "PROJECT_NOT_FOUND"
        )
        self.project_id = project_id


class SymbolNotFoundError(PASError):
    """Raised when symbol lookup fails."""
    
    def __init__(self, symbol_name: str, project_id: str):
        super().__init__(
            f"Symbol '{symbol_name}' not found in project {project_id}",
            "SYMBOL_NOT_FOUND"
        )
        self.symbol_name = symbol_name
        self.project_id = project_id


# =============================================================================
# Sampling Errors (MCP Host Communication)
# =============================================================================

class SamplingError(PASError):
    """Raised when MCP sampling request fails."""
    
    def __init__(self, detail: str):
        super().__init__(
            f"Sampling request failed: {detail}",
            "SAMPLING_ERROR"
        )
        self.detail = detail


class SamplingDeniedError(SamplingError):
    """Raised when user denies the sampling request."""
    
    def __init__(self):
        super().__init__("User denied sampling request")
        self.code = "SAMPLING_DENIED"
