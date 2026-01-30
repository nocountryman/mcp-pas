"""
PAS Test Configuration - Shared Fixtures + Layer 4 Failure Logging

Provides database connections, test sessions, and cleanup for tests.
Layer 4 (v42b): Logs test failures back into PAS for self-learning.
"""
import pytest
import asyncio
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment flag to disable PAS logging (for faster debugging)
PAS_LOG_FAILURES = os.getenv("PAS_LOG_FAILURES", "true").lower() != "false"

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def db_connection():
    """Provide test database connection with rollback."""
    from server import get_db_connection, safe_close_connection
    conn = get_db_connection()
    yield conn
    conn.rollback()  # Undo test changes
    safe_close_connection(conn)


@pytest.fixture
def test_user_id():
    """Consistent test user ID for trait tests."""
    return "test_v24_" + str(uuid.uuid4())[:8]


@pytest.fixture
def test_session_sync(db_connection):
    """Create a fresh PAS session synchronously using direct DB."""
    import json
    cur = db_connection.cursor()
    session_id = str(uuid.uuid4())
    goal = f"v24 test session {session_id[:8]}"
    
    cur.execute("""
        INSERT INTO reasoning_sessions (id, goal, context, state)
        VALUES (%s, %s, %s, 'active')
        RETURNING id
    """, (session_id, goal, json.dumps({"user_id": "test_v24_user"})))
    db_connection.commit()
    
    return session_id


@pytest.fixture
def db_cursor(db_connection):
    """Provide a cursor from the test connection."""
    return db_connection.cursor()


# =============================================================================
# LAYER 4: PAS Failure Logging (v42b Self-Learning Integration)
# =============================================================================

def categorize_failure(longrepr) -> str:
    """Categorize failure type for semantic surfacing."""
    text = str(longrepr).lower()
    if "assertionerror" in text:
        return "LOGIC"
    elif "psycopg" in text or "sql" in text or "database" in text:
        return "DB"
    elif "keyerror" in text or "attributeerror" in text:
        return "CODE"
    elif "timeout" in text or "connection" in text:
        return "INFRA"
    else:
        return "WORKFLOW"


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Layer 4: Log test failures to PAS for self-learning.
    
    Creates a PAS session for each failure with:
    - Categorized failure type (LOGIC, DB, CODE, WORKFLOW, INFRA)
    - Semantic failure_reason for future surfacing
    - Test file scope for scope-based matching
    
    Disable with: PAS_LOG_FAILURES=false pytest ...
    """
    outcome = yield
    report = outcome.get_result()
    
    # Only log actual test failures (not setup/teardown)
    if report.when != "call" or not report.failed:
        return
    
    # Skip if disabled for debugging
    if not PAS_LOG_FAILURES:
        return
    
    try:
        # Import PAS tools (lazy import to avoid circular deps)
        from server import start_reasoning_session, store_expansion, record_outcome
        
        # Get failure details
        failure_category = categorize_failure(report.longrepr)
        test_file = str(item.fspath).replace(str(item.config.rootdir), "").lstrip("/")
        failure_text = str(report.longrepr)[:500]
        
        # Create PAS session for the failure
        loop = asyncio.get_event_loop()
        
        session_result = loop.run_until_complete(
            start_reasoning_session(f"Test Bug: {item.name} in {test_file}")
        )
        
        if not session_result.get("success"):
            return  # Don't fail test run if PAS logging fails
        
        session_id = session_result["session_id"]
        
        # Store failure as hypothesis with scope
        loop.run_until_complete(
            store_expansion(
                session_id,
                parent_node_id=None,
                h1_text=failure_text,
                h1_confidence=0.95,
                h1_scope=f"[{failure_category}] {test_file}",
                skip_preflight=True  # Don't require codebase research for test failures
            )
        )
        
        # Record failure with semantic reason
        loop.run_until_complete(
            record_outcome(
                session_id,
                outcome="failure",
                failure_reason=f"{failure_category}: {item.name} - {failure_text[:200]}"
            )
        )
        
        print(f"\n[PAS] Logged failure: {failure_category} in {test_file}")
        
    except Exception as e:
        # Never fail the test run due to PAS logging issues
        print(f"\n[PAS] Warning: Could not log failure to PAS: {e}")
