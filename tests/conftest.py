"""
PAS Test Configuration - Shared Fixtures

Provides database connections, test sessions, and cleanup for v22/v23 tests.
"""
import pytest
import asyncio
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
