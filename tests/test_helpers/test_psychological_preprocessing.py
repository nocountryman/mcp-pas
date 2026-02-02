"""
Tests for Phase 7a: Psychological Pre-Processing.

Tests the auto-extraction of psychological signals from user prompts
at root expansion time.
"""
import pytest
import json


def test_perform_psychological_preprocessing_not_root():
    """Test that preprocessing is skipped for non-root expansions."""
    from pas.helpers.expansion import perform_psychological_preprocessing
    
    # Mock cursor - should not be called
    class MockCursor:
        def execute(self, *args):
            raise AssertionError("Should not execute for non-root expansion")
    
    extraction, prompt = perform_psychological_preprocessing(
        cur=MockCursor(),
        conn=None,
        session_id="test-session",
        is_root_expansion=False,
        get_embedding_fn=lambda x: [0.0] * 768
    )
    
    assert extraction is None
    assert prompt is None


def test_perform_psychological_preprocessing_cached():
    """Test that cached extraction is returned if available."""
    from pas.helpers.expansion import perform_psychological_preprocessing
    
    cached_data = {
        "hedging_detected": True,
        "hedging_markers": ["might", "maybe"],
        "inferred_priority": "should-have"
    }
    
    class MockCursor:
        def __init__(self):
            self.call_count = 0
            
        def execute(self, query, params=None):
            self.call_count += 1
            self.last_query = query
            
        def fetchone(self):
            # First call: session context with psychological_extraction
            if "context FROM reasoning_sessions" in self.last_query:
                return {"context": {"psychological_extraction": cached_data}}
            return None
    
    extraction, prompt = perform_psychological_preprocessing(
        cur=MockCursor(),
        conn=None,
        session_id="test-session",
        is_root_expansion=True,
        get_embedding_fn=lambda x: [0.0] * 768
    )
    
    assert extraction == cached_data
    assert prompt is None


def test_perform_psychological_preprocessing_generates_prompt():
    """Test that a prompt is generated when no cache exists."""
    from pas.helpers.expansion import perform_psychological_preprocessing
    
    class MockCursor:
        def __init__(self):
            self.call_count = 0
            
        def execute(self, query, params=None):
            self.call_count += 1
            self.last_query = query
            
        def fetchone(self):
            # Session context without psychological_extraction
            if "context FROM reasoning_sessions" in self.last_query:
                return {"context": {}}
            # Verbatim log exists
            if "conversation_log" in self.last_query:
                return {"raw_text": "I might want to add a feature"}
            return None
            
        def fetchall(self):
            # Return mock laws
            return [
                {"id": 1, "law_name": "Hedging Detection", "definition": "Test def 1"},
                {"id": 2, "law_name": "Speech Act Theory", "definition": "Test def 2"}
            ]
    
    extraction, prompt = perform_psychological_preprocessing(
        cur=MockCursor(),
        conn=None,
        session_id="test-session",
        is_root_expansion=True,
        get_embedding_fn=lambda x: [0.0] * 768
    )
    
    assert extraction is None
    assert prompt is not None
    assert "I might want to add a feature" in prompt
    assert "Hedging Detection" in prompt
    assert "hedging_detected" in prompt


def test_cache_psychological_extraction():
    """Test caching extraction data in session context."""
    from pas.helpers.expansion import cache_psychological_extraction
    
    class MockCursor:
        def __init__(self):
            self.executed_queries = []
            
        def execute(self, query, params=None):
            self.executed_queries.append((query, params))
    
    class MockConn:
        def __init__(self):
            self.committed = False
            
        def commit(self):
            self.committed = True
    
    cur = MockCursor()
    conn = MockConn()
    
    extraction_data = {
        "hedging_detected": True,
        "hedging_markers": ["might"],
        "inferred_priority": "should-have"
    }
    
    result = cache_psychological_extraction(cur, conn, "test-session", extraction_data)
    
    assert result is True
    assert conn.committed is True
    assert len(cur.executed_queries) == 1
    assert "psychological_extraction" in cur.executed_queries[0][0]
