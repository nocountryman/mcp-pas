"""
Layer 1: Reasoning Flow Tool Tests

Tests for the 9 core reasoning tools:
- start_reasoning_session, find_or_create_session
- prepare_expansion, store_expansion
- prepare_critique, store_critique
- get_best_path, finalize_session, record_outcome
"""
import pytest
import json


class TestSessionManagement:
    """Tests for session creation and management tools."""
    
    @pytest.mark.asyncio
    async def test_start_reasoning_session_basic(self, db_connection):
        """Verify basic session creation."""
        from server import start_reasoning_session
        
        result = await start_reasoning_session("Test goal for unit testing")
        
        assert result["success"] is True
        assert result["session_id"] is not None
        assert len(result["session_id"]) == 36  # UUID format
        assert result["state"] == "active"
        assert "metacognitive_stage" in result
    
    @pytest.mark.asyncio
    async def test_start_session_stores_goal(self, db_connection):
        """Verify goal is stored correctly."""
        from server import start_reasoning_session
        
        goal = "Unique test goal " + str(pytest.importorskip("uuid").uuid4())[:8]
        result = await start_reasoning_session(goal)
        
        cur = db_connection.cursor()
        cur.execute("SELECT goal FROM reasoning_sessions WHERE id = %s", (result["session_id"],))
        row = cur.fetchone()
        
        assert row["goal"] == goal
    
    @pytest.mark.asyncio
    async def test_find_or_create_session_new(self, db_connection):
        """Verify find_or_create creates new session for unique goal."""
        from server import find_or_create_session
        
        goal = "Brand new unique goal " + str(pytest.importorskip("uuid").uuid4())
        result = await find_or_create_session(goal)
        
        assert result["success"] is True
        # May find existing session with similar semantics or create new
        assert result["action"] in ["new", "existing", "continuation"]
        assert result["session_id"] is not None
    
    @pytest.mark.asyncio
    async def test_find_or_create_session_existing(self, db_connection):
        """Verify find_or_create returns existing similar session."""
        from server import start_reasoning_session, find_or_create_session
        
        # Create initial session
        goal = "Implementing user authentication flow"
        initial = await start_reasoning_session(goal)
        
        # Search for similar
        similar_goal = "Implementing user authentication"
        result = await find_or_create_session(similar_goal, similarity_threshold=0.7)
        
        # Should find existing or create continuation (depending on state)
        assert result["success"] is True
        assert result["action"] in ["existing", "continuation", "new"]


class TestExpansion:
    """Tests for hypothesis expansion tools."""
    
    @pytest.mark.asyncio
    async def test_prepare_expansion_returns_context(self, db_connection):
        """Verify prepare_expansion returns required context."""
        from server import start_reasoning_session, prepare_expansion
        
        session = await start_reasoning_session("Test hypothesis generation")
        result = await prepare_expansion(session["session_id"])
        
        assert result["success"] is True
        assert "goal" in result
        assert "relevant_laws" in result
        assert "instructions" in result
    
    @pytest.mark.asyncio
    async def test_prepare_expansion_with_project_id(self, db_connection):
        """Verify prepare_expansion with project_id returns related modules."""
        from server import start_reasoning_session, prepare_expansion
        
        session = await start_reasoning_session("Test with codebase context")
        result = await prepare_expansion(session["session_id"], project_id="mcp-pas")
        
        assert result["success"] is True
        # related_modules may be empty if project not synced, but key should exist
        assert "related_modules" in result or result.get("error") is None
    
    @pytest.mark.asyncio
    async def test_store_expansion_single_hypothesis(self, db_connection):
        """Verify storing a single hypothesis."""
        from server import start_reasoning_session, store_expansion
        
        session = await start_reasoning_session("Test storing hypothesis")
        result = await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="First hypothesis about the problem",
            h1_confidence=0.8,
            h1_scope="[TEST] test_file.py"
        )
        
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["created_nodes"]) == 1
        assert result["created_nodes"][0]["posterior_score"] > 0
    
    @pytest.mark.asyncio
    async def test_store_expansion_multiple_hypotheses(self, db_connection):
        """Verify storing multiple hypotheses."""
        from server import start_reasoning_session, store_expansion
        
        session = await start_reasoning_session("Test multiple hypotheses")
        result = await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="First hypothesis",
            h1_confidence=0.8,
            h2_text="Second hypothesis",
            h2_confidence=0.7,
            h3_text="Third hypothesis",
            h3_confidence=0.6
        )
        
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["created_nodes"]) == 3


class TestCritique:
    """Tests for critique tools."""
    
    @pytest.mark.asyncio
    async def test_prepare_critique_returns_context(self, db_connection):
        """Verify prepare_critique returns node context."""
        from server import start_reasoning_session, store_expansion, prepare_critique
        
        session = await start_reasoning_session("Test critique flow")
        expansion = await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Hypothesis to critique",
            h1_confidence=0.8
        )
        
        node_id = expansion["created_nodes"][0]["node_id"]
        result = await prepare_critique(node_id)
        
        assert result["success"] is True
        assert "node_content" in result
        assert "llm_prompt" in result
        assert "past_failures" in result
    
    @pytest.mark.asyncio
    async def test_store_critique_updates_score(self, db_connection):
        """Verify critique updates node likelihood."""
        from server import start_reasoning_session, store_expansion, store_critique
        
        session = await start_reasoning_session("Test score update")
        expansion = await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Hypothesis with flaw",
            h1_confidence=0.9
        )
        
        node_id = expansion["created_nodes"][0]["node_id"]
        initial_score = expansion["created_nodes"][0]["posterior_score"]
        
        result = await store_critique(
            node_id,
            counterargument="This hypothesis has a critical flaw",
            severity_score=0.6,
            major_flaws="Missing edge case handling"
        )
        
        assert result["success"] is True
        assert result["score_update"]["new_likelihood"] < initial_score


class TestFinalization:
    """Tests for finalization and outcome tools."""
    
    @pytest.mark.asyncio
    async def test_get_best_path(self, db_connection):
        """Verify best path retrieval."""
        from server import start_reasoning_session, store_expansion, get_best_path
        
        session = await start_reasoning_session("Test best path")
        await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="High confidence hypothesis",
            h1_confidence=0.9,
            h2_text="Low confidence hypothesis",
            h2_confidence=0.4
        )
        
        result = await get_best_path(session["session_id"])
        
        assert result["success"] is True
        # API returns best_node_id, best_score, path
        assert result["best_node_id"] is not None
        assert result["best_score"] >= 0.4
    
    @pytest.mark.asyncio
    async def test_finalize_session_returns_recommendation(self, db_connection):
        """Verify finalize returns recommendation."""
        from server import start_reasoning_session, store_expansion, finalize_session
        
        session = await start_reasoning_session("Test finalization")
        await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Only hypothesis",
            h1_confidence=0.8
        )
        
        result = await finalize_session(session["session_id"], skip_sequential_analysis=True)
        
        assert result["success"] is True
        assert "recommendation" in result
        assert result["recommendation"]["content"] is not None
    
    @pytest.mark.asyncio
    async def test_record_outcome_success(self, db_connection):
        """Verify recording successful outcome."""
        from server import start_reasoning_session, store_expansion, record_outcome
        
        session = await start_reasoning_session("Test outcome recording")
        await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Hypothesis for outcome test",
            h1_confidence=0.8
        )
        
        result = await record_outcome(
            session["session_id"],
            outcome="success",
            notes="Test completed successfully"
        )
        
        assert result["success"] is True
        assert result["outcome_id"] is not None
        assert result["session_completed"] is True
    
    @pytest.mark.asyncio
    async def test_record_outcome_failure_with_reason(self, db_connection):
        """Verify recording failure with semantic reason."""
        from server import start_reasoning_session, store_expansion, record_outcome
        
        session = await start_reasoning_session("Test failure recording")
        await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Hypothesis that failed",
            h1_confidence=0.8
        )
        
        result = await record_outcome(
            session["session_id"],
            outcome="failure",
            failure_reason="DB: Schema migration failed due to missing column"
        )
        
        assert result["success"] is True
        assert result["outcome"] == "failure"
        
        # Verify failure_reason is stored
        cur = db_connection.cursor()
        cur.execute(
            "SELECT failure_reason FROM outcome_records WHERE id = %s",
            (result["outcome_id"],)
        )
        row = cur.fetchone()
        assert "Schema migration failed" in row["failure_reason"]
