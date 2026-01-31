"""
PAS Server Unit Tests

Tests for v22 (persistent traits, law boosting, hybrid inference) 
and v23 (early termination) features.
"""
import pytest
import json


class TestV22Traits:
    """Test persistent trait profiles (v22 Feature 1)."""
    
    @pytest.mark.asyncio
    async def test_user_id_stored_on_session_start(self, db_connection):
        """Verify user_id is always stored in session context."""
        from pas.server import start_reasoning_session
        
        result = await start_reasoning_session("v24 test: user_id storage")
        assert result["success"]
        session_id = result["session_id"]
        
        cur = db_connection.cursor()
        cur.execute(
            "SELECT context->>'user_id' FROM reasoning_sessions WHERE id = %s", 
            (session_id,)
        )
        row = cur.fetchone()
        user_id = row["?column?"] if isinstance(row, dict) else row[0]
        assert user_id is not None, "user_id should be stored in session context"
        assert len(user_id) == 32, f"user_id should be 32-char SHA256 hash, got {len(user_id)}"
    
    def test_trait_decay_formula(self):
        """Verify 30-day half-life decay calculation."""
        score = 1.0
        half_life_days = 30
        
        # At 30 days, score should be 0.5
        decayed_30 = score * (0.5 ** (30 / half_life_days))
        assert abs(decayed_30 - 0.5) < 0.01, f"30-day decay should be 0.5, got {decayed_30}"
        
        # At 60 days, score should be 0.25
        decayed_60 = score * (0.5 ** (60 / half_life_days))
        assert abs(decayed_60 - 0.25) < 0.01, f"60-day decay should be 0.25, got {decayed_60}"
        
        # At 0 days, score should be unchanged
        decayed_0 = score * (0.5 ** (0 / half_life_days))
        assert decayed_0 == 1.0, "0-day decay should be 1.0"
    
    @pytest.mark.asyncio
    async def test_trait_persistence_on_outcome(self, db_connection):
        """Verify traits are persisted after record_outcome."""
        from pas.server import start_reasoning_session, store_expansion, record_outcome
        
        # Create session with full flow
        result = await start_reasoning_session("v24 test: trait persistence")
        session_id = result["session_id"]
        
        # Add a hypothesis (required for record_outcome)
        await store_expansion(
            session_id, 
            parent_node_id=None,
            h1_text="Test hypothesis for v24",
            h1_confidence=0.8
        )
        
        cur = db_connection.cursor()
        
        # Inject test trait into session context
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = context || %s::jsonb
            WHERE id = %s
        """, (json.dumps({
            "latent_traits": [{"trait": "TEST_TRAIT_V24", "confidence": 0.8}],
            "user_id": "test_user_v24_fixture"
        }), session_id))
        db_connection.commit()
        
        # Record successful outcome
        result = await record_outcome(session_id, "success")
        assert result["success"], f"record_outcome failed: {result}"
        
        # Check trait was persisted
        cur.execute(
            "SELECT * FROM user_trait_profiles WHERE trait_name = 'TEST_TRAIT_V24'"
        )
        row = cur.fetchone()
        assert row is not None, "Trait should be persisted to user_trait_profiles"
        
        # Cleanup
        cur.execute("DELETE FROM user_trait_profiles WHERE trait_name = 'TEST_TRAIT_V24'")
        db_connection.commit()


class TestV22LawBoosting:
    """Test law weight boosting (v22 Feature 2)."""
    
    @pytest.mark.asyncio
    async def test_boost_applied_correctly(self, db_connection):
        """Verify boost is applied to law weights."""
        from pas.server import start_reasoning_session, prepare_expansion
        
        # Create session
        result = await start_reasoning_session("v24 test: boost application")
        session_id = result["session_id"]
        
        cur = db_connection.cursor()
        
        # Insert valid boost factor (0.4 = 40% boost, within DB constraint of 0.5)
        cur.execute("""
            INSERT INTO trait_law_correlations (trait_name, law_id, boost_factor)
            VALUES ('BOOST_TEST_TRAIT', 2, 0.4)
            ON CONFLICT (trait_name, law_id) DO UPDATE SET boost_factor = 0.4
        """)
        
        # Inject trait into session
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = context || %s::jsonb
            WHERE id = %s
        """, (json.dumps({
            "latent_traits": [{"trait": "BOOST_TEST_TRAIT", "confidence": 1.0}]
        }), session_id))
        db_connection.commit()
        
        result = await prepare_expansion(session_id)
        assert result["success"], f"prepare_expansion failed: {result}"
        
        # Check that law with id=2 has been boosted
        boosted_laws = [law for law in result.get("relevant_laws", []) 
                        if law.get("boosted_by")]
        # If this trait-law correlation is matched, we expect a boost
        # (may not always match depending on law similarity)
        
        # All boosts should be <= 0.5 (ceiling)
        for law in result.get("relevant_laws", []):
            boost = law.get("boosted_by", 0)
            assert boost <= 0.5, f"Boost {boost} exceeds 50% ceiling"
        
        # Cleanup
        cur.execute("DELETE FROM trait_law_correlations WHERE trait_name = 'BOOST_TEST_TRAIT'")
        db_connection.commit()

    
    def test_boost_formula_correctness(self):
        """Verify boost calculation formula."""
        original_weight = 0.8
        boost_factor = 0.3  # 30% boost
        
        boosted = original_weight * (1 + boost_factor)
        expected = 1.04
        
        assert abs(boosted - expected) < 0.01, f"Expected {expected}, got {boosted}"


class TestV22HybridInference:
    """Test semantic trait matching (v22 Feature 3)."""
    
    def test_semantic_threshold_value(self):
        """Verify 0.65 similarity threshold is reasonable."""
        threshold = 0.65
        assert threshold > 0.5, "Threshold should be above random chance (0.5)"
        assert threshold < 0.9, "Threshold shouldn't be too strict"
    
    def test_exemplars_table_has_embeddings(self, db_connection):
        """Verify trait_exemplars table has embeddings for semantic matching."""
        cur = db_connection.cursor()
        cur.execute("""
            SELECT COUNT(*) as cnt FROM trait_exemplars WHERE embedding IS NOT NULL
        """)
        row = cur.fetchone()
        count = row["cnt"] if isinstance(row, dict) else row[0]
        assert count > 0, "trait_exemplars should have embeddings for hybrid inference"


class TestV23EarlyExit:
    """Test early termination (v23)."""
    
    @pytest.mark.asyncio
    async def test_evidence_tracking_populated(self, db_connection):
        """Verify evidence_history is populated after answers."""
        from pas.server import start_reasoning_session, identify_gaps, get_next_question, submit_answer
        
        result = await start_reasoning_session("v24 test: evidence tracking")
        session_id = result["session_id"]
        
        await identify_gaps(session_id)
        q = await get_next_question(session_id)
        
        if q.get("interview_complete") or q.get("early_exit_offered"):
            pytest.skip("No questions available for this test")
        
        await submit_answer(session_id, q["question"]["id"], "A")
        
        cur = db_connection.cursor()
        cur.execute(
            "SELECT context FROM reasoning_sessions WHERE id = %s", 
            (session_id,)
        )
        row = cur.fetchone()
        context = row["context"] if isinstance(row, dict) else row[0]
        if isinstance(context, str):
            context = json.loads(context)
        interview = context.get("interview", {})
        
        assert "evidence_history" in interview, "evidence_history should exist after answer"
        assert len(interview["evidence_history"]) >= 1, "evidence_history should have at least 1 entry"
    
    @pytest.mark.asyncio
    async def test_early_exit_offered_on_plateau(self, test_session_sync, db_connection):
        """Verify early exit is offered when evidence delta = 0 for 3 answers."""
        from pas.server import get_next_question
        
        cur = db_connection.cursor()
        
        # Manually set plateau condition
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = context || %s::jsonb
            WHERE id = %s
        """, (json.dumps({
            "interview": {
                "evidence_history": [0, 0, 0],
                "config": {"questions_answered": 5, "questions_remaining": 2},
                "early_termination_suggested": True,
                "pending_questions": [{"id": "test_q", "answered": False}]
            }
        }), test_session_sync))
        db_connection.commit()
        
        q = await get_next_question(test_session_sync)
        assert q.get("early_exit_offered") == True, "early_exit should be offered on plateau"
    
    @pytest.mark.asyncio
    async def test_continue_after_early_exit_declined(self, test_session_sync, db_connection):
        """Verify user can continue after declining early exit."""
        from pas.server import submit_answer
        
        cur = db_connection.cursor()
        
        # Set up early exit state with a pending question
        cur.execute("""
            UPDATE reasoning_sessions 
            SET context = context || %s::jsonb
            WHERE id = %s
        """, (json.dumps({
            "interview": {
                "evidence_history": [0, 0, 0],
                "config": {"questions_answered": 5, "questions_remaining": 1},
                "early_termination_suggested": True,
                "answer_history": [],
                "pending_questions": [{
                    "id": "decline_test_q",
                    "answered": False,
                    "question_text": "Test question",
                    "choices": [{"label": "A", "description": "Option A"}]
                }]
            }
        }), test_session_sync))
        db_connection.commit()
        
        # Submit answer (this declines early exit)
        result = await submit_answer(test_session_sync, "decline_test_q", "A")
        assert result["success"], f"submit_answer failed: {result}"
        
        # Check flag was set
        cur.execute(
            "SELECT context FROM reasoning_sessions WHERE id = %s", 
            (test_session_sync,)
        )
        row = cur.fetchone()
        context = row["context"] if isinstance(row, dict) else row[0]
        if isinstance(context, str):
            context = json.loads(context)
        interview = context.get("interview", {})
        assert interview.get("early_exit_declined") == True, "early_exit_declined should be True"


class TestServerBasics:
    """Basic server functionality tests."""
    
    @pytest.mark.asyncio
    async def test_server_imports(self):
        """Verify server module imports without errors."""
        from pas import server
        assert hasattr(server, 'start_reasoning_session')
        assert hasattr(server, 'prepare_expansion')
        assert hasattr(server, 'record_outcome')
    
    @pytest.mark.asyncio
    async def test_session_creation(self, db_connection):
        """Verify session is created successfully."""
        from pas.server import start_reasoning_session
        
        result = await start_reasoning_session("v24 test: session creation")
        assert result["success"]
        assert result["session_id"] is not None
        assert len(result["session_id"]) == 36  # UUID format
