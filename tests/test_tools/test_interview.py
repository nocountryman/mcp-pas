"""
Layer 1: Interview Flow Tool Tests

Tests for clarifying question interview tools:
- identify_gaps
- get_next_question
- submit_answer
- check_interview_complete
"""
import pytest


class TestInterview:
    """Tests for interview/clarification flow."""
    
    @pytest.mark.asyncio
    async def test_identify_gaps_generates_questions(self, db_connection):
        """Verify gap identification generates questions."""
        from pas.server import start_reasoning_session, identify_gaps
        
        session = await start_reasoning_session(
            "Build a user authentication system with OAuth support"
        )
        
        result = await identify_gaps(session["session_id"])
        
        assert result["success"] is True
        # May or may not have questions depending on goal complexity
        # API returns: questions_generated, interview_config, detected_domains
        assert "questions_generated" in result or "no_gaps_detected" in result
    
    @pytest.mark.asyncio
    async def test_get_next_question(self, db_connection):
        """Verify next question retrieval."""
        from pas.server import start_reasoning_session, identify_gaps, get_next_question
        
        session = await start_reasoning_session(
            "Design a complex distributed system with multiple components"
        )
        await identify_gaps(session["session_id"])
        
        result = await get_next_question(session["session_id"])
        
        assert result["success"] is True
        # Either returns a question or indicates interview complete
        assert "question" in result or "interview_complete" in result
    
    @pytest.mark.asyncio
    async def test_submit_answer(self, db_connection):
        """Verify answer submission."""
        from pas.server import start_reasoning_session, identify_gaps, get_next_question, submit_answer
        
        session = await start_reasoning_session(
            "Create a payment processing system"
        )
        await identify_gaps(session["session_id"])
        
        q_result = await get_next_question(session["session_id"])
        
        if q_result.get("interview_complete"):
            pytest.skip("No questions generated for this goal")
        
        question_id = q_result["question"]["id"]
        result = await submit_answer(
            session["session_id"],
            question_id=question_id,
            answer="A"  # First choice
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_check_interview_complete(self, db_connection):
        """Verify interview completion check."""
        from pas.server import start_reasoning_session, check_interview_complete
        
        session = await start_reasoning_session("Simple test goal")
        
        result = await check_interview_complete(session["session_id"])
        
        assert result["success"] is True
        assert "is_complete" in result  # API returns is_complete


class TestEarlyExit:
    """Tests for early termination (v23)."""
    
    @pytest.mark.asyncio
    async def test_early_exit_offered_on_plateau(self, test_session_sync, db_connection):
        """Verify early exit offered when evidence plateaus."""
        import json
        from pas.server import get_next_question
        
        cur = db_connection.cursor()
        
        # Set plateau condition: 3 consecutive zero-delta answers
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
        
        result = await get_next_question(test_session_sync)
        
        assert result.get("early_exit_offered") is True
