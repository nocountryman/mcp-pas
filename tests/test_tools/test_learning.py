"""
Layer 1: Learning Flow Tool Tests

Tests for the self-learning tools:
- record_outcome (partial covered in test_reasoning)
- refresh_law_weights
- parse_terminal_output
- get_calibration_stats
"""
import pytest


class TestLearning:
    """Tests for PAS self-learning mechanisms."""
    
    @pytest.mark.asyncio
    async def test_parse_terminal_output_success(self, db_connection):
        """Verify success pattern detection."""
        from server import start_reasoning_session, parse_terminal_output
        
        session = await start_reasoning_session("Test RLVR parsing")
        
        result = await parse_terminal_output(
            session["session_id"],
            terminal_text="All 15 tests passed!\n=== 15 passed in 2.34s ===",
            auto_record=False
        )
        
        assert result["success"] is True
        assert result["signal"] == "success"
        assert result["confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_parse_terminal_output_failure(self, db_connection):
        """Verify failure pattern detection."""
        from server import start_reasoning_session, parse_terminal_output
        
        session = await start_reasoning_session("Test RLVR failure detection")
        
        result = await parse_terminal_output(
            session["session_id"],
            terminal_text="FAILED test_foo.py::test_bar - AssertionError\n=== 1 failed, 14 passed ===",
            auto_record=False
        )
        
        assert result["success"] is True
        assert result["signal"] == "failure"
    
    @pytest.mark.asyncio
    async def test_parse_terminal_output_auto_record(self, db_connection):
        """Verify auto_record creates outcome."""
        from server import start_reasoning_session, store_expansion, parse_terminal_output
        
        session = await start_reasoning_session("Test RLVR auto-record")
        await store_expansion(
            session["session_id"],
            parent_node_id=None,
            h1_text="Hypothesis for auto-record",
            h1_confidence=0.8
        )
        
        result = await parse_terminal_output(
            session["session_id"],
            terminal_text="Build successful!\nAll checks passed.",
            auto_record=True
        )
        
        assert result["success"] is True
        if result.get("auto_recorded"):
            # outcome_id is nested inside outcome_result
            outcome_result = result.get("outcome_result", {})
            assert outcome_result.get("outcome_id") is not None
    
    @pytest.mark.asyncio
    async def test_get_calibration_stats(self, db_connection):
        """Verify calibration statistics retrieval."""
        from server import get_calibration_stats
        
        result = await get_calibration_stats(min_samples=1)
        
        assert result["success"] is True
        assert "sample_count" in result
        assert "brier_score" in result or result.get("insufficient_samples") is True
    
    @pytest.mark.asyncio
    async def test_refresh_law_weights(self, db_connection):
        """Verify law weight refresh mechanism."""
        from server import refresh_law_weights
        
        result = await refresh_law_weights(min_samples=1, blend_factor=0.3)
        
        assert result["success"] is True
        assert "laws_updated" in result
        # May be 0 if not enough outcome data
        assert result["laws_updated"] >= 0


class TestCalibration:
    """Tests for Calibrated Self-Rewarding (CSR) mechanism."""
    
    def test_calibration_record_table_exists(self, db_connection):
        """Verify calibration_records table exists."""
        cur = db_connection.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'calibration_records'
            )
        """)
        row = cur.fetchone()
        exists = row["exists"] if isinstance(row, dict) else row[0]
        assert exists is True
