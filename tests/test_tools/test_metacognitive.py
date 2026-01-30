"""
Layer 1: Metacognitive Flow Tool Tests

Tests for metacognitive prompting tools (v40):
- get_metacognitive_status
- advance_metacognitive_stage
- get_self_awareness
"""
import pytest


class TestMetacognitive:
    """Tests for 5-stage metacognitive prompting."""
    
    @pytest.mark.asyncio
    async def test_get_metacognitive_status(self, db_connection):
        """Verify metacognitive status retrieval."""
        from server import start_reasoning_session, get_metacognitive_status
        
        session = await start_reasoning_session("Test metacognitive flow")
        result = await get_metacognitive_status(session["session_id"])
        
        assert result["success"] is True
        assert "current_stage" in result
        assert result["current_stage"] >= 0
    
    @pytest.mark.asyncio
    async def test_advance_metacognitive_stage(self, db_connection):
        """Verify stage advancement."""
        from server import start_reasoning_session, advance_metacognitive_stage
        
        session = await start_reasoning_session("Test stage advancement")
        result = await advance_metacognitive_stage(session["session_id"])
        
        assert result["success"] is True
        assert "current_stage" in result  # API returns current_stage
        assert "stage_prompt" in result
    
    @pytest.mark.asyncio
    async def test_advance_to_specific_stage(self, db_connection):
        """Verify advancing through stages."""
        from server import start_reasoning_session, advance_metacognitive_stage
        
        session = await start_reasoning_session("Test stage progression")
        
        # Must advance incrementally - can't skip stages
        for target in range(1, 4):
            result = await advance_metacognitive_stage(
                session["session_id"],
                target_stage=target
            )
            assert result["success"] is True
        
        assert result["current_stage"] == 3


class TestSelfAwareness:
    """Tests for PAS self-awareness (v40 Phase 4)."""
    
    @pytest.mark.asyncio
    async def test_get_self_awareness_returns_schema(self, db_connection):
        """Verify self-awareness returns schema info."""
        from server import get_self_awareness
        
        result = await get_self_awareness()
        
        assert result["success"] is True
        assert "schema" in result
        assert "table_count" in result["schema"]
        assert result["schema"]["table_count"] > 0
    
    @pytest.mark.asyncio
    async def test_get_self_awareness_returns_tools(self, db_connection):
        """Verify self-awareness returns tool list."""
        from server import get_self_awareness
        
        result = await get_self_awareness()
        
        assert result["success"] is True
        assert "tools" in result
        # Tools is a dict: {"count": N, "categories": [...], "tools": [...]}
        tools_data = result["tools"]
        assert tools_data["count"] > 30  # Should have 38+ tools
        
        # Verify tool structure
        tool_list = tools_data.get("tools", [])
        assert len(tool_list) > 0
        tool = tool_list[0]
        assert "name" in tool
        assert "description" in tool
    
    @pytest.mark.asyncio
    async def test_get_self_awareness_returns_architecture(self, db_connection):
        """Verify self-awareness returns architecture info."""
        from server import get_self_awareness
        
        result = await get_self_awareness()
        
        assert result["success"] is True
        assert "architecture" in result
        assert "reasoning_flow" in result["architecture"]
        assert "learning_flow" in result["architecture"]


class TestMetacognitiveStages:
    """Tests for individual metacognitive stages."""
    
    STAGE_NAMES = [
        "Understanding",
        "Preliminary Judgment", 
        "Critical Evaluation",
        "Final Decision",
        "Confidence Expression"
    ]
    
    @pytest.mark.asyncio
    async def test_all_stages_have_prompts(self, db_connection):
        """Verify each of 5 stages has a prompt."""
        from server import start_reasoning_session, advance_metacognitive_stage
        
        session = await start_reasoning_session("Test all stages")
        
        for target_stage in range(1, 6):
            result = await advance_metacognitive_stage(
                session["session_id"],
                target_stage=target_stage
            )
            
            assert result["success"] is True
            assert "stage_prompt" in result
            assert len(result["stage_prompt"]) > 10  # Non-trivial prompt
