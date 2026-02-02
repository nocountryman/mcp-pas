"""
Tests for Phase 7d: Active Law Application.

Tests the build_law_application_block helper function that injects
law definitions and self-apply prompts into LLM prompts.
"""
import pytest


def test_build_law_application_block_standard():
    """Test that the helper produces formatted output with law name and definition."""
    from pas.helpers.critique import build_law_application_block
    
    law = {
        "law_name": "Conway's Law",
        "definition": "Organizations which design systems are constrained to produce designs which are copies of the communication structures."
    }
    
    result = build_law_application_block(law)
    
    # Should contain the law name
    assert "**MATCHED LAW**: Conway's Law" in result
    
    # Should contain the definition
    assert "**DEFINITION**: Organizations which design systems" in result
    
    # Should contain self-apply instructions
    assert "TASK: Before proceeding" in result
    assert "What markers or patterns should you look for?" in result
    assert "How should this influence your analysis?" in result


def test_build_law_application_block_missing_fields():
    """Test graceful handling of missing fields."""
    from pas.helpers.critique import build_law_application_block
    
    # Law with missing definition
    law = {"law_name": "Test Law"}
    result = build_law_application_block(law)
    
    assert "**MATCHED LAW**: Test Law" in result
    assert "**DEFINITION**: " in result  # Empty but still formatted
    
    # Law with missing name
    law2 = {"definition": "Some definition"}
    result2 = build_law_application_block(law2)
    
    assert "**MATCHED LAW**: Unknown" in result2
    
    # Empty law
    law3 = {}
    result3 = build_law_application_block(law3)
    
    assert "**MATCHED LAW**: Unknown" in result3
    assert "**DEFINITION**: " in result3


def test_build_critique_prompt_uses_law_blocks():
    """Test that build_critique_prompt uses full law blocks, not just names."""
    from pas.helpers.critique import build_critique_prompt
    
    laws = [
        {
            "law_name": "Gall's Law",
            "definition": "A complex system that works is invariably found to have evolved from a simple system that worked."
        },
        {
            "law_name": "Brooks' Law",
            "definition": "Adding manpower to a late software project makes it later."
        }
    ]
    
    prompt, system, expected_format = build_critique_prompt(
        node_content="Test hypothesis",
        session_goal="Test goal",
        laws_text=laws,
        critique_mode="standard"
    )
    
    # Should NOT contain the old format
    assert "Consider these scientific laws:" not in prompt
    
    # Should contain new section header
    assert "## Scientific Laws to Apply" in prompt
    
    # Should contain full law blocks
    assert "**MATCHED LAW**: Gall's Law" in prompt
    assert "**DEFINITION**: A complex system that works" in prompt
    assert "**MATCHED LAW**: Brooks' Law" in prompt
    
    # Should contain self-apply instructions
    assert "TASK: Before proceeding" in prompt


def test_build_critique_prompt_no_laws():
    """Test the prompt when no laws are matched."""
    from pas.helpers.critique import build_critique_prompt
    
    prompt, _, _ = build_critique_prompt(
        node_content="Test hypothesis",
        session_goal="Test goal",
        laws_text=[],
        critique_mode="standard"
    )
    
    assert "No laws matched" in prompt


def test_build_critique_prompt_negative_space_unchanged():
    """Test that negative_space mode is unchanged."""
    from pas.helpers.critique import build_critique_prompt
    
    prompt, system, _ = build_critique_prompt(
        node_content="Test hypothesis",
        session_goal="Test goal",
        laws_text=[{"law_name": "Test Law"}],
        critique_mode="negative_space"
    )
    
    # Should NOT use law blocks in negative_space mode
    assert "## Scientific Laws to Apply" not in prompt
    assert "Analyze what this hypothesis does NOT address" in prompt
