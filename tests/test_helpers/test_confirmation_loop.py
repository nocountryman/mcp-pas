"""
Tests for Phase 7b: Confirmation Loop Questions.

Tests the generate_confirmation_questions helper that creates
targeted questions based on psychological triggers.
"""
import pytest


def test_no_triggers_returns_empty():
    """Test that no triggers results in no questions."""
    from pas.helpers.interview import generate_confirmation_questions
    
    result = generate_confirmation_questions(
        session_context={},
        hypotheses=None
    )
    
    assert result == []


def test_hedging_trigger_generates_question():
    """Test that hedging detection triggers a confirmation question."""
    from pas.helpers.interview import generate_confirmation_questions
    
    context = {
        "psychological_extraction": {
            "hedging_detected": True,
            "hedging_markers": ["might", "maybe", "could"]
        }
    }
    
    result = generate_confirmation_questions(
        session_context=context,
        hypotheses=None
    )
    
    assert len(result) == 1
    assert result[0]["trigger"] == "hedging_detected"
    assert result[0]["source"] == "confirmation_loop"
    assert "might, maybe, could" in result[0]["question_text"]
    assert len(result[0]["choices"]) == 3


def test_low_confidence_trigger_generates_question():
    """Test that low confidence hypothesis triggers a confirmation question."""
    from pas.helpers.interview import generate_confirmation_questions
    
    hypotheses = [
        {"text": "Modify the database schema to add new column", "confidence": 0.5, "scope": "db.py"},
        {"text": "High confidence approach", "confidence": 0.9, "scope": "utils.py"}
    ]
    
    result = generate_confirmation_questions(
        session_context={},
        hypotheses=hypotheses
    )
    
    assert len(result) == 1
    assert result[0]["trigger"] == "low_confidence"
    assert "50%" in result[0]["question_text"]


def test_high_scope_trigger_generates_question():
    """Test that high scope complexity triggers a confirmation question."""
    from pas.helpers.interview import generate_confirmation_questions
    
    hypotheses = [
        {"text": "Major refactoring", "confidence": 0.8, "scope": "file1.py, file2.py, file3.py, file4.py, file5.py"}
    ]
    
    result = generate_confirmation_questions(
        session_context={},
        hypotheses=hypotheses
    )
    
    assert len(result) == 1
    assert result[0]["trigger"] == "high_scope"
    assert "5 files" in result[0]["question_text"]


def test_concern_areas_trigger_generates_question():
    """Test that concern areas trigger a confirmation question."""
    from pas.helpers.interview import generate_confirmation_questions
    
    context = {
        "psychological_extraction": {
            "concern_areas": ["security", "performance", "data integrity"]
        }
    }
    
    result = generate_confirmation_questions(
        session_context=context,
        hypotheses=None
    )
    
    assert len(result) == 1
    assert result[0]["trigger"] == "concern_areas"
    assert "security, performance, data integrity" in result[0]["question_text"]


def test_multiple_triggers_generate_multiple_questions():
    """Test that multiple triggers generate multiple questions."""
    from pas.helpers.interview import generate_confirmation_questions
    
    context = {
        "psychological_extraction": {
            "hedging_detected": True,
            "hedging_markers": ["might"],
            "concern_areas": ["security"]
        }
    }
    
    hypotheses = [
        {"text": "Low conf approach", "confidence": 0.6, "scope": "a.py, b.py, c.py, d.py"}
    ]
    
    result = generate_confirmation_questions(
        session_context=context,
        hypotheses=hypotheses
    )
    
    # Should have: hedging + low_confidence + high_scope + concerns = 4 questions
    assert len(result) == 4
    triggers = {q["trigger"] for q in result}
    assert triggers == {"hedging_detected", "low_confidence", "high_scope", "concern_areas"}


def test_question_structure_is_valid():
    """Test that generated questions have valid structure for interview system."""
    from pas.helpers.interview import generate_confirmation_questions
    
    context = {
        "psychological_extraction": {
            "hedging_detected": True,
            "hedging_markers": ["maybe"]
        }
    }
    
    result = generate_confirmation_questions(
        session_context=context,
        hypotheses=None
    )
    
    assert len(result) == 1
    q = result[0]
    
    # Required fields for interview system
    assert "id" in q
    assert "question_text" in q
    assert "question_type" in q
    assert "choices" in q
    assert "priority" in q
    assert "depth" in q
    assert "depends_on" in q
    assert "follow_up_rules" in q
    assert "answered" in q
    assert q["answered"] is False
    
    # Choices have required structure
    for choice in q["choices"]:
        assert "label" in choice
        assert "text" in choice
        assert "hidden_value" in choice
