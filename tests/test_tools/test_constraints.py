"""
Tests for Phase 7c Environment Constraints MCP tools.

Tests:
- constraint_validation helpers (get_active_constraints, validate_hypothesis)
- store_extracted_constraints MCP tool
- detect_constraint_drift MCP tool
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock


class TestConstraintValidation:
    """Tests for constraint_validation.py helpers."""
    
    def test_get_active_constraints_returns_list(self):
        """Test that get_active_constraints returns a list for valid project."""
        from pas.helpers.constraint_validation import get_active_constraints
        
        constraints = get_active_constraints("mcp-pas")
        assert isinstance(constraints, list)
        # mcp-pas should have at least no_mvp from seeding
        assert len(constraints) >= 1
    
    def test_get_active_constraints_empty_for_unknown_project(self):
        """Test that unknown project returns empty list."""
        from pas.helpers.constraint_validation import get_active_constraints
        
        constraints = get_active_constraints("nonexistent-project-xyz")
        assert constraints == []
    
    def test_validate_hypothesis_blocks_mvp_language(self):
        """Test that MVP language triggers blocking violation."""
        from pas.helpers.constraint_validation import validate_hypothesis
        
        # Test various MVP patterns (must match patterns in constraint_validation.py)
        mvp_texts = [
            "Build a v1 MVP of the feature",
            "Create a minimum viable product",
            "Build MVP first iteration",
            "Do MVP first, then iterate",
        ]
        
        for text in mvp_texts:
            result = validate_hypothesis(text, "mcp-pas")
            assert result["blocked"] is True, f"Should block: {text}"
            assert len(result["violations"]) >= 1
    
    def test_validate_hypothesis_passes_clean_text(self):
        """Test that clean production language passes."""
        from pas.helpers.constraint_validation import validate_hypothesis
        
        clean_texts = [
            "Build a production-grade authentication system",
            "Implement comprehensive error handling with logging",
            "Create full test coverage for the feature",
        ]
        
        for text in clean_texts:
            result = validate_hypothesis(text, "mcp-pas")
            assert result["passed"] is True, f"Should pass: {text}"
            assert result["blocked"] is False
    
    def test_get_constraint_summary_format(self):
        """Test that constraint summary returns expected structure."""
        from pas.helpers.constraint_validation import get_constraint_summary
        
        summary = get_constraint_summary("mcp-pas")
        
        assert "constraint_count" in summary
        assert "blocking_count" in summary
        assert "constraints" in summary
        assert summary["constraint_count"] >= 1


class TestGeminiSync:
    """Tests for gemini_sync.py helpers."""
    
    def test_parse_gemini_constraints_returns_prompt(self):
        """Test that GEMINI.md parsing returns extraction prompt."""
        from pathlib import Path
        from pas.helpers.gemini_sync import parse_gemini_constraints
        
        result = parse_gemini_constraints(Path("/home/nocoma/Documents/MCP/PAS/GEMINI.md"))
        
        assert result["success"] is True
        assert "extraction_prompt" in result
        assert len(result["extraction_prompt"]) > 100
    
    def test_parse_gemini_constraints_missing_file(self):
        """Test graceful handling of missing GEMINI.md."""
        from pathlib import Path
        from pas.helpers.gemini_sync import parse_gemini_constraints
        
        result = parse_gemini_constraints(Path("/nonexistent/GEMINI.md"))
        
        assert result["success"] is False
        assert "error" in result


class TestMCPTools:
    """Integration tests for MCP tools (requires running server)."""
    
    @pytest.mark.asyncio
    async def test_store_extracted_constraints_requires_list(self):
        """Test that invalid JSON format is rejected."""
        from pas.server import store_extracted_constraints
        
        result = await store_extracted_constraints(
            project_id="mcp-pas",
            constraints_json='{"not": "a list"}'
        )
        
        assert result["success"] is False
        assert "must be a JSON array" in result["error"]
    
    @pytest.mark.asyncio
    async def test_store_extracted_constraints_invalid_json(self):
        """Test that invalid JSON is rejected."""
        from pas.server import store_extracted_constraints
        
        result = await store_extracted_constraints(
            project_id="mcp-pas",
            constraints_json='not valid json'
        )
        
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]
    
    @pytest.mark.asyncio
    async def test_detect_constraint_drift_invalid_json(self):
        """Test that invalid JSON is rejected."""
        from pas.server import detect_constraint_drift
        
        result = await detect_constraint_drift(
            project_id="mcp-pas",
            constraints_json='not valid json'
        )
        
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]
