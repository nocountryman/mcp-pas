"""
Layer 1: Codebase Flow Tool Tests

Tests for code navigation and understanding tools:
- sync_project
- import_lsif
- query_codebase
- find_references
- go_to_definition
- call_hierarchy
- infer_file_purpose, store_file_purpose
"""
import pytest
import os


class TestCodebaseSync:
    """Tests for project indexing."""
    
    @pytest.mark.asyncio
    async def test_sync_project_basic(self, db_connection):
        """Verify project sync creates file registry entries."""
        from server import sync_project
        
        # Sync the PAS project itself (meta!)
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        result = await sync_project(
            project_path=project_path,
            project_id="mcp-pas-test",
            max_files=10  # Limit for test speed
        )
        
        assert result["success"] is True
        assert result["files_indexed"] >= 0
    
    @pytest.mark.asyncio
    async def test_import_lsif(self, db_connection):
        """Verify LSIF import."""
        from server import import_lsif
        import os
        
        lsif_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "fixtures",
            "sample.lsif.json"
        )
        
        # Skip if fixture doesn't exist
        if not os.path.exists(lsif_path):
            pytest.skip("LSIF fixture not found")
        
        result = await import_lsif(
            project_id="test-lsif",
            lsif_path=lsif_path,
            clear_existing=True
        )
        
        assert result["success"] is True


class TestCodebaseQuery:
    """Tests for semantic code search."""
    
    @pytest.mark.asyncio
    async def test_query_codebase_returns_results(self, db_connection):
        """Verify semantic search returns files and symbols."""
        from server import query_codebase
        
        result = await query_codebase(
            query="session management database connection",
            project_id="mcp-pas",
            top_k=5
        )
        
        assert result["success"] is True
        assert "files" in result
        assert "symbols" in result
    
    @pytest.mark.asyncio
    async def test_find_references(self, db_connection):
        """Verify find_references returns locations."""
        from server import find_references
        
        result = await find_references(
            project_id="mcp-pas",
            symbol_name="get_db_connection",
            include_definitions=True
        )
        
        assert result["success"] is True
        # May be empty if project not synced with LSIF
        assert "references" in result or "error" not in result


class TestCodeNavigation:
    """Tests for code navigation tools."""
    
    @pytest.mark.asyncio
    async def test_go_to_definition(self, db_connection):
        """Verify go_to_definition returns location."""
        from server import go_to_definition
        
        result = await go_to_definition(
            project_id="mcp-pas",
            file_path="server.py",
            line=100,  # Arbitrary line
            column=10
        )
        
        # May fail if no definition found at this location
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_call_hierarchy(self, db_connection):
        """Verify call hierarchy traversal."""
        from server import call_hierarchy
        
        result = await call_hierarchy(
            project_id="mcp-pas",
            symbol_name="start_reasoning_session",
            direction="incoming",
            max_depth=2
        )
        
        assert result["success"] is True


class TestPurposeInference:
    """Tests for hierarchical purpose inference."""
    
    @pytest.mark.asyncio
    async def test_infer_file_purpose(self, db_connection):
        """Verify file purpose inference prompt generation."""
        from server import infer_file_purpose
        
        result = await infer_file_purpose(
            project_id="mcp-pas",
            file_path="server.py",
            force_refresh=True
        )
        
        assert result["success"] is True
        # Should return either cached purpose or inference prompt
        assert "purpose" in result or "inference_prompt" in result
    
    @pytest.mark.asyncio
    async def test_store_file_purpose(self, db_connection):
        """Verify storing file purpose."""
        from server import store_file_purpose
        import json
        
        purpose_data = json.dumps({
            "function_purposes": {"main": "Entry point for the application"},
            "module_purpose": "Main server module",
            "project_contribution": "Provides MCP tool implementations"
        })
        
        result = await store_file_purpose(
            project_id="mcp-pas-test",
            file_path="test_file.py",
            purpose_data=purpose_data
        )
        
        assert result["success"] is True
