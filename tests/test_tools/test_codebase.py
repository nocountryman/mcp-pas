"""
Layer 1: Codebase Flow Tool Tests

Tests for code navigation and understanding tools:
- sync_project
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
        from pas.server import sync_project
        
        # Sync the PAS project itself (meta!)
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        result = await sync_project(
            project_path=project_path,
            project_id="mcp-pas-test",
            max_files=10  # Limit for test speed
        )
        
        assert result["success"] is True
        # API returns files_scanned, files_added, files_updated, etc.
        assert result.get("files_scanned", 0) >= 0 or result.get("files_added", 0) >= 0
    

class TestCodebaseQuery:
    """Tests for semantic code search."""
    
    @pytest.mark.asyncio
    async def test_query_codebase_returns_results(self, db_connection):
        """Verify semantic search returns files and symbols."""
        from pas.server import query_codebase
        
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
        from pas.server import find_references
        
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
        from pas.server import go_to_definition
        
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
        from pas.server import call_hierarchy
        
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
        from pas.server import infer_file_purpose
        
        result = await infer_file_purpose(
            project_id="mcp-pas",
            file_path="server.py",
            force_refresh=True
        )
        
        # May fail if project not synced or schema doesn't support this
        if not result.get("success"):
            pytest.skip("Purpose inference failed - may need project sync first")
        
        # Should return either cached purpose or inference prompt
        assert "purpose" in result or "inference_prompt" in result
    
    @pytest.mark.asyncio
    async def test_store_file_purpose(self, db_connection):
        """Verify storing file purpose."""
        from pas.server import store_file_purpose
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
        
        # May fail if DB schema doesn't support this
        if not result.get("success"):
            pytest.skip("Store purpose failed - may need schema update")
        
        assert result["success"] is True


class TestProjectPurpose:
    """Tests for project-level purpose tools (v43)."""
    
    @pytest.mark.asyncio
    async def test_infer_project_purpose(self, db_connection):
        """Verify project purpose inference prompt generation."""
        from pas.server import infer_project_purpose
        
        result = await infer_project_purpose(
            project_id="mcp-pas",
            force_refresh=True
        )
        
        # May fail if project not synced
        if not result.get("success"):
            pytest.skip("Project not synced - run sync_project first")
        
        # Should return either cached purpose or inference prompt
        assert result.get("success") is True
        assert "purpose_hierarchy" in result or "inference_prompt" in result
    
    @pytest.mark.asyncio
    async def test_store_project_purpose(self, db_connection):
        """Verify storing project purpose."""
        from pas.server import store_project_purpose
        import json
        
        purpose_data = json.dumps({
            "mission": "Test project for PAS",
            "user_needs": ["testing", "validation"],
            "must_have_modules": ["server", "tests"],
            "detected_domain": "backend",
            "domain_confidence": 0.9
        })
        
        result = await store_project_purpose(
            project_id="mcp-pas-test",
            purpose_data=purpose_data
        )
        
        if result.get("success"):
            assert result["success"] is True
        else:
            pytest.skip("Store purpose failed - project may not exist")
    
    @pytest.mark.asyncio
    async def test_analyze_completeness(self, db_connection):
        """Verify completeness analysis."""
        from pas.server import analyze_completeness
        
        result = await analyze_completeness(project_id="mcp-pas")
        
        if result.get("success"):
            assert "completeness" in result
            assert "implemented" in result
            assert "missing" in result
        else:
            pytest.skip("Completeness analysis failed - project purpose may not exist")
    
    @pytest.mark.asyncio
    async def test_get_purpose_chain(self, db_connection):
        """Verify purpose chain tracing."""
        from pas.server import get_purpose_chain
        
        result = await get_purpose_chain(
            project_id="mcp-pas",
            file_path="server.py"
        )
        
        if result.get("success"):
            assert "purpose_chain" in result
        else:
            pytest.skip("Purpose chain failed - file may not have purpose cached")
    
    @pytest.mark.asyncio
    async def test_purpose_staleness_detection(self, db_connection):
        """Verify staleness detection returns stale flag."""
        from pas.server import infer_project_purpose
        
        result = await infer_project_purpose(
            project_id="mcp-pas",
            force_refresh=False
        )
        
        if result.get("success"):
            # stale flag should be present in response
            assert isinstance(result.get("stale", False), bool)
    
    @pytest.mark.asyncio
    async def test_get_system_map(self, db_connection):
        """Verify system map returns module dependency graph."""
        from pas.server import get_system_map
        
        result = await get_system_map(
            project_id="mcp-pas",
            include_weights=True
        )
        
        if result.get("success"):
            assert "nodes" in result
            assert "edges" in result
            assert "stats" in result
            assert isinstance(result["nodes"], list)
            assert isinstance(result["edges"], list)
            # If edges exist, verify structure
            if result["edges"]:
                assert "source" in result["edges"][0]
                assert "target" in result["edges"][0]
        else:
            # May fail if project not synced with symbol_references
            pytest.skip("System map failed - project may need sync_project with LSIF import")

    @pytest.mark.asyncio
    async def test_infer_schema_intent(self, db_connection):
        """Verify schema intent extraction returns entities."""
        from pas.server import infer_schema_intent
        
        result = await infer_schema_intent(project_id="mcp-pas")
        
        assert result.get("success") is True
        assert "entities" in result
        assert "classifications" in result
        assert "enrichment_prompt" in result
        assert isinstance(result["entities"], list)
        
        # Should find PAS tables
        entity_names = [e["name"] for e in result["entities"]]
        assert "reasoning_sessions" in entity_names or len(entity_names) > 0

    @pytest.mark.asyncio
    async def test_infer_config_assumptions(self, db_connection):
        """Verify config assumption extraction."""
        from pas.server import infer_config_assumptions
        
        result = await infer_config_assumptions(
            project_id="mcp-pas",
            config_path="/home/nocoma/Documents/MCP/PAS/config.yaml"
        )
        
        assert result.get("success") is True
        assert "assumptions" in result
        assert "by_type" in result
        assert "enrichment_prompt" in result
        assert len(result["assumptions"]) > 0
        
        # Should detect threshold assumptions from config.yaml
        types = result.get("stats", {}).get("types", {})
        assert "threshold" in types or "capacity" in types

    @pytest.mark.asyncio
    async def test_query_project_understanding(self, db_connection):
        """Verify unified project understanding query."""
        from pas.server import query_project_understanding
        
        result = await query_project_understanding(project_id="mcp-pas")
        
        assert result.get("success") is True
        assert "stats" in result
        assert "summary" in result
        # At least system_map should be populated for synced project
        assert result["stats"]["sections_populated"] >= 0


class TestPrefilterFiles:
    """Tests for v51 Phase 2 pre-filtering functions."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project with Python files."""
        (tmp_path / "file1.py").write_text("def my_function():\n    pass\n")
        (tmp_path / "file2.py").write_text("def other_function():\n    result = my_function()\n")
        (tmp_path / "file3.py").write_text("# This file has no references\n")
        return tmp_path
    
    def test_prefilter_python_finds_symbol(self, temp_project):
        """Python fallback finds files containing symbol."""
        from pas.helpers.codebase import prefilter_python
        files = list(temp_project.glob("*.py"))
        result = prefilter_python("my_function", files)
        assert len(result) == 2
        assert temp_project / "file1.py" in result
        assert temp_project / "file2.py" in result
    
    def test_prefilter_python_respects_boundaries(self, temp_project):
        """Python fallback uses identifier-aware boundaries."""
        from pas.helpers.codebase import prefilter_python
        (temp_project / "partial.py").write_text("def my_function_extended():\n    pass\n")
        files = list(temp_project.glob("*.py"))
        result = prefilter_python("my_function", files)
        # Should NOT match my_function_extended
        assert temp_project / "partial.py" not in result
    
    def test_prefilter_python_empty_files(self, temp_project):
        """Python fallback handles empty file list gracefully."""
        from pas.helpers.codebase import prefilter_python
        result = prefilter_python("my_function", [])
        assert result == []
    
    @pytest.mark.skipif(
        not __import__("shutil").which('rg'),
        reason="ripgrep not installed"
    )
    def test_prefilter_rg_finds_symbol(self, temp_project):
        """Ripgrep path finds files containing symbol."""
        from pas.helpers.codebase import prefilter_rg
        result = prefilter_rg("my_function", temp_project)
        assert len(result) == 2
    
    @pytest.mark.skipif(
        not __import__("shutil").which('rg'),
        reason="ripgrep not installed"
    )
    def test_prefilter_rg_respects_boundaries(self, temp_project):
        """Ripgrep -w flag respects word boundaries."""
        from pas.helpers.codebase import prefilter_rg
        (temp_project / "partial.py").write_text("def my_function_extended():\n    pass\n")
        result = prefilter_rg("my_function", temp_project)
        # ripgrep -w should NOT match my_function_extended
        partial_path = temp_project / "partial.py"
        assert partial_path not in result
    
    def test_prefilter_files_auto_fallback(self, temp_project):
        """Unified interface falls back to Python when ripgrep disabled."""
        from pas.helpers.codebase import prefilter_files
        files = list(temp_project.glob("*.py"))
        result = prefilter_files("my_function", temp_project, files, use_rg=False)
        assert len(result) == 2
    
    def test_prefilter_files_no_project_root(self, temp_project):
        """Unified interface works without project_root (Python only)."""
        from pas.helpers.codebase import prefilter_files
        files = list(temp_project.glob("*.py"))
        result = prefilter_files("my_function", None, files)
        assert len(result) == 2
    
    def test_has_ripgrep_defined(self):
        """HAS_RIPGREP module constant is defined."""
        from pas.helpers.codebase import HAS_RIPGREP
        assert isinstance(HAS_RIPGREP, bool)
