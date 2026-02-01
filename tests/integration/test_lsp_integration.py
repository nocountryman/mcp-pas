"""Integration tests for pas.lsp module.

These tests require actual language servers to be installed.
Tests are skipped if the required servers are not available.
"""

import shutil
import pytest

from pas.lsp import LspManager
from pas.lsp.config import get_config_for_language, PYTHON_CONFIG, TYPESCRIPT_CONFIG


# Check for language server availability
PYRIGHT_AVAILABLE = (
    shutil.which("pyright") is not None or 
    shutil.which("basedpyright") is not None
)
TSSERVER_AVAILABLE = shutil.which("tsserver") is not None


class TestLanguageConfigs:
    """Tests for language configuration functions."""
    
    def test_python_config_structure(self):
        """PYTHON_CONFIG has required fields."""
        assert "code_language" in PYTHON_CONFIG
        assert PYTHON_CONFIG["code_language"] == "python"
    
    def test_typescript_config_structure(self):
        """TYPESCRIPT_CONFIG has required fields."""
        assert "code_language" in TYPESCRIPT_CONFIG
        assert TYPESCRIPT_CONFIG["code_language"] == "typescript"
    
    def test_get_config_for_python(self):
        """get_config_for_language returns Python config."""
        config = get_config_for_language("python")
        assert config["code_language"] == "python"
    
    def test_get_config_for_typescript(self):
        """get_config_for_language returns TypeScript config."""
        config = get_config_for_language("typescript")
        assert config["code_language"] == "typescript"
    
    def test_get_config_for_unknown_language(self):
        """get_config_for_language returns minimal config for unknown language."""
        config = get_config_for_language("rust")
        assert config["code_language"] == "rust"


@pytest.mark.integration
@pytest.mark.skipif(not PYRIGHT_AVAILABLE, reason="pyright/basedpyright not installed")
class TestPythonIntegration:
    """Integration tests with Python LSP server (pyright)."""
    
    @pytest.mark.asyncio
    async def test_start_python_server(self):
        """LspManager can start Python language server."""
        async with LspManager("/home/nocoma/Documents/MCP/PAS") as mgr:
            # Server should be started for Python project
            assert "python" in mgr.available_languages
    
    @pytest.mark.asyncio
    async def test_find_references_returns_list(self):
        """find_references returns a list (may be empty if no refs found)."""
        async with LspManager("/home/nocoma/Documents/MCP/PAS") as mgr:
            # Test on a known file - results depend on LSP indexing
            refs = await mgr.find_references("src/pas/server.py", 0, 0)
            assert isinstance(refs, list)
    
    @pytest.mark.asyncio
    async def test_find_definition_on_import(self):
        """find_definition works on import statements."""
        async with LspManager("/home/nocoma/Documents/MCP/PAS") as mgr:
            # Line 0 typically has imports - definition may or may not resolve
            result = await mgr.find_definition("src/pas/server.py", 0, 5)
            # Result can be dict or None, just verify no crash
            assert result is None or isinstance(result, dict)


@pytest.mark.integration  
@pytest.mark.skipif(not TSSERVER_AVAILABLE, reason="tsserver not installed")
class TestTypescriptIntegration:
    """Integration tests with TypeScript LSP server (tsserver).
    
    Note: These tests require a TypeScript project with node_modules installed.
    """
    
    @pytest.mark.asyncio
    async def test_typescript_detection(self, tmp_path):
        """LspManager detects TypeScript projects."""
        # Create minimal TS project structure
        (tmp_path / "package.json").write_text('{"name": "test"}')
        (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}')
        
        mgr = LspManager(str(tmp_path))
        assert any(s["language"] == "typescript" for s in mgr.subprojects)
