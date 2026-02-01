"""Tests for pas.lsp module."""

import pytest
from pathlib import Path

from pas.lsp.config import detect_subprojects
from pas.lsp.manager import LspManager


class TestDetectSubprojects:
    """Tests for detect_subprojects function."""
    
    def test_detects_python_at_root(self, tmp_path):
        """Detects Python project from pyproject.toml."""
        (tmp_path / "pyproject.toml").touch()
        result = detect_subprojects(str(tmp_path))
        
        assert len(result) == 1
        assert result[0]["language"] == "python"
        assert result[0]["path"] == str(tmp_path)
    
    def test_detects_typescript_at_root(self, tmp_path):
        """Detects TypeScript project from tsconfig.json + package.json."""
        (tmp_path / "package.json").touch()
        (tmp_path / "tsconfig.json").touch()
        result = detect_subprojects(str(tmp_path))
        
        assert len(result) == 1
        assert result[0]["language"] == "typescript"
        assert result[0]["path"] == str(tmp_path)
    
    def test_requires_tsconfig_for_typescript(self, tmp_path):
        """package.json alone is not sufficient for TypeScript detection."""
        (tmp_path / "package.json").touch()  # No tsconfig
        result = detect_subprojects(str(tmp_path))
        
        assert len(result) == 0
    
    def test_detects_monorepo_subdirs(self, tmp_path):
        """Detects projects in subdirectories (monorepo pattern)."""
        # Create backend/
        backend = tmp_path / "backend"
        backend.mkdir()
        (backend / "pyproject.toml").touch()
        
        # Create frontend/
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "package.json").touch()
        (frontend / "tsconfig.json").touch()
        
        result = detect_subprojects(str(tmp_path))
        languages = {r["language"] for r in result}
        paths = {r["path"] for r in result}
        
        assert languages == {"python", "typescript"}
        assert str(backend) in paths
        assert str(frontend) in paths
    
    def test_ignores_hidden_dirs(self, tmp_path):
        """Hidden directories are not scanned."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "pyproject.toml").touch()
        
        result = detect_subprojects(str(tmp_path))
        
        assert len(result) == 0
    
    def test_nonexistent_path_returns_empty(self, tmp_path):
        """Gracefully handles non-existent paths."""
        result = detect_subprojects(str(tmp_path / "does_not_exist"))
        
        assert result == []
    
    def test_detects_both_root_and_subdir(self, tmp_path):
        """Detects projects at both root and subdirs."""
        # Root Python project
        (tmp_path / "pyproject.toml").touch()
        
        # Subdir TypeScript project
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "package.json").touch()
        (frontend / "tsconfig.json").touch()
        
        result = detect_subprojects(str(tmp_path))
        
        assert len(result) == 2
        languages = {r["language"] for r in result}
        assert languages == {"python", "typescript"}


class TestLspManager:
    """Tests for LspManager class."""
    
    def test_init_with_empty_project(self, tmp_path):
        """LspManager initializes with no subprojects."""
        manager = LspManager(str(tmp_path))
        
        assert manager.subprojects == []
        assert manager.servers == {}
    
    def test_init_detects_subprojects(self, tmp_path):
        """LspManager detects subprojects on init."""
        (tmp_path / "pyproject.toml").touch()
        manager = LspManager(str(tmp_path))
        
        assert len(manager.subprojects) == 1
        assert manager.subprojects[0]["language"] == "python"
    
    def test_get_language_for_python_file(self, tmp_path):
        """Correctly identifies Python files."""
        manager = LspManager(str(tmp_path))
        
        assert manager._get_language_for_file("test.py") == "python"
        assert manager._get_language_for_file("/path/to/module.py") == "python"
    
    def test_get_language_for_typescript_files(self, tmp_path):
        """Correctly identifies TypeScript/JavaScript files."""
        manager = LspManager(str(tmp_path))
        
        assert manager._get_language_for_file("test.ts") == "typescript"
        assert manager._get_language_for_file("test.tsx") == "typescript"
        assert manager._get_language_for_file("test.js") == "typescript"
        assert manager._get_language_for_file("test.jsx") == "typescript"
    
    def test_get_language_for_unknown_file(self, tmp_path):
        """Returns None for unknown file types."""
        manager = LspManager(str(tmp_path))
        
        assert manager._get_language_for_file("test.txt") is None
        assert manager._get_language_for_file("test.md") is None
        assert manager._get_language_for_file("Makefile") is None
    
    def test_available_languages_empty_when_not_started(self, tmp_path):
        """available_languages is empty before starting servers."""
        (tmp_path / "pyproject.toml").touch()
        manager = LspManager(str(tmp_path))
        
        assert manager.available_languages == []


class TestLspManagerAsync:
    """Async tests for LspManager (integration)."""
    
    @pytest.mark.asyncio
    async def test_find_references_without_server_returns_empty(self, tmp_path):
        """find_references returns empty when no server running."""
        manager = LspManager(str(tmp_path))
        result = await manager.find_references("test.py", 0, 0)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_find_definition_without_server_returns_none(self, tmp_path):
        """find_definition returns None when no server running."""
        manager = LspManager(str(tmp_path))
        result = await manager.find_definition("test.py", 0, 0)
        
        assert result is None
