"""Tests for Phase 4a: inotify file watcher."""

import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Skip all tests on non-Linux platforms
pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="inotify is Linux-only"
)

from pas.lsp.watcher import InotifyWatcher


class TestInotifyWatcher:
    """Tests for InotifyWatcher class."""
    
    def test_init_default_extensions(self):
        """Test default file extensions are set."""
        watcher = InotifyWatcher("/tmp/test")
        assert ".py" in watcher.extensions
        assert ".ts" in watcher.extensions
        assert ".tsx" in watcher.extensions
    
    def test_init_custom_extensions(self):
        """Test custom file extensions."""
        watcher = InotifyWatcher("/tmp/test", extensions=(".go", ".rs"))
        assert watcher.extensions == (".go", ".rs")
        assert ".py" not in watcher.extensions
    
    def test_ignore_patterns(self):
        """Test default ignore patterns."""
        watcher = InotifyWatcher("/tmp/test")
        assert "node_modules" in watcher.ignore_patterns
        assert ".venv" in watcher.ignore_patterns
        assert ".git" in watcher.ignore_patterns
    
    def test_should_watch_dir_allowed(self):
        """Test allowed directories pass filter."""
        watcher = InotifyWatcher("/tmp/test")
        assert watcher._should_watch_dir(Path("/tmp/test/src"))
        assert watcher._should_watch_dir(Path("/tmp/test/lib"))
    
    def test_should_watch_dir_ignored(self):
        """Test ignored directories are filtered."""
        watcher = InotifyWatcher("/tmp/test")
        assert not watcher._should_watch_dir(Path("/tmp/test/node_modules"))
        assert not watcher._should_watch_dir(Path("/tmp/test/.venv"))
        assert not watcher._should_watch_dir(Path("/tmp/test/__pycache__"))
    
    def test_should_notify_file_matches(self):
        """Test matching files pass filter."""
        watcher = InotifyWatcher("/tmp/test")
        assert watcher._should_notify_file("server.py")
        assert watcher._should_notify_file("component.tsx")
        assert watcher._should_notify_file("utils.ts")
    
    def test_should_notify_file_no_match(self):
        """Test non-matching files are filtered."""
        watcher = InotifyWatcher("/tmp/test")
        assert not watcher._should_notify_file("README.md")
        assert not watcher._should_notify_file("config.yaml")
        assert not watcher._should_notify_file("image.png")
    
    def test_is_running_initially_false(self):
        """Test watcher is not running initially."""
        watcher = InotifyWatcher("/tmp/test")
        assert not watcher.is_running
    
    def test_watched_directories_initially_zero(self):
        """Test watched directories is zero initially."""
        watcher = InotifyWatcher("/tmp/test")
        assert watcher.watched_directories == 0


class TestInotifyWatcherAsync:
    """Async tests for InotifyWatcher."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = InotifyWatcher(tmpdir)
            on_change = AsyncMock()
            
            await watcher.start(on_change)
            assert watcher.is_running
            assert watcher.watched_directories >= 1
            
            await watcher.stop()
            assert not watcher.is_running
            assert watcher.watched_directories == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            async with InotifyWatcher(tmpdir) as watcher:
                # Context manager doesn't auto-start
                assert not watcher.is_running
    
    @pytest.mark.asyncio
    async def test_watches_subdirectories(self):
        """Test that subdirectories are watched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories
            os.makedirs(os.path.join(tmpdir, "src", "components"))
            os.makedirs(os.path.join(tmpdir, "tests"))
            
            watcher = InotifyWatcher(tmpdir)
            await watcher.start(AsyncMock())
            
            # Should watch root + src + src/components + tests = 4
            assert watcher.watched_directories >= 4
            
            await watcher.stop()
    
    @pytest.mark.asyncio
    async def test_ignores_node_modules(self):
        """Test that node_modules is not watched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create node_modules
            os.makedirs(os.path.join(tmpdir, "node_modules", "lodash"))
            os.makedirs(os.path.join(tmpdir, "src"))
            
            watcher = InotifyWatcher(tmpdir)
            await watcher.start(AsyncMock())
            
            # node_modules should not be in watched paths
            watched_paths = [str(p) for p in watcher._watches.values()]
            assert not any("node_modules" in p for p in watched_paths)
            
            await watcher.stop()
