"""File watcher using Linux inotify.

Provides async file watching for triggering sync_project delta updates.
Uses ctypes to access Linux inotify syscalls - no external dependencies.
"""

import asyncio
import ctypes
import ctypes.util
import logging
import os
import struct
from pathlib import Path
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)

# inotify event constants (from linux/inotify.h)
IN_MODIFY = 0x00000002
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CLOSE_WRITE = 0x00000008  # File closed after writing

# Event struct format: int wd, uint32_t mask, uint32_t cookie, uint32_t len
EVENT_HEADER_SIZE = struct.calcsize("iIII")

# Load libc for inotify syscalls
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _inotify_init() -> int:
    """Initialize inotify instance."""
    fd = _libc.inotify_init()
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd


def _inotify_add_watch(fd: int, path: str, mask: int) -> int:
    """Add a watch to an inotify instance."""
    path_bytes = path.encode('utf-8')
    wd = _libc.inotify_add_watch(fd, path_bytes, mask)
    if wd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return wd


def _inotify_rm_watch(fd: int, wd: int) -> None:
    """Remove a watch from an inotify instance."""
    result = _libc.inotify_rm_watch(fd, wd)
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


class InotifyWatcher:
    """Async file watcher using Linux inotify.
    
    Watches for file changes in a project directory and triggers
    callbacks on modifications. Designed for triggering sync_project
    delta updates on save.
    
    Example:
        async def on_file_change(path: str):
            await sync_project_incremental(path)
        
        watcher = InotifyWatcher("/path/to/project")
        await watcher.start(on_file_change)
    """
    
    def __init__(
        self,
        project_path: str,
        extensions: tuple[str, ...] = (".py", ".ts", ".tsx", ".js", ".jsx"),
        ignore_patterns: tuple[str, ...] = ("node_modules", ".venv", "__pycache__", ".git", "dist"),
    ):
        """Initialize watcher.
        
        Args:
            project_path: Root directory to watch
            extensions: File extensions to watch
            ignore_patterns: Directory patterns to ignore
        """
        self.project_path = Path(project_path).resolve()
        self.extensions = extensions
        self.ignore_patterns = ignore_patterns
        
        self._fd: Optional[int] = None
        self._watches: dict[int, Path] = {}  # wd -> directory path
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def _should_watch_dir(self, path: Path) -> bool:
        """Check if directory should be watched."""
        for pattern in self.ignore_patterns:
            if pattern in path.parts:
                return False
        return True
    
    def _should_notify_file(self, filename: str) -> bool:
        """Check if file matches watched extensions."""
        return any(filename.endswith(ext) for ext in self.extensions)
    
    async def start(self, on_change: Callable[[str], None]) -> None:
        """Start watching for file changes.
        
        Args:
            on_change: Async callback(file_path) called on file changes
        """
        if self._running:
            logger.warning("Watcher already running")
            return
        
        # Initialize inotify
        self._fd = _inotify_init()
        
        # Add watches for all directories
        await self._add_watches()
        
        self._running = True
        self._task = asyncio.create_task(self._watch_loop(on_change))
        logger.info(f"Watcher started: {len(self._watches)} directories")
    
    async def _add_watches(self) -> None:
        """Add inotify watches for all directories."""
        mask = IN_CLOSE_WRITE | IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO
        
        for root, dirs, _ in os.walk(self.project_path):
            root_path = Path(root)
            
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if self._should_watch_dir(root_path / d)]
            
            if self._should_watch_dir(root_path):
                try:
                    wd = _inotify_add_watch(self._fd, str(root_path), mask)
                    self._watches[wd] = root_path
                except OSError as e:
                    logger.warning(f"Failed to watch {root_path}: {e}")
    
    async def _watch_loop(self, on_change: Callable[[str], None]) -> None:
        """Main event loop reading inotify events."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                # Read events (blocking in executor to not block event loop)
                data = await loop.run_in_executor(
                    None, lambda: os.read(self._fd, 4096)
                )
                
                # Parse events
                offset = 0
                while offset < len(data):
                    wd, mask, cookie, length = struct.unpack_from(
                        "iIII", data, offset
                    )
                    offset += EVENT_HEADER_SIZE
                    
                    if length:
                        filename = data[offset:offset + length].rstrip(b'\x00').decode('utf-8')
                        offset += length
                    else:
                        filename = ""
                    
                    # Handle event
                    if wd in self._watches and self._should_notify_file(filename):
                        file_path = self._watches[wd] / filename
                        logger.debug(f"File changed: {file_path}")
                        
                        try:
                            if asyncio.iscoroutinefunction(on_change):
                                await on_change(str(file_path))
                            else:
                                on_change(str(file_path))
                        except Exception as e:
                            logger.error(f"on_change callback failed: {e}")
                            
            except OSError as e:
                if self._running:
                    logger.error(f"Watcher error: {e}")
                break
    
    async def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        # Remove all watches
        for wd in list(self._watches.keys()):
            try:
                _inotify_rm_watch(self._fd, wd)
            except OSError:
                pass
        self._watches.clear()
        
        # Close inotify fd
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        
        logger.info("Watcher stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._running
    
    @property
    def watched_directories(self) -> int:
        """Number of directories being watched."""
        return len(self._watches)
    
    async def __aenter__(self) -> "InotifyWatcher":
        """Async context manager entry (does not auto-start)."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()


class DebouncedSyncHandler:
    """Debounces file changes and triggers sync after quiet period.
    
    Multiple rapid saves to the same file result in only one sync
    after the debounce period expires.
    
    Example:
        handler = DebouncedSyncHandler(my_sync_func, debounce_seconds=2.0)
        watcher = InotifyWatcher("/path/to/project")
        await watcher.start(handler.on_change)
    """
    
    def __init__(
        self, 
        sync_callback: Callable[[str], None], 
        debounce_seconds: float = 2.0
    ):
        """Initialize handler.
        
        Args:
            sync_callback: Async function(file_path) to call after debounce
            debounce_seconds: Wait time after last change before syncing
        """
        self.sync_callback = sync_callback
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, asyncio.Task] = {}
    
    async def on_change(self, file_path: str) -> None:
        """Handle file change with debouncing."""
        # Cancel existing pending sync for this file
        if file_path in self._pending:
            self._pending[file_path].cancel()
            try:
                await self._pending[file_path]
            except asyncio.CancelledError:
                pass
        
        # Schedule new sync after debounce period
        self._pending[file_path] = asyncio.create_task(
            self._delayed_sync(file_path)
        )
    
    async def _delayed_sync(self, file_path: str) -> None:
        """Wait for debounce period then execute sync."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            self._pending.pop(file_path, None)
            
            logger.debug(f"Debounce complete, syncing: {file_path}")
            
            if asyncio.iscoroutinefunction(self.sync_callback):
                await self.sync_callback(file_path)
            else:
                self.sync_callback(file_path)
                
        except asyncio.CancelledError:
            # Another change came in, this sync was cancelled
            pass
        except Exception as e:
            logger.error(f"Auto-sync failed for {file_path}: {e}")
    
    @property
    def pending_count(self) -> int:
        """Number of files waiting to sync."""
        return len(self._pending)

