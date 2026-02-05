"""
LSP Server Pool - Persistent server management for fast queries.

v52c: Now uses subprocess isolation (subprocess_worker.py) to avoid
anyio task group conflicts with MCP's event loop.
"""

import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger("pas-server")

# Module-level singleton
_pool: Optional["LspPool"] = None


class LspPool:
    """
    Persistent LSP server pool with subprocess isolation.
    
    Instead of running LSP in the MCP process (which causes anyio conflicts),
    runs lsp-client in a subprocess and communicates via queues.
    
    Usage:
        pool = await LspPool.get("/path/to/project")
        refs = await pool.find_references(file, line, col)
        # No explicit cleanup - servers stay warm
    """
    
    def __init__(self, project_root: str, language: str = None):
        self.project_root = Path(project_root).resolve()
        self.language = language  # v53: Track language (None = auto-detect)
        self._subprocess: Optional[Any] = None
        self._started = False
    
    @classmethod
    async def get(cls, project_root: str, language: str = None) -> "LspPool":
        """
        Get or create pool for project.
        
        v53: Now supports language param. Auto-detects if not specified.
        Returns existing pool if project and language match.
        """
        global _pool
        
        resolved = Path(project_root).resolve()
        
        # v53: Auto-detect language if not specified
        if language is None:
            from pas.lsp.subprocess_worker import detect_project_language
            language = detect_project_language(str(resolved))
        
        # Reuse existing pool if same project AND language
        if (_pool is not None 
            and _pool.project_root == resolved 
            and _pool.language == language
            and _pool._started):
            return _pool
        
        # Close old pool if different project or language
        if _pool is not None and _pool._started:
            await _pool.shutdown()
        
        # Create and start new pool with language
        _pool = cls(str(resolved), language=language)
        await _pool._ensure_started()
        return _pool
    
    async def _ensure_started(self) -> None:
        """Start subprocess if not already running."""
        if self._started:
            return
        
        try:
            from pas.lsp.subprocess_worker import get_lsp_subprocess
            
            # v53: Pass language to subprocess
            self._subprocess = await get_lsp_subprocess(str(self.project_root), language=self.language)
            self._started = self._subprocess._started
            
            # Update language from subprocess (in case it was auto-detected)
            if self._subprocess:
                self.language = self._subprocess.language
            
            if self._started:
                logger.info(f"LspPool: Subprocess ready for {self.project_root} ({self.language})")
            else:
                logger.warning(f"LspPool: Subprocess failed to start for {self.project_root}")
        except Exception as e:
            logger.error(f"LspPool: Failed to start subprocess: {e}")
            self._started = False
    
    async def shutdown(self) -> None:
        """Stop subprocess (called when switching projects)."""
        if self._subprocess and self._started:
            try:
                self._subprocess.stop()
                logger.info(f"LspPool: Stopped subprocess for {self.project_root}")
            except Exception as e:
                logger.warning(f"LspPool: Error stopping subprocess: {e}")
        self._started = False
        self._subprocess = None
    
    # =========================================================================
    # Delegated LSP operations (all use subprocess)
    # =========================================================================
    
    async def find_references(
        self, 
        file_path: str, 
        line: int, 
        col: int
    ) -> list[dict]:
        """Find all references to symbol at position."""
        await self._ensure_started()
        if not self._subprocess or not self._started:
            return []
        return await self._subprocess.find_references(file_path, line, col)
    
    async def find_definition(
        self, 
        file_path: str, 
        line: int, 
        col: int
    ) -> Optional[dict]:
        """Find definition of symbol at position."""
        await self._ensure_started()
        if not self._subprocess or not self._started:
            return None
        return await self._subprocess.find_definition(file_path, line, col)
    
    async def call_hierarchy(
        self,
        file_path: str,
        line: int,
        col: int,
        direction: str = "incoming"
        ) -> list[dict]:
        """Get call hierarchy for symbol at position.
        
        Uses subprocess to call LSP prepareCallHierarchy + incomingCalls/outgoingCalls.
        """
        await self._ensure_started()
        if not self._subprocess or not self._started:
            return []
        return await self._subprocess.call_hierarchy(file_path, line, col, direction)

    
    async def document_symbols(self, file_path: str) -> list[dict]:
        """Get all symbols in a document via subprocess."""
        await self._ensure_started()
        if not self._subprocess or not self._started:
            return []
        return await self._subprocess.document_symbols(file_path)
    
    @property
    def available_languages(self) -> list[str]:
        """Languages with running servers."""
        if not self._subprocess or not self._started:
            return []
        # v53: Return actual language being used
        return [self.language] if self.language else ["python"]


async def get_pool(project_root: str, language: str = None) -> LspPool:
    """Convenience function to get the LSP pool.
    
    v53: Now accepts optional language param.
    """
    return await LspPool.get(project_root, language=language)


async def shutdown_pool() -> None:
    """Shutdown the global pool (for cleanup)."""
    global _pool
    if _pool:
        await _pool.shutdown()
        _pool = None
