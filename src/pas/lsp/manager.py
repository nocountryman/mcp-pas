"""LSP Manager - orchestrates language servers via lsp-client.

v52b: Migrated from multilspy (broken JediServer) to lsp-client with
BasedpyrightClient for Python and TypescriptClient for JS/TS.
"""

import logging
from pathlib import Path
from typing import Optional, Any

from lsp_client import (
    BasedpyrightClient,
    TypescriptClient,
    LocalServer,
    Position,
)

from pas.lsp.config import detect_subprojects

logger = logging.getLogger(__name__)


class LspManager:
    """Manages language servers for a project.
    
    Uses lsp-client to wrap language servers (basedpyright, tsserver) and provides
    a unified interface for code navigation operations.
    
    v52b: Now uses lsp-client instead of multilspy (which had broken Python support).
    
    Usage:
        async with LspManager("/path/to/project") as mgr:
            refs = await mgr.find_references("src/main.py", 10, 5)
    """
    
    def __init__(self, project_root: str):
        """Initialize LspManager.
        
        Args:
            project_root: Absolute path to project root directory.
        """
        self.project_root = Path(project_root).resolve()
        self.clients: dict[str, Any] = {}  # lang -> client instance
        self._started: dict[str, bool] = {}
        self.subprojects = detect_subprojects(str(self.project_root))
    
    async def start_servers(self) -> None:
        """Start language servers for detected subprojects."""
        for subproject in self.subprojects:
            lang = subproject["language"]
            path = Path(subproject["path"]).resolve()
            
            if lang in self.clients and self._started.get(lang):
                continue  # Already started
            
            try:
                if lang == "python":
                    await self._start_python_server(path)
                elif lang == "typescript":
                    await self._start_typescript_server(path)
                else:
                    logger.debug(f"Unsupported language: {lang}")
            except Exception as e:
                logger.warning(f"Failed to start {lang} LSP server: {e}")
    
    async def _start_python_server(self, workspace: Path) -> None:
        """Start basedpyright for Python."""
        server = LocalServer(
            program="basedpyright-langserver",
            args=["--stdio"],
            cwd=workspace
        )
        client = BasedpyrightClient(
            server=server,
            workspace=str(workspace)
        )
        await client.__aenter__()
        self.clients["python"] = client
        self._started["python"] = True
        logger.info(f"Started Python LSP server (basedpyright) for {workspace}")
    
    async def _start_typescript_server(self, workspace: Path) -> None:
        """Start typescript-language-server for JS/TS."""
        server = LocalServer(
            program="typescript-language-server",
            args=["--stdio"],
            cwd=workspace
        )
        client = TypescriptClient(
            server=server,
            workspace=str(workspace)
        )
        await client.__aenter__()
        self.clients["typescript"] = client
        self._started["typescript"] = True
        logger.info(f"Started TypeScript LSP server for {workspace}")
    
    async def stop_servers(self) -> None:
        """Stop all running language servers."""
        for lang in list(self.clients.keys()):
            try:
                client = self.clients[lang]
                await client.__aexit__(None, None, None)
                del self.clients[lang]
                self._started[lang] = False
                logger.info(f"Stopped {lang} LSP server")
            except Exception as e:
                logger.warning(f"Error stopping {lang} LSP server: {e}")
    
    async def find_references(
        self, 
        file_path: str, 
        line: int, 
        col: int
    ) -> list[dict]:
        """Find all references to symbol at position.
        
        Args:
            file_path: Path to file (relative to project root or absolute)
            line: 0-indexed line number
            col: 0-indexed column number
            
        Returns:
            List of reference locations with file, line, col
        """
        lang = self._get_language_for_file(file_path)
        if lang not in self.clients or not self._started.get(lang):
            logger.debug(f"No {lang} server available for {file_path}")
            return []
        
        try:
            client = self.clients[lang]
            pos = Position(line=line, character=col)
            
            # Normalize file path to relative
            rel_path = self._to_relative_path(file_path)
            
            refs = await client.request_references(rel_path, pos, include_declaration=True)
            
            # Convert lsp-client locations to dict format
            return self._convert_locations(refs)
        except Exception as e:
            logger.warning(f"find_references failed: {e}")
            return []
    
    async def find_definition(
        self, 
        file_path: str, 
        line: int, 
        col: int
    ) -> Optional[dict]:
        """Find definition of symbol at position.
        
        Args:
            file_path: Path to file (relative to project root or absolute)
            line: 0-indexed line number
            col: 0-indexed column number
            
        Returns:
            Definition location with file, line, col, or None
        """
        lang = self._get_language_for_file(file_path)
        if lang not in self.clients or not self._started.get(lang):
            logger.debug(f"No {lang} server available for {file_path}")
            return None
        
        try:
            client = self.clients[lang]
            pos = Position(line=line, character=col)
            rel_path = self._to_relative_path(file_path)
            
            result = await client.request_definition(rel_path, pos)
            
            if result:
                locs = self._convert_locations([result] if not isinstance(result, list) else result)
                return locs[0] if locs else None
            return None
        except Exception as e:
            logger.warning(f"find_definition failed: {e}")
            return None
    
    def _to_relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative path from project root."""
        path = Path(file_path)
        if path.is_absolute():
            try:
                return str(path.relative_to(self.project_root))
            except ValueError:
                return file_path
        return file_path
    
    def _convert_locations(self, locations: list | None) -> list[dict]:
        """Convert lsp-client Location objects to dict format."""
        if not locations:
            return []
        
        result = []
        for loc in locations:
            try:
                # Handle lsp-client Location object
                uri = getattr(loc, 'uri', None) or loc.get('uri') if isinstance(loc, dict) else str(loc.uri)
                range_obj = getattr(loc, 'range', None) or (loc.get('range') if isinstance(loc, dict) else None)
                
                # Extract file path from URI
                file_path = uri.replace('file://', '') if uri.startswith('file://') else uri
                
                # Extract position
                if range_obj:
                    start = getattr(range_obj, 'start', None) or range_obj.get('start', {})
                    line = getattr(start, 'line', None) if hasattr(start, 'line') else start.get('line', 0)
                    col = getattr(start, 'character', None) if hasattr(start, 'character') else start.get('character', 0)
                else:
                    line = col = 0
                
                result.append({
                    'uri': uri,
                    'absolutePath': file_path,
                    'relativePath': self._to_relative_path(file_path),
                    'range': {
                        'start': {'line': line, 'character': col},
                        'end': {'line': line, 'character': col}
                    }
                })
            except Exception as e:
                logger.debug(f"Error converting location: {e}")
        
        return result
    
    def _get_language_for_file(self, file_path: str) -> Optional[str]:
        """Determine language from file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Language identifier or None if unknown
        """
        ext = Path(file_path).suffix.lower()
        if ext in (".py",):
            return "python"
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            return "typescript"
        return None
    
    async def call_hierarchy(
        self, 
        file_path: str, 
        line: int, 
        col: int,
        direction: str = "incoming"
    ) -> list[dict]:
        """Get call hierarchy for symbol at position.
        
        Args:
            file_path: Path to file (relative or absolute)
            line: 0-indexed line number
            col: 0-indexed column number
            direction: "incoming" (callers) or "outgoing" (callees)
            
        Returns:
            List of CallHierarchyItem dicts
        """
        lang = self._get_language_for_file(file_path)
        if lang not in self.clients or not self._started.get(lang):
            logger.debug(f"No {lang} server available for call hierarchy")
            return []
        
        try:
            client = self.clients[lang]
            pos = Position(line=line, character=col)
            rel_path = self._to_relative_path(file_path)
            
            # lsp-client uses prepare + incoming/outgoing pattern
            items = await client.request_prepare_call_hierarchy(rel_path, pos)
            
            if not items:
                return []
            
            result = []
            for item in items:
                if direction == "incoming":
                    calls = await client.request_incoming_calls(item)
                else:
                    calls = await client.request_outgoing_calls(item)
                
                for call in (calls or []):
                    result.append(self._convert_call_hierarchy_item(call))
            
            return result
        except Exception as e:
            logger.warning(f"call_hierarchy failed: {e}")
            return []
    
    def _convert_call_hierarchy_item(self, item: Any) -> dict:
        """Convert CallHierarchyItem to dict."""
        try:
            # Handle both incoming and outgoing call items
            call_item = getattr(item, 'from_', None) or getattr(item, 'to', None) or item
            
            return {
                'name': getattr(call_item, 'name', 'unknown'),
                'kind': getattr(call_item, 'kind', 0),
                'uri': getattr(call_item, 'uri', ''),
                'range': {
                    'start': {'line': getattr(call_item.range.start, 'line', 0) if hasattr(call_item, 'range') else 0},
                }
            }
        except Exception as e:
            logger.debug(f"Error converting call hierarchy item: {e}")
            return {}
    
    async def document_symbols(self, file_path: str) -> list[dict]:
        """Get all symbols in a document.
        
        Used by sync_project for symbol extraction.
        
        Args:
            file_path: Path to file (relative or absolute)
            
        Returns:
            List of symbol dicts with name, kind, range, children
        """
        lang = self._get_language_for_file(file_path)
        if lang not in self.clients or not self._started.get(lang):
            logger.debug(f"No {lang} server available for document_symbols")
            return []
        
        try:
            client = self.clients[lang]
            rel_path = self._to_relative_path(file_path)
            
            result = await client.request_document_symbol(rel_path)
            
            # Normalize to flat list with line info
            return self._flatten_symbols(result or [])
        except Exception as e:
            logger.warning(f"document_symbols failed: {e}")
            return []
    
    def _flatten_symbols(self, symbols: list, parent_name: str = "") -> list[dict]:
        """Flatten nested document symbols to a flat list.
        
        Args:
            symbols: LSP DocumentSymbol or SymbolInformation list
            parent_name: Parent symbol name for nested symbols
            
        Returns:
            Flat list of symbol dicts compatible with PAS format
        """
        result = []
        for sym in symbols:
            try:
                # Handle both object and dict formats
                name = getattr(sym, 'name', None) or (sym.get('name') if isinstance(sym, dict) else '')
                kind = getattr(sym, 'kind', None) or (sym.get('kind') if isinstance(sym, dict) else 0)
                
                # Get range
                range_obj = getattr(sym, 'range', None) or (sym.get('range') if isinstance(sym, dict) else None)
                if range_obj:
                    start = getattr(range_obj, 'start', None) or range_obj.get('start', {})
                    end = getattr(range_obj, 'end', None) or range_obj.get('end', {})
                    start_line = getattr(start, 'line', 0) if hasattr(start, 'line') else start.get('line', 0)
                    end_line = getattr(end, 'line', 0) if hasattr(end, 'line') else end.get('line', 0)
                else:
                    start_line = end_line = 0
                
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                result.append({
                    "name": full_name,
                    "type": self._kind_to_type(kind),
                    "line_start": start_line,
                    "line_end": end_line,
                })
                
                # Recurse into children
                children = getattr(sym, 'children', None) or (sym.get('children') if isinstance(sym, dict) else None)
                if children:
                    result.extend(self._flatten_symbols(children, full_name))
            except Exception as e:
                logger.debug(f"Error flattening symbol: {e}")
        
        return result
    
    def _kind_to_type(self, kind: int) -> str:
        """Convert LSP SymbolKind to PAS type string."""
        # LSP SymbolKind constants
        kind_map = {
            5: "class",      # Class
            6: "function",   # Method
            12: "function",  # Function
            13: "variable",  # Variable
            14: "constant",  # Constant
        }
        return kind_map.get(kind, "symbol")
    
    @property
    def available_languages(self) -> list[str]:
        """List of languages with running servers."""
        return [lang for lang, started in self._started.items() if started]
    
    async def __aenter__(self) -> "LspManager":
        """Async context manager entry."""
        await self.start_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop_servers()
