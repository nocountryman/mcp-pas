"""LSP subprocess worker - runs lsp-client in isolated process.

v52c: Solves anyio task group conflict with MCP by running LSP in subprocess.
The main process communicates via multiprocessing.Queue.
"""

import asyncio
import json
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# CRITICAL: Use spawn instead of fork to avoid deadlocks with anyio/threading
# fork inherits parent's locks which causes hangs in MCP subprocess context
mp_ctx = mp.get_context('spawn')

# Subprocess worker function - runs in separate process
def _lsp_worker(request_queue: mp.Queue, response_queue: mp.Queue, project_root: str):
    """Worker process that handles LSP requests."""
    import asyncio
    
    async def run_worker():
        from lsp_client import BasedpyrightClient, LocalServer, Position
        
        # Start client once
        server = LocalServer(
            program="basedpyright-langserver",
            args=["--stdio"],
            cwd=Path(project_root)
        )
        
        try:
            async with BasedpyrightClient(
                server=server,
                workspace=project_root
            ) as client:
                logger.info(f"LSP worker connected, waiting for analysis...")
                # Wait for LSP to analyze project files
                await asyncio.sleep(3)
                logger.info(f"LSP worker ready for {project_root}")
                response_queue.put({"status": "ready"})
                
                while True:
                    # Check for requests (non-blocking)
                    try:
                        request = request_queue.get(timeout=0.1)
                    except:
                        continue
                    
                    if request.get("command") == "shutdown":
                        break
                    
                    try:
                        result = await handle_request(client, request, project_root)
                        response_queue.put({"id": request.get("id"), "result": result})
                    except Exception as e:
                        response_queue.put({"id": request.get("id"), "error": str(e)})
        except Exception as e:
            response_queue.put({"status": "error", "error": str(e)})
    
    asyncio.run(run_worker())


async def handle_request(client: Any, request: dict, project_root: str) -> Any:
    """Handle a single LSP request."""
    from lsp_client import Position
    
    cmd = request.get("command")
    
    if cmd == "find_references":
        file_path = request["file_path"]
        line = request["line"]
        col = request["col"]
        
        # Make path relative if absolute
        path = Path(file_path)
        if path.is_absolute():
            try:
                file_path = str(path.relative_to(project_root))
            except ValueError:
                pass
        
        pos = Position(line=line, character=col)
        refs = await client.request_references(file_path, pos, include_declaration=True)
        
        # Convert to serializable format
        result = []
        for ref in (refs or []):
            try:
                uri = getattr(ref, 'uri', '') or ''
                range_obj = getattr(ref, 'range', None)
                if range_obj:
                    start = getattr(range_obj, 'start', None)
                    line_num = getattr(start, 'line', 0) if start else 0
                    col_num = getattr(start, 'character', 0) if start else 0
                else:
                    line_num = col_num = 0
                
                result.append({
                    "uri": uri,
                    "line": line_num,
                    "col": col_num
                })
            except:
                pass
        
        return result
    
    elif cmd == "find_definition":
        file_path = request["file_path"]
        line = request["line"]
        col = request["col"]
        
        path = Path(file_path)
        if path.is_absolute():
            try:
                file_path = str(path.relative_to(project_root))
            except ValueError:
                pass
        
        pos = Position(line=line, character=col)
        result = await client.request_definition(file_path, pos)
        
        if result:
            loc = result[0] if isinstance(result, list) else result
            return {
                "uri": getattr(loc, 'uri', ''),
                "line": getattr(getattr(loc, 'range', None), 'start', None).line if hasattr(loc, 'range') else 0
            }
        return None
    
    elif cmd == "document_symbols":
        file_path = request["file_path"]
        
        # Make path relative if absolute
        path = Path(file_path)
        if path.is_absolute():
            try:
                file_path = str(path.relative_to(project_root))
            except ValueError:
                pass
        
        # Get document symbols from LSP
        result = await client.request_document_symbol(file_path)
        
        # Flatten and convert to serializable format
        symbols = []
        
        def flatten_symbols(items, parent_name=None):
            """Recursively flatten nested symbols."""
            for item in (items or []):
                name = getattr(item, 'name', '') or ''
                kind = getattr(item, 'kind', None)
                # Convert SymbolKind enum to string
                kind_str = kind.name if hasattr(kind, 'name') else str(kind)
                
                # Get range info
                range_obj = getattr(item, 'range', None) or getattr(item, 'location', {}).get('range')
                if range_obj:
                    start_line = getattr(getattr(range_obj, 'start', None), 'line', 0)
                    end_line = getattr(getattr(range_obj, 'end', None), 'line', start_line)
                else:
                    start_line = end_line = 0
                
                detail = getattr(item, 'detail', '') or ''
                
                symbols.append({
                    "name": name,
                    "kind": kind_str,
                    "line": start_line,
                    "end_line": end_line,
                    "detail": detail,
                    "parent": parent_name
                })
                
                # Process children if any
                children = getattr(item, 'children', None)
                if children:
                    flatten_symbols(children, name)
        
        flatten_symbols(result)
        return symbols
    
    return None


class LspSubprocess:
    """Manages LSP subprocess for isolated execution."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self._process: Optional[mp.Process] = None
        self._request_queue: Optional[mp.Queue] = None
        self._response_queue: Optional[mp.Queue] = None
        self._request_id = 0
        self._started = False
    
    async def start(self) -> bool:
        """Start the LSP subprocess."""
        if self._started:
            return True
        
        try:
            self._request_queue = mp_ctx.Queue()
            self._response_queue = mp_ctx.Queue()
            
            self._process = mp_ctx.Process(
                target=_lsp_worker,
                args=(self._request_queue, self._response_queue, str(self.project_root)),
                daemon=True
            )
            self._process.start()
            
            # Wait for ready signal
            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._response_queue.get, True, 30
                    ),
                    timeout=30
                )
                if response.get("status") == "ready":
                    self._started = True
                    logger.info(f"LSP subprocess ready for {self.project_root}")
                    return True
                elif response.get("status") == "error":
                    logger.error(f"LSP subprocess error: {response.get('error')}")
                    return False
            except asyncio.TimeoutError:
                logger.error("LSP subprocess startup timeout")
                self.stop()
                return False
        except Exception as e:
            logger.error(f"Failed to start LSP subprocess: {e}")
            return False
        
        return False
    
    def stop(self):
        """Stop the LSP subprocess."""
        if self._request_queue:
            try:
                self._request_queue.put({"command": "shutdown"})
            except:
                pass
        if self._process:
            self._process.terminate()
            self._process.join(timeout=2)
        self._started = False
    
    async def find_references(self, file_path: str, line: int, col: int) -> list[dict]:
        """Find references via subprocess."""
        if not self._started:
            if not await self.start():
                return []
        
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "find_references",
            "file_path": file_path,
            "line": line,
            "col": col
        }
        
        try:
            self._request_queue.put(request)
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._response_queue.get, True, 30
                ),
                timeout=30
            )
            if response.get("id") == self._request_id:
                return response.get("result", [])
            return []
        except asyncio.TimeoutError:
            logger.warning("LSP find_references timeout")
            return []
        except Exception as e:
            logger.warning(f"LSP find_references error: {e}")
            return []
    
    async def find_definition(self, file_path: str, line: int, col: int) -> Optional[dict]:
        """Find definition via subprocess."""
        if not self._started:
            if not await self.start():
                return None
        
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "find_definition",
            "file_path": file_path,
            "line": line,
            "col": col
        }
        
        try:
            self._request_queue.put(request)
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._response_queue.get, True, 30
                ),
                timeout=30
            )
            if response.get("id") == self._request_id:
                return response.get("result")
            return None
        except asyncio.TimeoutError:
            logger.warning("LSP find_definition timeout")
            return None
        except Exception as e:
            logger.warning(f"LSP find_definition error: {e}")
            return None
    
    async def document_symbols(self, file_path: str) -> list[dict]:
        """Get all symbols in a document via subprocess."""
        if not self._started:
            if not await self.start():
                return []
        
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "document_symbols",
            "file_path": file_path
        }
        
        try:
            self._request_queue.put(request)
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._response_queue.get, True, 10
                ),
                timeout=10
            )
            if response.get("id") == self._request_id:
                return response.get("result", [])
            return []
        except asyncio.TimeoutError:
            logger.warning("LSP document_symbols timeout")
            return []
        except Exception as e:
            logger.warning(f"LSP document_symbols error: {e}")
            return []


# Module-level singleton
_subprocess: Optional[LspSubprocess] = None


async def get_lsp_subprocess(project_root: str) -> LspSubprocess:
    """Get or create the LSP subprocess singleton."""
    global _subprocess
    
    resolved = Path(project_root).resolve()
    
    if _subprocess is not None and _subprocess.project_root == resolved and _subprocess._started:
        return _subprocess
    
    if _subprocess is not None:
        _subprocess.stop()
    
    _subprocess = LspSubprocess(str(resolved))
    await _subprocess.start()
    return _subprocess
