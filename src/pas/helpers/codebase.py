"""
PAS Codebase Helper Functions

Pure functions for codebase indexing, symbol extraction,
and code navigation utilities.
"""

import logging
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger("pas-server")

# =============================================================================
# Language Configuration
# =============================================================================

# Mapping of file extensions to tree-sitter language names
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",  # v53: C# support
}

# Extensions to skip during indexing
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".lib",
    ".dll", ".exe", ".bin", ".dat",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp3", ".mp4", ".wav", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".rar",
    ".pdf", ".doc", ".docx",
}

# Directories to skip during indexing
SKIP_DIRS = {
    "__pycache__", ".git", ".svn", ".hg",
    "node_modules", "venv", ".venv", "env",
    "dist", "build", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache",
    ".tox", "eggs", "*.egg-info",
}


# =============================================================================
# Symbol Extraction
# =============================================================================

# LSP SymbolKind to our type mapping
LSP_KIND_MAP = {
    "Class": "class",
    "Function": "function",
    "Method": "method",
    "Variable": "variable",
    "Constant": "constant",
    "Module": "module",
    "Property": "property",
    "Field": "field",
    "Constructor": "method",
    "Interface": "class",
    "Enum": "class",
    "EnumMember": "constant",
}


async def extract_symbols_lsp(file_path: str, lsp_pool) -> list[dict]:
    """
    Extract symbols using LSP (accurate).
    
    Uses LSP document_symbols for authoritative symbol data.
    Falls back gracefully if LSP is unavailable.
    
    Args:
        file_path: Path to the file (absolute or relative)
        lsp_pool: LspPool instance
        
    Returns:
        List of dicts with: name, type, line_start, line_end, signature
    """
    if not lsp_pool:
        return []
    
    try:
        raw_symbols = await lsp_pool.document_symbols(file_path)
        
        if not raw_symbols:
            return []
        
        result = []
        for sym in raw_symbols:
            kind = sym.get("kind", "")
            symbol_type = LSP_KIND_MAP.get(kind, "other")
            
            # Skip variables/constants for now (too noisy)
            if symbol_type in ("variable", "constant", "field"):
                continue
            
            result.append({
                "name": sym.get("name", ""),
                "type": symbol_type,
                "line_start": sym.get("line", 0) + 1,  # LSP is 0-indexed
                "line_end": sym.get("end_line", sym.get("line", 0)) + 1,
                "signature": sym.get("detail", ""),
                "docstring": "",  # LSP doesn't provide docstrings
            })
        
        return result
    except Exception as e:
        logger.debug(f"LSP symbol extraction failed: {e}")
        return []


def extract_symbols(content: str, language: str) -> list[dict]:
    """
    Extract function/class symbols from source code using tree-sitter.
    
    Args:
        content: Source code content
        language: tree-sitter language name (e.g., 'python', 'javascript')
        
    Returns:
        List of symbol dicts with type, name, line_start, line_end, signature
    """
    try:
        # v37b: Package renamed from tree_sitter_languages to tree_sitter_language_pack
        import tree_sitter_language_pack as ts_pack
        from tree_sitter import Parser
    except ImportError:
        logger.warning("tree-sitter-language-pack not installed, skipping symbol extraction")
        return []
    
    try:
        parser = Parser(ts_pack.get_language(language))
        tree = parser.parse(content.encode())
        
        symbols = []
        
        # Walk the tree looking for function/class definitions
        def walk_node(node, parent_name=None):
            node_type = node.type
            
            # Python-specific
            if node_type == 'function_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    sym = {
                        'type': 'function',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0],
                    }
                    # Extract docstring if present
                    if node.child_count > 0:
                        for child in node.children:
                            if child.type == 'expression_statement':
                                expr = child.child(0)
                                if expr and expr.type == 'string':
                                    sym['docstring'] = expr.text.decode().strip('"""\'\'\'')
                                    break
                    symbols.append(sym)
            
            elif node_type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node:
                    sym = {
                        'type': 'class',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0],
                    }
                    symbols.append(sym)
            
            # JavaScript/TypeScript function
            elif node_type in ('function_declaration', 'method_definition', 'arrow_function'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols.append({
                        'type': 'function',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0][:200],
                    })
            
            # v53: C# method and class extraction
            elif node_type == 'method_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols.append({
                        'type': 'method',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0][:300],
                    })
            
            elif node_type == 'class_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    symbols.append({
                        'type': 'class',
                        'name': name_node.text.decode(),
                        'line_start': node.start_point[0] + 1,
                        'line_end': node.end_point[0] + 1,
                        'signature': content[node.start_byte:node.end_byte].split('\n')[0][:200],
                    })
            
            # Recurse children
            for child in node.children:
                walk_node(child, parent_name)
        
        walk_node(tree.root_node)
        return symbols
        
    except Exception as e:
        logger.warning(f"Symbol extraction failed for {language}: {e}")
        return []


def get_language_from_path(file_path: str) -> Optional[str]:
    """
    Determine programming language from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language name for tree-sitter, or None if unsupported
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    return LANGUAGE_MAP.get(ext)


def should_skip_file(file_path: str, max_size_kb: int = 100) -> tuple[bool, str]:
    """
    Check if a file should be skipped during indexing.
    
    Args:
        file_path: Path to the file
        max_size_kb: Maximum file size in KB
        
    Returns:
        Tuple of (should_skip, reason)
    """
    path = Path(file_path)
    
    # Check extension
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True, "binary_extension"
    
    # Check directory
    for part in path.parts:
        if part in SKIP_DIRS or part.endswith('.egg-info'):
            return True, "skip_directory"
    
    # Check size
    try:
        size_kb = path.stat().st_size / 1024
        if size_kb > max_size_kb:
            return True, f"too_large_{size_kb:.0f}kb"
    except (OSError, IOError):
        return True, "unreadable"
    
    return False, ""


def compute_file_hash(content: str) -> str:
    """
    Compute a hash for file content to detect changes.
    
    Args:
        content: File content string
        
    Returns:
        MD5 hash hex string
    """
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()


def derive_project_id(project_path: str) -> str:
    """
    Derive a project ID from the project path.
    
    Args:
        project_path: Absolute path to project root
        
    Returns:
        Project ID string (folder name, lowercase, sanitized)
    """
    path = Path(project_path)
    return path.name.lower().replace(" ", "-").replace("_", "-")


# =============================================================================
# Symbol Pattern Extraction
# =============================================================================

def extract_symbol_patterns_from_text(text: str) -> list[str]:
    """
    Extract potential symbol names from natural language text.
    
    Looks for snake_case, CamelCase, and other code-like patterns.
    
    Args:
        text: Natural language text (goal, hypothesis, etc.)
        
    Returns:
        List of potential symbol names
    """
    import re
    
    patterns = []
    
    # snake_case: word_word
    snake_matches = re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', text)
    patterns.extend(snake_matches)
    
    # CamelCase: WordWord
    camel_matches = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    patterns.extend(camel_matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    
    return unique


def build_reference_summary(references: list[dict]) -> dict:
    """
    Build a summary of references for a symbol.
    
    Args:
        references: List of reference dicts from find_references
        
    Returns:
        Summary dict with counts and file distribution
    """
    total = len(references)
    files = set()
    types = {}
    
    for ref in references:
        files.add(ref.get("file_path", ref.get("file", "")))
        ref_type = ref.get("reference_type", ref.get("relation", "unknown"))
        types[ref_type] = types.get(ref_type, 0) + 1
    
    return {
        "total_references": total,
        "unique_files": len(files),
        "reference_types": types,
        "files": list(files)[:10]  # Limit for display
    }


# =============================================================================
# Phase 6: find_references Helpers
# =============================================================================

# v51 Phase 2: Module-level ripgrep detection (cached at import time)
import shutil
HAS_RIPGREP = shutil.which('rg') is not None


def resolve_project_root(rel_paths: list[str]) -> Optional[Path]:
    """
    Resolve project root from relative file paths using heuristics.
    
    Args:
        rel_paths: List of relative file paths from file_registry
        
    Returns:
        Path to project root, or None if cannot be resolved
    """
    if not rel_paths:
        return None
    
    # Try to find common parent directory
    for path_str in rel_paths[:5]:  # Check first 5 paths
        path = Path(path_str)
        if path.is_absolute():
            # Find parent with common project markers
            for parent in path.parents:
                if (parent / "pyproject.toml").exists() or \
                   (parent / "setup.py").exists() or \
                   (parent / ".git").exists():
                    return parent
    return None


def fetch_project_root(project_id: str, cur) -> Optional[Path]:
    """
    Fetch project_root from project_registry.
    
    Args:
        project_id: Project identifier
        cur: Database cursor
        
    Returns:
        Path to project root, or None if not found
    """
    cur.execute(
        "SELECT project_root FROM project_registry WHERE project_id = %s",
        (project_id,)
    )
    row = cur.fetchone()
    if row and row.get('project_root'):
        return Path(row['project_root'])
    return None


def prefilter_files(
    symbol: str,
    project_root: Optional[Path],
    file_paths: list[Path],
    use_rg: Optional[bool] = None
) -> list[Path]:
    """
    Unified pre-filter interface for find_references.
    
    v51 Phase 2: Reduces O(n) scanning to O(k) by pre-filtering.
    
    Args:
        symbol: Symbol name to search for
        project_root: Project root path
        file_paths: List of file paths
        use_rg: Force ripgrep (True), Python (False), or auto (None)
        
    Returns:
        List of candidate file paths containing the symbol
    """
    import subprocess
    import re
    
    should_use_rg = use_rg if use_rg is not None else (HAS_RIPGREP and project_root is not None)
    
    if should_use_rg and project_root:
        try:
            escaped = re.escape(symbol)
            result = subprocess.run(
                ['rg', '-lw', '--type', 'py', escaped, str(project_root)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [Path(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]
        except Exception:
            pass
    
    # Python fallback
    pattern = re.compile(r'(?<![a-zA-Z0-9_])' + re.escape(symbol) + r'(?![a-zA-Z0-9_])')
    candidates = []
    for path in file_paths:
        try:
            if path.exists():
                content = path.read_text(encoding='utf-8', errors='replace')
                if pattern.search(content):
                    candidates.append(path)
        except Exception:
            pass
    return candidates


def prefilter_rg(symbol: str, project_root: Path) -> list[Path]:
    """Pre-filter files using ripgrep word-boundary matching."""
    import subprocess
    import re
    
    try:
        escaped = re.escape(symbol)
        result = subprocess.run(
            ['rg', '-lw', '--type', 'py', escaped, str(project_root)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return [Path(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]
        return []
    except Exception:
        return []


def prefilter_python(symbol: str, file_paths: list[Path]) -> list[Path]:
    """Pre-filter files using Python regex with identifier-aware boundaries."""
    import re
    
    pattern = re.compile(r'(?<![a-zA-Z0-9_])' + re.escape(symbol) + r'(?![a-zA-Z0-9_])')
    candidates = []
    for path in file_paths:
        try:
            if path.exists():
                content = path.read_text(encoding='utf-8', errors='replace')
                if pattern.search(content):
                    candidates.append(path)
        except Exception:
            pass
    return candidates

# NOTE: Jedi fallback functions removed (v53 DB-first architecture)
# scan_file_for_references and find_references_jedi were here
# Now queries use DB cache populated by sync_project




def deduplicate_references(references: list[dict], include_definitions: bool) -> list[dict]:
    """Remove duplicate references and optionally filter definitions."""
    seen = set()
    unique = []
    for ref in references:
        key = (ref.get('file'), ref.get('line'), ref.get('symbol'))
        if key not in seen:
            seen.add(key)
            if include_definitions or ref.get('relation') != 'definition':
                unique.append(ref)
    return unique


# =============================================================================
# Incremental Sync (for auto-sync watcher)
# =============================================================================

async def sync_file_incremental(
    file_path: str,
    project_id: str,
    project_root: str,
    lsp_pool=None,
    include_references: bool = False,
    include_call_hierarchy: bool = False
) -> dict:
    """
    Sync a single file to the database.
    
    Used by auto-sync watcher for real-time updates.
    LSP-first symbol extraction with tree-sitter fallback.
    
    Args:
        file_path: Absolute path to the file
        project_id: Project identifier
        project_root: Project root directory
        lsp_pool: Optional LspPool instance for LSP extraction
        include_references: If True, index symbol references via LSP
        include_call_hierarchy: If True, index call hierarchy via LSP
        
    Returns:
        Dict with success status, symbol count, and reference count
    """
    from pas.utils import get_embedding, get_db_connection
    
    path = Path(file_path)
    
    # Phase 11: Handle file deletions
    if not path.exists():
        # File was deleted - remove from DB
        try:
            # Compute relative path for DB lookup
            try:
                rel_path = str(path.relative_to(project_root))
            except ValueError:
                rel_path = str(path)
            
            conn = get_db_connection()
            try:
                cur = conn.cursor()
                
                # Delete file_symbols first (FK constraint)
                cur.execute("""
                    DELETE FROM file_symbols 
                    WHERE file_id IN (
                        SELECT id FROM file_registry 
                        WHERE project_id = %s AND file_path = %s
                    )
                """, (project_id, rel_path))
                
                # Delete symbol_references for this file
                cur.execute("""
                    DELETE FROM symbol_references 
                    WHERE project_id = %s AND (source_file = %s OR target_file = %s)
                """, (project_id, rel_path, rel_path))
                
                # Delete file_registry entry
                cur.execute("""
                    DELETE FROM file_registry 
                    WHERE project_id = %s AND file_path = %s
                    RETURNING id
                """, (project_id, rel_path))
                deleted = cur.fetchone()
                
                conn.commit()
                
                if deleted:
                    logger.info(f"Phase 11: Deleted orphan file {rel_path} from DB")
                    return {"success": True, "action": "deleted", "file": rel_path}
                else:
                    return {"success": True, "action": "not_found", "file": rel_path}
                    
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"sync_file_incremental delete error: {e}")
            return {"success": False, "error": str(e)}
    
    # Get language
    language = get_language_from_path(file_path)
    if not language:
        return {"success": False, "error": "Unsupported file type"}
    
    try:
        # Read content
        content = path.read_text(encoding='utf-8', errors='replace')
        file_hash = compute_file_hash(content)
        mtime_ns = path.stat().st_mtime_ns
        
        # Relative path for storage
        try:
            rel_path = str(path.relative_to(project_root))
        except ValueError:
            rel_path = str(path)
        
        # Extract symbols (LSP-first)
        symbols = []
        lsp_used = False
        # v53: Enable LSP extraction for Python and C#
        if lsp_pool and language in ("python", "csharp"):
            try:
                symbols = await extract_symbols_lsp(file_path, lsp_pool)
                if symbols:
                    lsp_used = True
            except Exception as e:
                logger.debug(f"LSP symbol extraction failed: {e}")
        
        if not symbols:
            symbols = extract_symbols(content, language)
        
        # Generate content embedding
        content_embedding = get_embedding(content[:2000])
        
        # Update database
        conn = get_db_connection()
        references_indexed = 0
        calls_indexed = 0
        
        try:
            cur = conn.cursor()
            
            # Upsert file_registry
            cur.execute(
                """
                INSERT INTO file_registry (project_id, file_path, file_hash, language, content_embedding, mtime_ns)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id, file_path) 
                DO UPDATE SET file_hash = EXCLUDED.file_hash, 
                              language = EXCLUDED.language,
                              content_embedding = EXCLUDED.content_embedding,
                              mtime_ns = EXCLUDED.mtime_ns
                RETURNING id
                """,
                (project_id, rel_path, file_hash, language, content_embedding, mtime_ns)
            )
            file_id = cur.fetchone()['id']
            
            # Clear old symbols
            cur.execute("DELETE FROM file_symbols WHERE file_id = %s", (file_id,))
            
            # Clear old references for this file if re-indexing
            if include_references or include_call_hierarchy:
                cur.execute(
                    "DELETE FROM symbol_references WHERE source_file = %s",
                    (rel_path,)
                )
            
            # Insert new symbols
            for sym in symbols:
                embed_text = sym.get('signature', '') + '\n' + sym.get('docstring', '')
                sym_embedding = get_embedding(embed_text[:500])
                
                cur.execute(
                    """
                    INSERT INTO file_symbols (file_id, symbol_type, symbol_name, line_start, line_end, signature, docstring, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (file_id, sym['type'], sym['name'], sym.get('line_start'), 
                     sym.get('line_end'), sym.get('signature'), sym.get('docstring'), sym_embedding)
                )
            
            # Index references via LSP if requested
            # v53: Enable for Python and C#
            if include_references and lsp_pool and language in ("python", "csharp"):
                for sym in symbols:
                    try:
                        line = sym.get('line_start', 1) - 1  # LSP is 0-indexed
                        refs = await lsp_pool.find_references(file_path, line, 0)
                        for ref in refs:
                            target_uri = ref.get('uri', '')
                            if target_uri.startswith('file://'):
                                target_file = target_uri[7:]  # Remove file:// prefix
                                try:
                                    target_rel = str(Path(target_file).relative_to(project_root))
                                except ValueError:
                                    target_rel = target_file
                                
                                cur.execute(
                                    """
                                    INSERT INTO symbol_references 
                                    (source_file, source_symbol, target_file, target_line, direction, project_id)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT DO NOTHING
                                    """,
                                    (rel_path, sym['name'], target_rel, ref.get('line', 0), 'reference', project_id)
                                )
                                references_indexed += 1
                    except Exception as e:
                        logger.debug(f"Reference indexing failed for {sym['name']}: {e}")
            
            # Index call hierarchy via LSP if requested
            # v53: Enable for Python and C#
            if include_call_hierarchy and lsp_pool and language in ("python", "csharp"):
                for sym in symbols:
                    if sym.get('type') in ('function', 'method'):
                        try:
                            line = sym.get('line_start', 1) - 1  # LSP is 0-indexed
                            calls = await lsp_pool.call_hierarchy(file_path, line, 0, 'incoming')
                            for call in calls:
                                caller_uri = call.get('uri', '')
                                if caller_uri.startswith('file://'):
                                    caller_file = caller_uri[7:]
                                    try:
                                        caller_rel = str(Path(caller_file).relative_to(project_root))
                                    except ValueError:
                                        caller_rel = caller_file
                                    
                                    cur.execute(
                                        """
                                        INSERT INTO symbol_references 
                                        (source_file, source_symbol, target_file, target_line, direction, project_id)
                                        VALUES (%s, %s, %s, %s, %s, %s)
                                        ON CONFLICT DO NOTHING
                                        """,
                                        (caller_rel, call.get('name', ''), rel_path, line, 'incoming', project_id)
                                    )
                                    calls_indexed += 1
                        except Exception as e:
                            logger.debug(f"Call hierarchy indexing failed for {sym['name']}: {e}")
            
            conn.commit()
            
            return {
                "success": True, 
                "file": rel_path,
                "symbols": len(symbols),
                "lsp_used": lsp_used,
                "references_indexed": references_indexed,
                "calls_indexed": calls_indexed
            }
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"sync_file_incremental error: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# v61: Deep Project Understanding
# =============================================================================

async def populate_project_understanding(
    project_id: str,
    project_path: str,
    conn: Any
) -> dict[str, Any]:
    """
    Populate deep understanding data for a project.
    
    Calls system_map, schema_intent, and config_assumptions extraction
    and stores results in project_registry.meta JSONB field.
    
    Args:
        project_id: Project identifier
        project_path: Absolute path to project root
        conn: Database connection
        
    Returns:
        Dict with results from each extraction
    """
    import json
    from datetime import datetime
    from pas.helpers.self_awareness import get_schema_info
    from pas.helpers.schema_intent import extract_schema_entities, build_enrichment_prompt
    from pas.helpers.config_assumptions import (
        parse_config_file, extract_assumptions, 
        build_enrichment_prompt as build_config_prompt
    )
    
    results = {
        'system_map': None,
        'schema_intent': None,
        'config_assumptions': None,
        'errors': []
    }
    
    cur = conn.cursor()
    
    # 1. System Map (from symbol references in DB)
    try:
        # Check if symbol_references table has data for this project
        cur.execute("""
            SELECT COUNT(*) as cnt FROM symbol_references WHERE project_id = %s
        """, (project_id,))
        ref_count = cur.fetchone()['cnt']
        
        if ref_count > 0:
            # Aggregate cross-file references into module dependencies
            cur.execute("""
                SELECT 
                    COALESCE(NULLIF(regexp_replace(source_file, '/[^/]+$', ''), ''), 'root') as source_module,
                    COALESCE(NULLIF(regexp_replace(target_file, '/[^/]+$', ''), ''), 'root') as target_module,
                    COUNT(*) as weight
                FROM symbol_references
                WHERE project_id = %s
                GROUP BY source_module, target_module
                HAVING COALESCE(NULLIF(regexp_replace(source_file, '/[^/]+$', ''), ''), 'root') != 
                       COALESCE(NULLIF(regexp_replace(target_file, '/[^/]+$', ''), ''), 'root')
            """, (project_id,))
            
            edges = []
            nodes = set()
            for row in cur.fetchall():
                src, tgt, weight = row['source_module'], row['target_module'], row['weight']
                edges.append({'source': src, 'target': tgt, 'weight': weight})
                nodes.add(src)
                nodes.add(tgt)
            
            results['system_map'] = {
                'nodes': list(nodes),
                'edges': edges,
                'stats': {'module_count': len(nodes), 'edge_count': len(edges)}
            }
        else:
            # Fallback: derive from file_registry directories
            cur.execute("""
                SELECT DISTINCT COALESCE(NULLIF(regexp_replace(file_path, '/[^/]+$', ''), ''), 'root') as module
                FROM file_registry WHERE project_id = %s
            """, (project_id,))
            modules = [row['module'] for row in cur.fetchall()]
            results['system_map'] = {
                'nodes': modules,
                'edges': [],
                'stats': {'module_count': len(modules), 'edge_count': 0, 'note': 'No symbol references indexed'}
            }
            
    except Exception as e:
        logger.warning(f"populate_project_understanding: system_map failed: {e}")
        results['errors'].append(f"system_map: {str(e)}")
    
    # 2. Schema Intent (from information_schema)
    try:
        schema_info = get_schema_info(conn)
        if schema_info.get('tables'):
            entities = extract_schema_entities(
                schema_info['tables'],
                schema_info.get('relationships', [])
            )
            entities['enrichment_prompt'] = build_enrichment_prompt(
                entities['entities'],
                entities['relationships']
            )
            results['schema_intent'] = entities
    except Exception as e:
        logger.warning(f"populate_project_understanding: schema_intent failed: {e}")
        results['errors'].append(f"schema_intent: {str(e)}")
    
    # 3. Config Assumptions (from config.yaml if exists)
    try:
        # v61 fix: Search common config locations, not just project root
        config_candidates = ['config.yaml', 'config.yml', 'config.json']
        search_dirs = [
            Path(project_path),  # Project root
            Path(project_path) / 'config',  # config/
            Path(project_path) / 'src' / 'config',  # src/config/
        ]
        # Also check src/<pkg>/config/ pattern
        src_dir = Path(project_path) / 'src'
        if src_dir.exists():
            for child in src_dir.iterdir():
                if child.is_dir() and not child.name.startswith('.'):
                    config_subdir = child / 'config'
                    if config_subdir.exists():
                        search_dirs.append(config_subdir)
        
        config_path = None
        for search_dir in search_dirs:
            for candidate in config_candidates:
                candidate_path = search_dir / candidate
                if candidate_path.exists():
                    config_path = candidate_path
                    break
            if config_path:
                break
        
        if config_path:
            config_data = parse_config_file(str(config_path))
            assumptions = extract_assumptions(config_data)
            results['config_assumptions'] = {
                'config_file': str(config_path.relative_to(project_path)),
                'assumptions': assumptions,
                'enrichment_prompt': build_config_prompt(assumptions, str(config_path))
            }
        else:
            results['config_assumptions'] = {
                'config_file': None,
                'assumptions': [],
                'note': 'No config.yaml/yml/json found in project root or common subdirs'
            }
    except Exception as e:
        logger.warning(f"populate_project_understanding: config_assumptions failed: {e}")
        results['errors'].append(f"config_assumptions: {str(e)}")
    
    # Store in project_registry (existing columns: detected_entities, config_assumptions)
    try:
        # Note: system_map is computed live by get_system_map() from symbol_references
        # Only need to store schema_intent and config_assumptions
        cur.execute("""
            UPDATE project_registry 
            SET detected_entities = %s,
                config_assumptions = %s,
                updated_at = NOW()
            WHERE project_id = %s
        """, (
            json.dumps(results['schema_intent']) if results['schema_intent'] else None,
            json.dumps(results['config_assumptions']) if results['config_assumptions'] else None,
            project_id
        ))
        conn.commit()
        results['stored'] = True
    except Exception as e:
        logger.warning(f"populate_project_understanding: storage failed: {e}")
        results['errors'].append(f"storage: {str(e)}")
        results['stored'] = False
    
    return results

