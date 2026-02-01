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


def scan_file_for_references(file_path: Path, symbol_name: str) -> list[dict]:
    """Scan a single file for symbol references using Jedi."""
    try:
        import jedi
    except ImportError:
        return []
    
    references = []
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        script = jedi.Script(source, path=file_path)
        for i, line in enumerate(source.splitlines(), 1):
            if symbol_name in line:
                col = line.find(symbol_name)
                if col >= 0:
                    try:
                        refs = script.get_references(line=i, column=col, scope='file')
                        for ref in refs:
                            if ref.name == symbol_name:
                                references.append({
                                    "file": str(file_path),
                                    "line": ref.line,
                                    "symbol": ref.name,
                                    "relation": "definition" if ref.is_definition() else "reference"
                                })
                        break
                    except Exception:
                        continue
    except Exception:
        pass
    return references


def find_references_jedi(
    project_root: Optional[Path],
    rel_paths: list[str],
    symbol_name: str
) -> list[dict]:
    """Find references across project using Jedi live analysis."""
    all_refs = []
    for rel_path in rel_paths:
        file_path = project_root / rel_path if project_root else Path(rel_path)
        if file_path.exists():
            all_refs.extend(scan_file_for_references(file_path, symbol_name))
    return all_refs



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
    lsp_pool=None
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
        
    Returns:
        Dict with success status and symbol count
    """
    from pas.utils import get_embedding, get_db_connection
    
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": "File not found"}
    
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
        if lsp_pool and language == "python":
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
        try:
            cur = conn.cursor()
            
            # Upsert file_registry
            cur.execute(
                """
                INSERT INTO file_registry (project_id, file_path, file_hash, language, embedding, mtime_ns)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id, file_path) 
                DO UPDATE SET file_hash = EXCLUDED.file_hash, 
                              language = EXCLUDED.language,
                              embedding = EXCLUDED.embedding,
                              mtime_ns = EXCLUDED.mtime_ns
                RETURNING id
                """,
                (project_id, rel_path, file_hash, language, content_embedding, mtime_ns)
            )
            file_id = cur.fetchone()['id']
            
            # Clear old symbols
            cur.execute("DELETE FROM file_symbols WHERE file_id = %s", (file_id,))
            
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
            
            conn.commit()
            
            return {
                "success": True, 
                "file": rel_path,
                "symbols": len(symbols),
                "lsp_used": lsp_used
            }
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"sync_file_incremental error: {e}")
        return {"success": False, "error": str(e)}
