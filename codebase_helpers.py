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
        files.add(ref.get("file_path", ""))
        ref_type = ref.get("reference_type", "unknown")
        types[ref_type] = types.get(ref_type, 0) + 1
    
    return {
        "total_references": total,
        "unique_files": len(files),
        "reference_types": types,
        "files": list(files)[:10]  # Limit for display
    }
