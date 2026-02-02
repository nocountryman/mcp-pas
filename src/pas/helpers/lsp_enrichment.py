"""
LSP Enrichment Helpers

Pure functions for gathering LSP impact data to inform implementation plans.
Part of v52 Phase 2: LSP enrichment for PAS workflow.
"""

import logging
import re
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger("pas-server")


def scope_to_file_paths(declared_scope: str, project_root: Optional[str] = None) -> list[str]:
    """
    Parse declared_scope string into a list of file paths.
    
    Handles common scope formats:
    - "file.py, other.py" (comma-separated)
    - "module/file.py" (relative paths)
    - "CODE: file.py, DATA: schema.sql" (layer prefixes)
    - "file.py::function" (symbol suffixes)
    
    Args:
        declared_scope: Comma-separated scope string from hypothesis
        project_root: Optional project root to resolve relative paths
        
    Returns:
        List of normalized file paths
    """
    if not declared_scope:
        return []
    
    file_paths = []
    
    # Split by comma
    parts = [p.strip() for p in declared_scope.split(",")]
    
    for part in parts:
        if not part:
            continue
            
        # Remove layer prefixes like "CODE:", "DATA:", "UI:"
        part = re.sub(r"^[A-Z]+:\s*", "", part)
        
        # Remove symbol suffixes like "::FunctionName"
        if "::" in part:
            part = part.split("::")[0]
        
        # Skip non-file entries (e.g., "migration", "tests")
        if "." not in part and "/" not in part:
            continue
        
        # Resolve path
        path = Path(part)
        if project_root and not path.is_absolute():
            path = Path(project_root) / path
        
        # Only add if it looks like a file
        if path.suffix or "/" in part:
            file_paths.append(str(path))
    
    return file_paths


async def get_lsp_impact_from_scope(
    declared_scope: str,
    project_root: Optional[str] = None,
    lsp_pool: Any = None,
) -> dict:
    """
    Convenience wrapper: parse scope and get LSP impact in one call.
    
    Args:
        declared_scope: Comma-separated scope string from hypothesis
        project_root: Optional project root for path resolution
        lsp_pool: LspPool instance (or None for graceful fallback)
        
    Returns:
        LSP impact dict (see get_lsp_impact)
    """
    file_paths = scope_to_file_paths(declared_scope, project_root)
    
    if not file_paths:
        return {
            "symbols_by_file": {},
            "affected_files": {},
            "callers_outside_scope": [],
            "lsp_available": False,
            "error": "No file paths found in scope",
            "scope_parsed": declared_scope,
        }
    
    impact = await get_lsp_impact(file_paths, lsp_pool)
    impact["scope_parsed"] = declared_scope
    impact["file_paths_extracted"] = file_paths
    return impact


async def get_lsp_impact(
    file_paths: list[str],
    lsp_pool: Any,
    max_symbols_per_file: int = 10,
    max_refs_per_symbol: int = 20,
) -> dict:
    """
    Gather LSP impact data for implementation planning.
    
    Extracts symbols from each file, then finds references to discover
    affected files outside the original scope.
    
    Args:
        file_paths: List of file paths to analyze
        lsp_pool: LspPool instance (or None for graceful fallback)
        max_symbols_per_file: Limit symbols per file for performance
        max_refs_per_symbol: Limit references per symbol
        
    Returns:
        {
            "symbols_by_file": {"file.py": [{"name": "func", "line": 10, "type": "function"}, ...]},
            "affected_files": {"other.py": {"symbols_used": ["func"], "count": 3}},
            "callers_outside_scope": ["other.py", ...],
            "lsp_available": True/False,
            "error": None or "reason"
        }
    """
    result = {
        "symbols_by_file": {},
        "affected_files": {},
        "callers_outside_scope": [],
        "lsp_available": False,
        "error": None,
    }
    
    # Graceful fallback if no LSP
    if lsp_pool is None:
        result["error"] = "No LSP pool available"
        return result
    
    try:
        # Ensure pool is ready
        if hasattr(lsp_pool, '_ensure_started'):
            await lsp_pool._ensure_started()
        
        if not getattr(lsp_pool, '_started', False):
            result["error"] = "LSP not started"
            return result
        
        result["lsp_available"] = True
        
        # Normalize input paths
        scope_files = set()
        for fp in file_paths:
            resolved = Path(fp).resolve()
            scope_files.add(str(resolved))
        
        all_affected = {}  # file -> {symbols_used: [], count: 0}
        
        for file_path in file_paths:
            resolved_path = str(Path(file_path).resolve())
            
            # Get symbols from file
            symbols = await lsp_pool.document_symbols(resolved_path)
            if not symbols:
                result["symbols_by_file"][file_path] = []
                continue
            
            # Limit and store symbols
            limited_symbols = symbols[:max_symbols_per_file]
            result["symbols_by_file"][file_path] = [
                {"name": s.get("name"), "line": s.get("line_start", 0), "type": s.get("type")}
                for s in limited_symbols
            ]
            
            # Find references for top-level symbols (functions, classes)
            for sym in limited_symbols:
                sym_name = sym.get("name")
                sym_type = sym.get("type", "")
                sym_line = sym.get("line_start", 0)
                
                # Skip private/internal symbols and variables
                if sym_name.startswith("_") or sym_type in ("variable", "constant"):
                    continue
                
                try:
                    refs = await lsp_pool.find_references(
                        resolved_path, 
                        sym_line, 
                        0  # Column 0 for function/class definitions
                    )
                    
                    if not refs:
                        continue
                    
                    # Limit references
                    for ref in refs[:max_refs_per_symbol]:
                        ref_file = ref.get("file", "")
                        if not ref_file:
                            continue
                            
                        ref_resolved = str(Path(ref_file).resolve())
                        
                        # Track all affected files
                        if ref_resolved not in all_affected:
                            all_affected[ref_resolved] = {
                                "symbols_used": [],
                                "count": 0
                            }
                        
                        if sym_name not in all_affected[ref_resolved]["symbols_used"]:
                            all_affected[ref_resolved]["symbols_used"].append(sym_name)
                        all_affected[ref_resolved]["count"] += 1
                        
                except Exception as e:
                    logger.debug(f"Error finding references for {sym_name}: {e}")
                    continue
        
        # Separate affected files vs callers outside scope
        for affected_file, data in all_affected.items():
            result["affected_files"][affected_file] = data
            if affected_file not in scope_files:
                result["callers_outside_scope"].append(affected_file)
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"get_lsp_impact error: {e}")
        return result


def format_lsp_impact_for_plan(impact: dict) -> str:
    """
    Format LSP impact data as markdown for implementation plans.
    
    Args:
        impact: Result from get_lsp_impact()
        
    Returns:
        Markdown string for plan's LSP Impact Analysis section
    """
    lines = ["## LSP Impact Analysis", ""]
    
    if not impact.get("lsp_available"):
        lines.append(f"> LSP unavailable: {impact.get('error', 'unknown')}")
        lines.append("> Please manually call find_references for key symbols.")
        return "\n".join(lines)
    
    # Symbols in scope
    lines.append("**Symbols in scope** (from document_symbols):")
    lines.append("| File | Key Symbols |")
    lines.append("|------|-------------|")
    
    for file, symbols in impact.get("symbols_by_file", {}).items():
        file_basename = Path(file).name
        symbol_names = [s["name"] for s in symbols[:5]]
        symbols_str = ", ".join(f"`{n}`" for n in symbol_names)
        if len(symbols) > 5:
            symbols_str += f" (+{len(symbols) - 5} more)"
        lines.append(f"| `{file_basename}` | {symbols_str} |")
    
    lines.append("")
    
    # Affected files
    lines.append("**Affected files** (from find_references):")
    lines.append("| File | Symbols Used | References |")
    lines.append("|------|--------------|------------|")
    
    for file, data in sorted(
        impact.get("affected_files", {}).items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )[:10]:
        file_basename = Path(file).name
        symbols_str = ", ".join(f"`{s}`" for s in data["symbols_used"][:3])
        if len(data["symbols_used"]) > 3:
            symbols_str += f" (+{len(data['symbols_used']) - 3})"
        lines.append(f"| `{file_basename}` | {symbols_str} | {data['count']} |")
    
    lines.append("")
    
    # Callers outside scope
    callers = impact.get("callers_outside_scope", [])
    if callers:
        lines.append("**Callers outside scope** (consider adding to scope):")
        for caller in sorted(callers)[:10]:
            lines.append(f"- `{Path(caller).name}`")
        if len(callers) > 10:
            lines.append(f"- ... and {len(callers) - 10} more")
    else:
        lines.append("**Callers outside scope**: None discovered")
    
    return "\n".join(lines)


# =============================================================================
# Phase 9: LSP Session Tracking
# =============================================================================


def log_lsp_lookup(
    session_id: str,
    symbol_name: str,
    lookup_type: str,  # 'find_references' or 'call_hierarchy'
    result_count: int,
    conn=None
) -> bool:
    """
    Record LSP lookup in session context for tracking.
    
    Args:
        session_id: The reasoning session UUID
        symbol_name: Symbol that was looked up
        lookup_type: Type of lookup performed
        result_count: Number of results found
        conn: Optional existing connection
        
    Returns:
        True if logged successfully
    """
    from pas.utils import get_db_connection
    import json
    
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        
        # Get current context
        cur.execute(
            "SELECT context FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        context = row["context"] or {}
        
        # Initialize lsp_lookups list if needed
        if "lsp_lookups" not in context:
            context["lsp_lookups"] = []
        
        # Append lookup record
        context["lsp_lookups"].append({
            "symbol": symbol_name,
            "type": lookup_type,
            "count": result_count
        })
        
        # Update session context
        cur.execute(
            "UPDATE reasoning_sessions SET context = %s WHERE id = %s",
            (json.dumps(context), session_id)
        )
        conn.commit()
        
        return True
        
    except Exception as e:
        logger.debug(f"log_lsp_lookup failed: {e}")
        return False
    finally:
        if should_close:
            conn.close()


def get_lsp_summary(session_id: str, conn=None) -> dict:
    """
    Get LSP lookup summary for a session.
    
    Compares performed lookups against suggested lookups from prepare_expansion.
    
    Args:
        session_id: The reasoning session UUID
        conn: Optional existing connection
        
    Returns:
        {
            "performed": [{"symbol": str, "type": str, "count": int}, ...],
            "suggested": [str, ...],  # From prepare_expansion suggested_lookups
            "coverage_ratio": float,  # performed/suggested
            "missing": [str, ...]     # Suggested but not performed
        }
    """
    from pas.utils import get_db_connection
    
    should_close = conn is None
    if conn is None:
        conn = get_db_connection()
    
    try:
        cur = conn.cursor()
        
        cur.execute(
            "SELECT context FROM reasoning_sessions WHERE id = %s",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return {"performed": [], "suggested": [], "coverage_ratio": 0.0, "missing": []}
        
        context = row["context"] or {}
        
        performed = context.get("lsp_lookups", [])
        suggested = context.get("suggested_lsp_lookups", [])
        
        # Extract performed symbol names
        performed_symbols = {p.get("symbol") for p in performed if isinstance(p, dict)}
        suggested_symbols = set(suggested) if isinstance(suggested, list) else set()
        
        # Compute coverage
        if suggested_symbols:
            covered = performed_symbols & suggested_symbols
            coverage_ratio = len(covered) / len(suggested_symbols)
        else:
            coverage_ratio = 1.0 if performed_symbols else 0.0
        
        missing = list(suggested_symbols - performed_symbols)
        
        return {
            "performed": performed,
            "suggested": suggested,
            "coverage_ratio": round(coverage_ratio, 2),
            "missing": missing
        }
        
    except Exception as e:
        logger.debug(f"get_lsp_summary failed: {e}")
        return {"performed": [], "suggested": [], "coverage_ratio": 0.0, "missing": [], "error": str(e)}
    finally:
        if should_close:
            conn.close()
