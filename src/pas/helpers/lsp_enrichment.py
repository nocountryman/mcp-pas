"""
LSP Enrichment Helpers

Pure functions for gathering LSP impact data to inform implementation plans.
Part of v52 Phase 2: LSP enrichment for PAS workflow.
"""

import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger("pas-server")


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
