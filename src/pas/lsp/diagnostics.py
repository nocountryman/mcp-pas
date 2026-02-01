"""Diagnostics handling for LSP.

Provides access to LSP diagnostics (errors, warnings, hints).

v52b: Removed multilspy dependency. Diagnostics are typically pushed via
notifications and not directly exposed as a request in most LSP libraries.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_diagnostics(
    client: Any,
    file_path: str
) -> list[dict]:
    """Get diagnostics for a file.
    
    Note: lsp-client receives diagnostics via notifications (WithReceivePublishDiagnostics).
    This function returns empty for now - can be enhanced to collect pushed diagnostics.
    
    Args:
        client: Active lsp-client instance
        file_path: Path to file
        
    Returns:
        List of diagnostic dicts with message, severity, range
    """
    # Diagnostics come via textDocument/publishDiagnostics notifications
    # For now, return empty - can be enhanced to collect from notification handler
    logger.debug(f"Diagnostics not yet implemented for {file_path}")
    return []


def format_diagnostic(diag: dict) -> dict:
    """Format raw LSP Diagnostic for PAS consumption.
    
    Args:
        diag: Raw LSP Diagnostic object
        
    Returns:
        Simplified dict with message, severity, location
    """
    severity_map = {
        1: "error",
        2: "warning", 
        3: "info",
        4: "hint",
    }
    
    result = {
        "message": diag.get("message", ""),
        "severity": severity_map.get(diag.get("severity", 4), "hint"),
        "source": diag.get("source", ""),
    }
    
    if "range" in diag:
        r = diag["range"]
        result["start_line"] = r.get("start", {}).get("line", 0)
        result["start_col"] = r.get("start", {}).get("character", 0)
        result["end_line"] = r.get("end", {}).get("line", 0)
        result["end_col"] = r.get("end", {}).get("character", 0)
    
    return result


def diagnostics_to_critique_prompts(diagnostics: list[dict]) -> list[str]:
    """Convert diagnostics to critique prompts for PAS.
    
    Args:
        diagnostics: List of formatted diagnostics
        
    Returns:
        List of critique prompt strings
    """
    prompts = []
    for diag in diagnostics:
        severity = diag.get("severity", "hint")
        if severity in ("error", "warning"):
            line = diag.get("start_line", 0)
            msg = diag.get("message", "Unknown issue")
            prompts.append(f"LSP {severity.upper()}: {msg} (line {line})")
    return prompts
