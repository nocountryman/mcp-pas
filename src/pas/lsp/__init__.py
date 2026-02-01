"""PAS LSP Integration Module."""

from pas.lsp.manager import LspManager
from pas.lsp.config import detect_subprojects, get_config_for_language
from pas.lsp.diagnostics import get_diagnostics, diagnostics_to_critique_prompts
from pas.lsp.watcher import InotifyWatcher
from pas.lsp.lsp_pool import LspPool, get_pool, shutdown_pool

__all__ = [
    "LspManager",
    "LspPool",
    "get_pool",
    "shutdown_pool",
    "detect_subprojects",
    "get_config_for_language",
    "get_diagnostics",
    "diagnostics_to_critique_prompts",
    "InotifyWatcher",
]
