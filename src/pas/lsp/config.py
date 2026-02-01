"""Language server configuration and detection."""

from pathlib import Path


LANGUAGE_CONFIGS = {
    "python": {
        "code_language": "python",
        "markers": ["pyproject.toml", "setup.py", "requirements.txt"],
    },
    "typescript": {
        "code_language": "typescript",
        "markers": ["tsconfig.json"],
        "requires": ["package.json"],
    },
}

# lsp-client configuration for Python (basedpyright)
PYTHON_CONFIG = {
    "code_language": "python",
    "trace_lsp_communication": False,
}

# lsp-client configuration for TypeScript (tsserver)
TYPESCRIPT_CONFIG = {
    "code_language": "typescript",
    "trace_lsp_communication": False,
}


def get_config_for_language(language: str) -> dict:
    """Get lsp-client config for language.
    
    Args:
        language: Language identifier ("python", "typescript")
        
    Returns:
        Config dict for lsp-client
    """
    configs = {
        "python": PYTHON_CONFIG,
        "typescript": TYPESCRIPT_CONFIG,
    }
    return configs.get(language, {"code_language": language})


def detect_subprojects(project_root: str) -> list[dict]:
    """Detect language subprojects with their paths.
    
    Supports monorepos where frontend/backend are in subdirs.
    
    Args:
        project_root: Absolute path to project root
        
    Returns:
        List of {"language": str, "path": str}
    """
    subprojects = []
    root = Path(project_root)
    
    if not root.exists():
        return subprojects
    
    # Check root and immediate subdirs
    dirs_to_check = [root] + [
        d for d in root.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    for dir_path in dirs_to_check:
        # Python detection
        if (dir_path / "pyproject.toml").exists():
            subprojects.append({
                "language": "python",
                "path": str(dir_path),
            })
        
        # TypeScript detection (requires both tsconfig.json AND package.json)
        if (dir_path / "tsconfig.json").exists() and (dir_path / "package.json").exists():
            subprojects.append({
                "language": "typescript",
                "path": str(dir_path),
            })
    
    return subprojects
