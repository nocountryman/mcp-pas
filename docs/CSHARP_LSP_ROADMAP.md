# Adding C# LSP Support to PAS

## Current State
- PAS uses `basedpyright` for Python via subprocess isolation
- Hardcoded in `subprocess_worker.py` line 26-32

## Required Changes

### Phase 1: Install C# LSP Server

```bash
# Option A: csharp-ls (lighter, faster)
dotnet tool install --global csharp-ls

# Option B: OmniSharp (more features, heavier)
# Download from https://github.com/OmniSharp/omnisharp-roslyn/releases
```

### Phase 2: Multi-Language LspSubprocess

Modify `subprocess_worker.py`:

```python
# Line 21-34: Replace hardcoded basedpyright with language detection
def _lsp_worker(request_queue, response_queue, project_root, language):
    """Worker process that handles LSP requests."""
    
    async def run_worker():
        if language == "python":
            from lsp_client import BasedpyrightClient, LocalServer
            server = LocalServer(
                program="basedpyright-langserver",
                args=["--stdio"],
                cwd=Path(project_root)
            )
            ClientClass = BasedpyrightClient
            
        elif language == "csharp":
            from lsp_client import LspClient, LocalServer  # Generic client
            server = LocalServer(
                program="csharp-ls",  # Or: "OmniSharp" 
                args=[],
                cwd=Path(project_root)
            )
            ClientClass = LspClient
            
        # ... rest of worker
```

### Phase 3: Language Detection

```python
# In lsp_pool.py
def detect_language(project_root: str) -> str:
    """Detect project language from files."""
    root = Path(project_root)
    
    # Check for C# project files
    if list(root.glob("**/*.csproj")) or list(root.glob("**/*.sln")):
        return "csharp"
    
    # Check for Python files
    if list(root.glob("**/*.py")) or (root / "pyproject.toml").exists():
        return "python"
    
    return "unknown"
```

### Phase 4: Update PAS Tools

Update `find_references`, `call_hierarchy` in `server.py` to pass language:

```python
@mcp.tool()
async def find_references(project_id: str, symbol_name: str):
    project = get_project(project_id)
    language = detect_language(project.path)
    pool = await LspPool.get(project.path, language=language)
    # ...
```

## Effort Estimate

| Phase | Hours | Notes |
|-------|-------|-------|
| 1. Install LSP | 0.5 | Just `dotnet tool install` |
| 2. Multi-lang worker | 2 | Refactor subprocess_worker.py |
| 3. Language detection | 0.5 | Simple file glob |
| 4. Update tools | 1 | Pass language param |
| **Total** | **4 hours** | |

## Immediate Workaround

Until implemented, use grep-based call flow:

```bash
grep -n "RunOnce\|MonitorGap\|SpawnHelper" Program.cs
```

## Decision Point

**Do you want me to implement this now?** 

It would give PAS full C# support for:
- `call_hierarchy(symbol_name)` 
- `find_references(symbol_name)`
- `query_codebase()` with symbols
