"""
SCIP Indexer Service - On-demand per-module code indexing

Provides code graph data via SCIP for code intelligence.
Runs as a standalone systemd service on port 5010.
No file watcher - uses task-boundary reindexing strategy.
"""
import os
import asyncio
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scip-indexer")

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://mcp_admin:12345@localhost:5432/mcp_pas"
)

# SCIP CLI paths
SCIP_CLI = os.path.expanduser("~/go/bin/scip")
SCIP_PYTHON = "/usr/sbin/scip-python"
SCIP_TYPESCRIPT = "scip-typescript"  # npm global

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


class IndexRequest(BaseModel):
    project_id: str
    module_path: str
    language: str = "python"


class IndexStaleRequest(BaseModel):
    project_id: str
    module_id: str


class IndexResponse(BaseModel):
    status: str
    module_id: str
    occurrences: int
    duration_ms: int


class StatusResponse(BaseModel):
    project_id: str
    modules: List[Dict[str, Any]]


async def get_pool() -> asyncpg.Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return _pool


async def get_last_index_time(project_id: str, module_id: str) -> Optional[datetime]:
    """Get last index time for a module."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT indexed_at FROM scip_modules 
            WHERE project_id = $1 AND module_id = $2
        """, project_id, module_id)
        return row['indexed_at'] if row else None


async def any_files_modified_since(module_path: str, since: datetime) -> bool:
    """Check if any files in module modified since given time."""
    if since is None:
        return True
    
    path = Path(module_path)
    if not path.exists():
        return False
    
    since_ts = since.timestamp()
    for f in path.rglob("*.py"):
        if f.stat().st_mtime > since_ts:
            return True
    for f in path.rglob("*.ts"):
        if f.stat().st_mtime > since_ts:
            return True
    for f in path.rglob("*.tsx"):
        if f.stat().st_mtime > since_ts:
            return True
    return False


async def run_scip_index(module_path: str, language: str) -> Path:
    """Run SCIP indexer on a module."""
    cmd = {
        "python": [SCIP_PYTHON, "index", "."],
        "typescript": [SCIP_TYPESCRIPT, "index", "."],
    }.get(language, [SCIP_PYTHON, "index", "."])
    
    logger.info(f"Running SCIP: {' '.join(cmd)} in {module_path}")
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=module_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"SCIP failed: {stderr.decode()}")
        raise RuntimeError(f"SCIP indexing failed: {stderr.decode()[:500]}")
    
    return Path(module_path) / "index.scip"

async def parse_scip_and_insert(
    scip_path: Path, 
    project_id: str, 
    module_id: str,
    module_path: str
) -> int:
    """Parse SCIP file and insert occurrences into database.
    
    Phase 35: Full occurrence parsing for code graph.
    Uses scip print --json to extract all symbol occurrences.
    """
    import json
    
    if not scip_path.exists():
        raise RuntimeError(f"SCIP index file not found: {scip_path}")
    
    # Phase 35: Use scip print --json for full occurrence data
    logger.info(f"Running scip print --json on: {scip_path}")
    proc = await asyncio.create_subprocess_exec(
        SCIP_CLI, "print", "--json", "index.scip",
        cwd=scip_path.parent,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    
    if stderr:
        logger.warning(f"scip print stderr: {stderr.decode()[:200]}")
    
    stdout_str = stdout.decode().strip()
    if not stdout_str:
        raise RuntimeError(f"scip print returned empty output for {scip_path}")
    
    try:
        data = json.loads(stdout_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}, stdout: {stdout_str[:500]}")
        raise RuntimeError(f"Failed to parse scip print output: {e}")
    
    documents = data.get("documents", [])
    logger.info(f"Parsing {len(documents)} documents from SCIP index")
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Delete existing module data (cascades to occurrences via FK)
        await conn.execute("""
            DELETE FROM scip_modules WHERE project_id = $1 AND module_id = $2
        """, project_id, module_id)
        
        # Insert module record
        await conn.execute("""
            INSERT INTO scip_modules (project_id, module_id, module_path, indexed_at, file_count)
            VALUES ($1, $2, $3, NOW(), $4)
        """, project_id, module_id, module_path, len(documents))
        
        # Parse and insert occurrences
        occurrence_count = 0
        batch = []
        
        for doc in documents:
            file_path = doc.get("relative_path", "")
            
            for occ in doc.get("occurrences", []):
                symbol = occ.get("symbol", "")
                
                # Skip local variables (too noisy)
                if not symbol or symbol.startswith("local "):
                    continue
                
                # Parse range: [line, col] or [line, col, endLine, endCol]
                range_arr = occ.get("range", [0, 0])
                line_start = range_arr[0] if len(range_arr) > 0 else 0
                col_start = range_arr[1] if len(range_arr) > 1 else 0
                line_end = range_arr[2] if len(range_arr) > 2 else line_start
                col_end = range_arr[3] if len(range_arr) > 3 else col_start
                
                role = occ.get("symbol_roles", 0)
                enclosing = occ.get("enclosing_symbol", "")
                
                batch.append((
                    project_id, module_id, file_path,
                    line_start, col_start, line_end, col_end,
                    symbol, role, enclosing
                ))
                occurrence_count += 1
                
                # Insert in batches of 500
                if len(batch) >= 500:
                    await conn.executemany("""
                        INSERT INTO scip_occurrences 
                        (project_id, module_id, file_path, line_start, col_start, 
                         line_end, col_end, symbol, symbol_role, enclosing_symbol)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, batch)
                    batch = []
        
        # Insert remaining batch
        if batch:
            await conn.executemany("""
                INSERT INTO scip_occurrences 
                (project_id, module_id, file_path, line_start, col_start, 
                 line_end, col_end, symbol, symbol_role, enclosing_symbol)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, batch)
        
        logger.info(f"Inserted {occurrence_count} occurrences for {module_id}")
    
    return occurrence_count


# FastAPI app
app = FastAPI(
    title="SCIP Indexer Service",
    description="On-demand per-module code indexing for code intelligence",
    version="1.0.0"
)


@app.post("/index", response_model=IndexResponse)
async def index_module(request: IndexRequest):
    """Index a specific module."""
    import time
    start = time.time()
    
    try:
        # Derive module_id from path
        module_id = request.module_path.replace("/", ".").replace("\\", ".")
        
        # Run SCIP
        scip_path = await run_scip_index(request.module_path, request.language)
        
        # Parse and insert
        count = await parse_scip_and_insert(
            scip_path, 
            request.project_id, 
            module_id,
            request.module_path
        )
        
        duration = int((time.time() - start) * 1000)
        return IndexResponse(
            status="indexed",
            module_id=module_id,
            occurrences=count,
            duration_ms=duration
        )
    except Exception as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-stale", response_model=IndexResponse)
async def index_if_stale(request: IndexStaleRequest):
    """Only reindex if files modified since last index."""
    try:
        last_indexed = await get_last_index_time(request.project_id, request.module_id)
        
        # TODO: Get module_path from DB or config
        module_path = request.module_id.replace(".", "/")
        
        if await any_files_modified_since(module_path, last_indexed):
            return await index_module(IndexRequest(
                project_id=request.project_id,
                module_path=module_path,
                language="python"
            ))
        
        return IndexResponse(
            status="up-to-date",
            module_id=request.module_id,
            occurrences=0,
            duration_ms=0
        )
    except Exception as e:
        logger.error(f"Stale check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{project_id}", response_model=StatusResponse)
async def get_status(project_id: str):
    """Get index status for a project."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT module_id, module_path, indexed_at, file_count
            FROM scip_modules
            WHERE project_id = $1
            ORDER BY module_id
        """, project_id)
        
        return StatusResponse(
            project_id=project_id,
            modules=[dict(r) for r in rows]
        )


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "scip_cli": os.path.exists(SCIP_CLI),
        "scip_python": os.path.exists(SCIP_PYTHON)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
