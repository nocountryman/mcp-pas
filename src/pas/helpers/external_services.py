"""
External Services Client - HTTP clients for code intelligence services.

Provides async HTTP clients for:
- Embedding Service (bge-large-en-v1.5 on port 5020)
- SCIP Indexer Service (on port 5010)

This module replaces the embedded model loading with external service calls.
"""
import os
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger("pas.external_services")

# Service URLs (configurable via environment)
EMBEDDING_URL = os.getenv("PAS_EMBEDDING_URL", "http://127.0.0.1:5020")
SCIP_URL = os.getenv("PAS_SCIP_URL", "http://127.0.0.1:5010")

# Timeout settings
DEFAULT_TIMEOUT = 30.0  # seconds
EMBEDDING_TIMEOUT = 60.0  # embeddings can take longer on first call


async def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text from external service.
    
    Replaces the embedded SentenceTransformer model.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats (1024 dimensions for bge-large)
        
    Raises:
        httpx.HTTPError: If service is unavailable
    """
    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{EMBEDDING_URL}/embed",
                json={"text": text}
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except httpx.HTTPError as e:
            logger.error(f"Embedding service error: {e}")
            raise


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts from external service.
    
    More efficient than calling get_embedding() in a loop.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embeddings (each 1024 dimensions)
    """
    if not texts:
        return []
    
    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{EMBEDDING_URL}/batch",
                json={"texts": texts}
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except httpx.HTTPError as e:
            logger.error(f"Embedding batch error: {e}")
            raise


async def check_embedding_service() -> bool:
    """Check if embedding service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{EMBEDDING_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def trigger_index(
    project_id: str, 
    module_path: str, 
    language: str = "python"
) -> dict:
    """
    Trigger SCIP indexing for a module.
    
    Args:
        project_id: Project identifier
        module_path: Absolute path to module directory
        language: Programming language (python, typescript)
        
    Returns:
        Index result with occurrences count and duration
    """
    async with httpx.AsyncClient(timeout=120.0) as client:  # Indexing can be slow
        try:
            resp = await client.post(
                f"{SCIP_URL}/index",
                json={
                    "project_id": project_id,
                    "module_path": module_path,
                    "language": language
                }
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"SCIP index error: {e}")
            raise


async def index_if_stale(project_id: str, module_id: str) -> dict:
    """
    Index a module only if files have changed.
    
    Args:
        project_id: Project identifier
        module_id: Module identifier (dot-separated path)
        
    Returns:
        Index result (status: "indexed" or "up-to-date")
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"{SCIP_URL}/index-stale",
                json={
                    "project_id": project_id,
                    "module_id": module_id
                }
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"SCIP stale check error: {e}")
            raise


async def get_index_status(project_id: str) -> dict:
    """Get index status for a project."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            resp = await client.get(f"{SCIP_URL}/status/{project_id}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"SCIP status error: {e}")
            raise


async def check_scip_service() -> bool:
    """Check if SCIP indexer service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SCIP_URL}/health")
            return resp.status_code == 200
    except Exception:
        return False


async def get_indexed_modules(project_id: str) -> list[dict]:
    """Get list of indexed modules for a project."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            resp = await client.get(f"{SCIP_URL}/status/{project_id}")
            resp.raise_for_status()
            return resp.json().get("modules", [])
        except httpx.HTTPError as e:
            logger.error(f"SCIP get modules error: {e}")
            return []


async def ensure_fresh_index(project_id: str, module_path: str, language: str = "python") -> dict:
    """
    Ensure a module is indexed, triggering reindex if stale.
    
    Call this at task boundaries (before research, after edits).
    """
    # Derive module_id from path
    module_id = module_path.replace("/", ".").replace("\\", ".")
    
    try:
        result = await index_if_stale(project_id, module_id)
        return result
    except httpx.HTTPError:
        # If stale check fails, try full index
        return await trigger_index(project_id, module_path, language)

