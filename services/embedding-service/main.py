"""
Embedding Service - External semantic embedding via bge-large-en-v1.5

Provides GPU-accelerated embeddings for code intelligence.
Runs as a standalone systemd service on port 5020.
"""
import os
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding-service")

# Model configuration
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model instance
_model = None


class EmbedRequest(BaseModel):
    text: str


class BatchEmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embedding: List[float]
    model: str
    device: str


class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    cuda_available: bool


def get_model():
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        logger.info(f"Loading model: {MODEL_NAME} on {DEVICE}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info(f"Model loaded successfully")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting embedding service...")
    get_model()  # Pre-load model
    yield
    logger.info("Shutting down embedding service...")


app = FastAPI(
    title="Embedding Service",
    description="GPU-accelerated semantic embeddings for code intelligence",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embedding for a single text."""
    try:
        model = get_model()
        embedding = model.encode(request.text, normalize_embeddings=True)
        return EmbedResponse(
            embedding=embedding.tolist(),
            model=MODEL_NAME,
            device=DEVICE
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchEmbedResponse)
async def batch_embed(request: BatchEmbedRequest):
    """Generate embeddings for multiple texts."""
    try:
        model = get_model()
        embeddings = model.encode(request.texts, normalize_embeddings=True)
        return BatchEmbedResponse(
            embeddings=[e.tolist() for e in embeddings],
            count=len(request.texts),
            model=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        device=DEVICE,
        cuda_available=torch.cuda.is_available()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5020)
