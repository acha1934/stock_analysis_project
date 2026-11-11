"""FastAPI backend server — Application Layer entry point.

Endpoints
─────────
POST /api/chat       → accept a natural-language query, return answer + chart
GET  /api/tools      → list all registered tools
POST /api/benchmark  → run evaluation over a test query suite

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import settings
from schemas import ChatRequest, ChatResponse, ToolInfo, BenchmarkRequest, BenchmarkResult
from services.agent_service import agent_service

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Application Layer | routing=%s | port=%d", settings.routing_mode, settings.api_port)
    yield
    logger.info("Shutting down Application Layer")


# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Agent — Application Layer",
    description="Full-stack infrastructure for the AI stock-analysis agent",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/api/tools", response_model=list[ToolInfo])
async def get_tools():
    """Return the current index of tools available in the system."""
    return agent_service.list_tools()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Accept a user query, invoke the Model Layer, return textual + chart output."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = agent_service.process_query(req.query, routing_mode=req.routing_mode,
                                          history=req.history)
    return ChatResponse(**result)


@app.post("/api/benchmark", response_model=BenchmarkResult)
async def benchmark(req: BenchmarkRequest):
    """Run automated batch evaluation over different library sizes and routing modes."""
    # TODO: wire to benchmark.py runner
    raise HTTPException(status_code=501, detail="Benchmark endpoint not yet implemented")


@app.get("/api/health")
async def health():
    """Simple health-check for container orchestration."""
    return {"status": "ok", "routing_mode": settings.routing_mode}


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
