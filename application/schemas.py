"""Pydantic request / response models for the API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /api/chat payload."""
    query: str = Field(..., description="Natural-language user query")
    routing_mode: Optional[str] = Field(
        default=None,
        description="Override routing mode: 'naive' or 'vector'",
    )
    history: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Prior conversation turns as [{role, content}] for context resolution",
    )


class BenchmarkRequest(BaseModel):
    """POST /api/benchmark payload."""
    library_sizes: list[int] = Field(
        default=[10, 50, 100],
        description="Tool library sizes to evaluate",
    )
    routing_modes: list[str] = Field(
        default=["naive", "vector"],
        description="Routing strategies to compare",
    )
    test_queries: Optional[list[str]] = Field(
        default=None,
        description="Custom query suite; defaults to built-in set",
    )


# ── Responses ───────────────────────────────────────────────────────

class ToolInfo(BaseModel):
    """Metadata for a single tool."""
    name: str
    description: str
    parameters: dict[str, Any]


class ChatResponse(BaseModel):
    """POST /api/chat response."""
    answer: str = Field(..., description="Textual answer from the agent")
    tool_used: Optional[str] = Field(None, description="Name of the invoked tool")
    tool_args: Optional[dict[str, Any]] = Field(None, description="Arguments passed")
    chart_path: Optional[str] = Field(None, description="Path to generated chart image")
    routing_mode: str = Field(..., description="Which routing mode was used")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")


class BenchmarkResult(BaseModel):
    """POST /api/benchmark response."""
    results: list[dict[str, Any]] = Field(
        ..., description="Per-run benchmark metrics"
    )
    summary: dict[str, Any] = Field(..., description="Aggregated summary stats")
