"""Central configuration for the Application Layer."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── API server ──────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Model Layer endpoint (Part 1 bridge) ────────────────────
    # Points to the local Ollama / OpenAI-compatible endpoint used by agent.py
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"

    # ── Routing mode ────────────────────────────────────────────
    # "naive"  – full-prompt routing (all tools in context)
    # "vector" – semantic retrieval routing (top-k tools)
    routing_mode: str = "vector"
    top_k_tools: int = 5

    # ── Groq API (LLM backend) ─────────────────────────────────
    # Get a free key at https://console.groq.com
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"

    # ── Paths ───────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent
    charts_dir: Path = base_dir / "static" / "charts"
    dataset_path: Path = base_dir.parent / "data" / "indian_stocks.csv"


settings = Settings()

# Ensure chart output directory exists
settings.charts_dir.mkdir(parents=True, exist_ok=True)
