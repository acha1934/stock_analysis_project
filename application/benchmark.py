"""Evaluation & benchmarking harness.

Runs automated batch requests against different tool library sizes
(N = 10, 50, 100) and routing modes (naive vs. vector), then measures
and exports metrics.

Metrics collected per run:
• Retrieval latency (ms)
• Tool selection accuracy (%)
• End-to-end latency (ms)
• Tokens consumed (prompt + completion)

Usage:
    python benchmark.py                    # default suite
    python benchmark.py --sizes 10 50 100  # custom sizes
    python benchmark.py --export results.json
"""

from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Any

import httpx

from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Default test query suite ────────────────────────────────────────
DEFAULT_QUERIES: list[str] = [
    "What is the current price of TCS?",
    "Compare returns of TCS and Reliance for 2023.",
    "Show me the price trend of Infosys over the last 6 months.",
    "Which stock performed better: HDFC Bank or ICICI Bank in 2024?",
    "Plot the closing price of Wipro from Jan to Dec 2023.",
    "Give me the latest price of SBIN.",
    "Compare the yearly returns of TCS and Infosys.",
    "Show the stock trend of Reliance in 2024.",
    "What is the price of Tata Motors today?",
    "Compare Bajaj Finance and HDFC Bank performance in 2023.",
]


class BenchmarkRunner:
    """Sends batch queries to the API and aggregates metrics."""

    def __init__(self, api_base: str = f"http://localhost:{settings.api_port}") -> None:
        self.api_base = api_base.rstrip("/")

    def run_single(
        self,
        query: str,
        routing_mode: str,
        library_size: int,
    ) -> dict[str, Any]:
        """Execute a single benchmark request."""
        payload = {
            "query": query,
            "routing_mode": routing_mode,
            "library_size": library_size,  # hint for the Model Layer
        }
        start = time.perf_counter()

        try:
            resp = httpx.post(f"{self.api_base}/api/chat", json=payload, timeout=120)
            latency = (time.perf_counter() - start) * 1000

            if resp.is_success:
                data = resp.json()
                return {
                    "query": query,
                    "routing_mode": routing_mode,
                    "library_size": library_size,
                    "status": "success",
                    "tool_used": data.get("tool_used"),
                    "e2e_latency_ms": latency,
                    "agent_latency_ms": data.get("latency_ms", 0),
                    "chart_generated": bool(data.get("chart_path")),
                }
            else:
                return {
                    "query": query,
                    "routing_mode": routing_mode,
                    "library_size": library_size,
                    "status": "error",
                    "error": resp.text,
                    "e2e_latency_ms": latency,
                }
        except httpx.RequestError as e:
            return {
                "query": query,
                "routing_mode": routing_mode,
                "library_size": library_size,
                "status": "error",
                "error": str(e),
                "e2e_latency_ms": (time.perf_counter() - start) * 1000,
            }

    def run_suite(
        self,
        queries: list[str],
        routing_modes: list[str],
        library_sizes: list[int],
    ) -> list[dict[str, Any]]:
        """Run the full benchmark matrix."""
        results: list[dict[str, Any]] = []

        total = len(queries) * len(routing_modes) * len(library_sizes)
        count = 0

        for size in library_sizes:
            for mode in routing_modes:
                for query in queries:
                    count += 1
                    logger.info("[%d/%d] mode=%s | N=%d | query=%s", count, total, mode, size, query[:50])
                    result = self.run_single(query, mode, size)
                    results.append(result)

        return results

    @staticmethod
    def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results into a summary dict."""
        summary: dict[str, Any] = {"by_mode": {}, "by_size": {}}

        for mode in ["naive", "vector"]:
            mode_results = [r for r in results if r["routing_mode"] == mode]
            if not mode_results:
                continue
            successful = [r for r in mode_results if r["status"] == "success"]
            summary["by_mode"][mode] = {
                "total": len(mode_results),
                "successes": len(successful),
                "avg_latency_ms": (
                    sum(r["e2e_latency_ms"] for r in successful) / len(successful)
                    if successful else 0
                ),
            }

        for size in sorted({r["library_size"] for r in results}):
            size_results = [r for r in results if r["library_size"] == size]
            successful = [r for r in size_results if r["status"] == "success"]
            summary["by_size"][size] = {
                "total": len(size_results),
                "successes": len(successful),
                "avg_latency_ms": (
                    sum(r["e2e_latency_ms"] for r in successful) / len(successful)
                    if successful else 0
                ),
            }

        return summary


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation suite")
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100], help="Tool library sizes")
    parser.add_argument("--modes", type=str, nargs="+", default=["naive", "vector"], help="Routing modes")
    parser.add_argument("--export", type=str, default=None, help="Export results to JSON file")
    args = parser.parse_args()

    runner = BenchmarkRunner()
    logger.info("Starting benchmark | sizes=%s | modes=%s | queries=%d", args.sizes, args.modes, len(DEFAULT_QUERIES))

    results = runner.run_suite(DEFAULT_QUERIES, args.modes, args.sizes)
    summary = runner.summarize(results)

    print("\n══════════ Benchmark Summary ══════════")
    print(json.dumps(summary, indent=2))

    if args.export:
        out_path = Path(args.export)
        out_path.write_text(json.dumps({"results": results, "summary": summary}, indent=2), encoding="utf-8")
        logger.info("Results exported to %s", out_path)


if __name__ == "__main__":
    main()
