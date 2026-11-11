"""Bridge between the Application Layer and the Model Layer (Part 1).

This service is the single integration point. It orchestrates the full
agent pipeline:

1. Retrieve candidate tools (naive = all, vector = top-k via FAISS)
2. Ask the LLM (Groq) to select a tool and extract parameters
3. Execute the selected tool in the sandbox
4. Synthesize a natural-language answer from the tool's output
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import settings
from model.tools import TOOL_REGISTRY, list_tools as _list_tools_meta
from model.retriever import ToolRetriever
from model.agent import Agent
from model.executor import Executor

logger = logging.getLogger(__name__)


class AgentService:
    """Orchestrates tool retrieval, LLM planning, code execution, and synthesis."""

    def __init__(self, routing_mode: str | None = None) -> None:
        self.routing_mode = routing_mode or settings.routing_mode

        # Initialize Model Layer components
        self.retriever = ToolRetriever()
        self.agent = Agent()
        self.executor = Executor()

        logger.info(
            "AgentService initialized | routing=%s | tools=%d | model=%s",
            self.routing_mode, len(TOOL_REGISTRY), settings.groq_model,
        )

    # ── Tool index ───────────────────────────────────────────────

    def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata for every registered tool."""
        return _list_tools_meta()

    # ── Chat / query processing ──────────────────────────────────

    def process_query(self, query: str, routing_mode: str | None = None,
                      history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Run the full agent pipeline on a user query.

        Steps:
        1. Retrieve candidate tools (naive = all, vector = top-k).
        2. Ask LLM to select a tool and extract parameters (JSON).
        3. Execute the selected tool in the sandbox.
        4. Synthesize a natural-language answer.
        5. Return answer text, tool info, and optional chart path.
        """
        start = time.perf_counter()
        mode = routing_mode or self.routing_mode

        logger.info("Processing query | mode=%s | query=%s", mode, query)

        # ── Step 1: Retrieve candidate tools ──────────────────
        if mode == "naive":
            candidate_tools = TOOL_REGISTRY
            logger.info("Naive mode: using all %d tools", len(candidate_tools))
        else:
            retrieved = self.retriever.retrieve(query, top_k=settings.top_k_tools)
            candidate_tools = [r["tool"] for r in retrieved]
            logger.info(
                "Vector mode: retrieved %d tools: %s",
                len(candidate_tools),
                [t["name"] for t in candidate_tools],
            )

        # ── Step 2: LLM tool selection + parameter extraction ──
        llm_result = self.agent.select_tool(query, candidate_tools, history=history)

        tool_used = llm_result.get("tool")
        tool_args = llm_result.get("args")

        # ── Step 3: Execute the selected tool ─────────────────
        chart_path = None

        if tool_used and tool_args:
            exec_result = self.executor.execute(tool_used, tool_args)
            tool_output = exec_result["result"]
            error = exec_result["error"]

            if error:
                answer = f"⚠️ {error}"
            elif tool_output and "error" in tool_output:
                answer = f"⚠️ {tool_output['error']}"
            else:
                # Extract chart path if the tool produced one
                if isinstance(tool_output, dict) and "chart_path" in tool_output:
                    chart_path = tool_output["chart_path"]

                # ── Step 4: Synthesize natural-language answer ──
                answer = self.agent.synthesize_answer(query, tool_used, tool_args, tool_output)
        else:
            answer = (
                "I couldn't find a suitable tool for your query. "
                "Try asking about stock prices, comparing returns, or plotting trends "
                "for Indian stocks like TCS, Reliance, Infosys, etc."
            )

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Query completed | tool=%s | latency=%.0fms", tool_used, latency_ms)

        return {
            "answer": answer,
            "tool_used": tool_used,
            "tool_args": tool_args,
            "chart_path": chart_path,
            "routing_mode": mode,
            "latency_ms": latency_ms,
        }


# Singleton instance for import across modules
agent_service = AgentService()
