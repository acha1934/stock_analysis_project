"""Agent orchestrator — LLM-based tool selection and answer synthesis.

Uses the Groq API (OpenAI-compatible) with Llama 3.1 to:
1. Select the most appropriate tool from the candidate set
2. Extract the required parameters as structured JSON
3. Synthesize a natural-language answer from the tool's output

This implements Phase 2 of the research paper:
- Intent Analysis & Task Planning (select_tool)
- Visualization & Synthesis (synthesize_answer)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from config import settings
from model.tools import list_available_tickers

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class Agent:
    """LLM orchestrator that selects tools and generates answers."""

    def __init__(self) -> None:
        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com"
            )
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = settings.groq_model

    # ── Tool selection & parameter extraction ──────────────────────

    def select_tool(
        self, query: str, candidate_tools: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Ask the LLM to pick a tool and extract parameters.

        Args:
            query: Natural-language user query.
            candidate_tools: Tools retrieved by the retriever (or all tools for naive mode).
            history: Prior conversation turns as [{role, content}] for context resolution.

        Returns:
            Dict with 'tool' (name or None) and 'args' (dict or None).
        """
        tools_json = json.dumps(
            [
                {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
                for t in candidate_tools
            ],
            indent=2,
        )

        tickers = list_available_tickers()
        tickers_str = ", ".join(tickers)

        system_prompt = f"""You are a stock analysis assistant. Based on the user's query and conversation history, select the most appropriate tool and extract the required parameters.

Available tools:
{tools_json}

Available stock tickers in the dataset: {tickers_str}

Respond with ONLY a valid JSON object in this exact format:
{{"tool": "tool_name", "args": {{...}}}}

Rules:
- Use the conversation history to resolve references like "this", "that stock", "same period", "the whole year", "show as a graph", etc.
- Map company names to the correct ticker (e.g., "TCS" -> "TCS.NS", "Reliance" -> "RELIANCE.NS", "Infosys" -> "INFY.NS", "HDFC Bank" -> "HDFCBANK.NS", "Wipro" -> "WIPRO.NS", "SBI" -> "SBIN.NS", "HCL" -> "HCLTECH.NS", "Maruti" -> "MARUTI.NS", "Sun Pharma" -> "SUNPHARMA.NS", "Bajaj Finance" -> "BAJFINANCE.NS", "Asian Paints" -> "ASIANPAINT.NS", "HUL" -> "HINDUNILVR.NS", "L&T" -> "LT.NS", "Airtel" -> "BHARTIARTL.NS", "ITC" -> "ITC.NS", "Adani" -> "ADANIENT.NS", "Tata Steel" -> "TATASTEEL.NS", "Dr Reddy" -> "DRREDDY.NS")
- If the user mentions THREE OR MORE stocks, or uses "multiple", "all", "several" -> use compare_multiple with a tickers list
- If the user says "compare" AND ("graph" OR "chart" OR "plot" OR "visual") for EXACTLY TWO stocks -> use plot_comparison
- If the user says "compare", "which is better", "who performed better" (no graph/chart) for two stocks -> use compare_returns
- If the user says "show as graph", "plot it", "visualise", "chart" for a SINGLE stock -> use plot_stock_trend
- If the user says "price", "current", "latest", "how much" -> use get_stock_price
- If the user doesn't specify a date range and history has no clue, use "2024-01-01" to "2024-12-31"
- If the user says "whole year" and a year can be inferred from history, use that full year
- If the user says "2023", use "2023-01-01" to "2023-12-31"
- If the user says "2024", use "2024-01-01" to "2024-12-31"
- If the user says "last 2 years", use "2022-01-01" to "2024-12-31"
- If no tool matches the query at all, respond with {{"tool": null, "args": null}}
- Do NOT include any text outside the JSON object"""

        # Build messages — inject history for context resolution
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            # Only include last 6 turns to stay within token limits
            for turn in history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
                extra_body={"tool_choice": "none"},  # prevent native function-calling
            )
            content = response.choices[0].message.content.strip()
            parsed = self._parse_json_response(content)
            logger.info("Agent selected tool=%s | args=%s", parsed.get("tool"), parsed.get("args"))
            return parsed
        except Exception as e:
            logger.error("LLM tool selection failed: %s", e)
            return {"tool": None, "args": None, "error": str(e)}

    # ── Answer synthesis ───────────────────────────────────────────

    def synthesize_answer(
        self,
        query: str,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        tool_result: dict[str, Any],
    ) -> str:
        """Generate a concise natural-language summary of the tool's output.

        Args:
            query: The original user query.
            tool_name: Name of the tool that was called.
            tool_args: Parameters that were passed.
            tool_result: The raw output from the tool.

        Returns:
            Natural-language answer string.
        """
        if "error" in tool_result:
            return f"⚠️ {tool_result['error']}"

        system_prompt = """You are a stock analysis assistant. The user asked a question, a tool was executed, and you need to summarize the results in clear, concise natural language.

Format guidelines:
- Use bullet points for structured data
- Include specific numbers (prices, percentages)
- Keep it under 5-6 lines
- If a comparison was made, state which stock performed better
- Be direct and factual"""

        user_content = (
            f"User query: {query}\n\n"
            f"Tool used: {tool_name}\n"
            f"Tool arguments: {json.dumps(tool_args, default=str)}\n\n"
            f"Tool result:\n{json.dumps(tool_result, indent=2, default=str)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=500,
                extra_body={"tool_choice": "none"},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Answer synthesis failed: %s", e)
            # Fall back to raw JSON if synthesis fails
            return json.dumps(tool_result, indent=2, default=str)

    # ── JSON parsing helper ────────────────────────────────────────

    @staticmethod
    def _parse_json_response(content: str) -> dict[str, Any]:
        """Parse LLM response as JSON, handling markdown code blocks."""
        # Strip markdown code fences if present
        if "```" in content:
            # Extract content between code fences
            match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        return json.loads(content)
