"""Code execution sandbox — safely invokes selected tools.

Takes the LLM's tool selection (name + parameters), looks up the
function in the tool registry, and executes it with error handling.
This is the "Execution & Fallback" step from the research paper.

If the tool crashes, the error is caught and returned gracefully
rather than crashing the entire API.
"""

from __future__ import annotations

import logging
from typing import Any

from model.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class Executor:
    """Sandboxed tool executor with graceful error handling."""

    def __init__(self) -> None:
        self.tools = {t["name"]: t["function"] for t in TOOL_REGISTRY}

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a named tool with the given arguments.

        Args:
            tool_name: Name of the tool to call (must be in TOOL_REGISTRY).
            args: Keyword arguments dict to pass to the tool function.

        Returns:
            Dict with 'result' (tool output or None) and 'error' (str or None).
        """
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            logger.error("Unknown tool: %s (available: %s)", tool_name, available)
            return {
                "result": None,
                "error": f"Unknown tool '{tool_name}'. Available: {available}",
            }

        func = self.tools[tool_name]
        logger.info("Executing tool: %s | args=%s", tool_name, args)

        try:
            result = func(**args)
            logger.info("Tool %s executed successfully", tool_name)
            return {"result": result, "error": None}
        except TypeError as e:
            # Parameter mismatch — wrong arg names or missing required params
            logger.error("Parameter error for %s: %s", tool_name, e)
            return {"result": None, "error": f"Parameter error: {e}"}
        except Exception as e:
            logger.error("Execution error for %s: %s", tool_name, e, exc_info=True)
            return {"result": None, "error": f"Execution error: {e}"}

    def list_available_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self.tools.keys())
