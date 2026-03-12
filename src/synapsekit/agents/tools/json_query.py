from __future__ import annotations

import json
from typing import Any

from ..base import BaseTool, ToolResult


class JSONQueryTool(BaseTool):
    """Query JSON data using dot-notation paths."""

    name = "json_query"
    description = (
        "Query JSON data using a dot-notation path (e.g. 'users.0.name'). "
        "Input: json_data (string or object) and path."
    )
    parameters = {
        "type": "object",
        "properties": {
            "json_data": {
                "type": "string",
                "description": "JSON string to query",
            },
            "path": {
                "type": "string",
                "description": "Dot-notation path (e.g. 'users.0.name', 'data.items')",
            },
        },
        "required": ["json_data", "path"],
    }

    async def run(
        self,
        json_data: str = "",
        path: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        if not json_data:
            return ToolResult(output="", error="No JSON data provided.")
        if not path:
            return ToolResult(output="", error="No path provided.")

        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
        except json.JSONDecodeError as e:
            return ToolResult(output="", error=f"Invalid JSON: {e}")

        try:
            result = self._traverse(data, path)
            if isinstance(result, (dict, list)):
                return ToolResult(output=json.dumps(result, indent=2))
            return ToolResult(output=str(result))
        except (KeyError, IndexError, TypeError) as e:
            return ToolResult(output="", error=f"Path error: {e}")

    def _traverse(self, data: Any, path: str) -> Any:
        """Traverse JSON using dot-notation path."""
        current = data
        for part in path.split("."):
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise TypeError(f"Cannot traverse into {type(current).__name__} with key {part!r}")
        return current
