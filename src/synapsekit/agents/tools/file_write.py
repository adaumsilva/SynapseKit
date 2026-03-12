from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class FileWriteTool(BaseTool):
    """Write content to a local file."""

    name = "file_write"
    description = (
        "Write content to a file on disk. Creates parent directories if needed. "
        "Input: path and content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path to write to"},
            "content": {"type": "string", "description": "The content to write"},
            "append": {
                "type": "boolean",
                "description": "Append to file instead of overwriting (default: false)",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    async def run(
        self,
        path: str = "",
        content: str = "",
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        if not path:
            return ToolResult(output="", error="No file path provided.")

        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

            action = "Appended to" if append else "Written to"
            return ToolResult(output=f"{action} {path} ({len(content)} chars)")
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error writing file: {e}")
