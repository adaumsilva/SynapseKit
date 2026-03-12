from __future__ import annotations

import os
from typing import Any

from ..base import BaseTool, ToolResult


class FileListTool(BaseTool):
    """List files and directories at a given path."""

    name = "file_list"
    description = (
        "List files and directories in a given path. "
        "Input: directory path, optional recursive flag and glob pattern."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list"},
            "recursive": {
                "type": "boolean",
                "description": "List recursively (default: false)",
                "default": False,
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter (e.g. '*.py')",
                "default": "",
            },
        },
        "required": ["path"],
    }

    async def run(
        self,
        path: str = "",
        recursive: bool = False,
        pattern: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        path = path or kwargs.get("input", ".")
        if not os.path.isdir(path):
            return ToolResult(output="", error=f"Not a directory: {path!r}")

        try:
            if recursive:
                import fnmatch

                entries = []
                for root, dirs, files in os.walk(path):
                    for name in dirs + files:
                        full = os.path.join(root, name)
                        rel = os.path.relpath(full, path)
                        if pattern and not fnmatch.fnmatch(name, pattern):
                            continue
                        suffix = "/" if os.path.isdir(full) else ""
                        entries.append(rel + suffix)
                entries.sort()
            else:
                import fnmatch

                items = sorted(os.listdir(path))
                entries = []
                for name in items:
                    if pattern and not fnmatch.fnmatch(name, pattern):
                        continue
                    full = os.path.join(path, name)
                    suffix = "/" if os.path.isdir(full) else ""
                    entries.append(name + suffix)

            if not entries:
                return ToolResult(output="(empty directory)")
            return ToolResult(output="\n".join(entries))
        except PermissionError:
            return ToolResult(output="", error=f"Permission denied: {path!r}")
        except Exception as e:
            return ToolResult(output="", error=f"Error listing directory: {e}")
