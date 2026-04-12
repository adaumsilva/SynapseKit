from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

from .base import Document

if TYPE_CHECKING:
    from dropbox import Dropbox

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".json",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".rb",
    ".go",
    ".rs",
    ".php",
    ".sh",
    ".bat",
}


class DropboxLoader:
    """Load files from a Dropbox folder as Documents."""

    def __init__(
        self,
        access_token: str,
        path: str,
        limit: int | None = None,
    ) -> None:
        self._access_token = access_token
        self._path = path
        self._limit = limit

    def load(self) -> list[Document]:
        try:
            import dropbox
        except ImportError:
            raise ImportError("dropbox required: pip install synapsekit[dropbox]") from None

        dbx = dropbox.Dropbox(self._access_token)
        docs: list[Document] = []

        entries = self._list_folder(dbx, self._path)

        for _entry in entries:
            if self._limit is not None and len(docs) >= self._limit:
                break

            entry = self._normalise_entry(_entry)

            if entry.get(".tag") != "file":
                continue

            name = entry.get("name", "")
            path_display = entry.get("path_display", "")
            server_modified = entry.get("server_modified", "")

            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            try:
                content = self._download_file(dbx, path_display)
            except Exception:
                continue

            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                continue

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "source": "dropbox",
                        "filename": name,
                        "path": path_display,
                        "modified": server_modified,
                    },
                )
            )

        return docs

    async def aload(self) -> list[Document]:
        """Load files asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.load)

    @staticmethod
    def _normalise_entry(entry: Any) -> dict[str, Any]:
        """Convert a Dropbox SDK entry object or dict to a canonical dict.

        The real Dropbox SDK (v12+) returns Stone-generated objects like
        FileMetadata/FolderMetadata — not dicts.  Test mocks use plain dicts.
        This normaliser handles both so the rest of load() stays unchanged.
        """
        if isinstance(entry, dict):
            return entry

        # Real Dropbox SDK object: use attribute access.
        type_name = type(entry).__name__  # "FileMetadata" | "FolderMetadata" | ...
        if type_name == "FileMetadata":
            tag = "file"
        elif type_name == "FolderMetadata":
            tag = "folder"
        else:
            tag = type_name.replace("Metadata", "").lower()

        modified = getattr(entry, "server_modified", None)
        if modified is not None and hasattr(modified, "isoformat"):
            modified_str = modified.isoformat()  # type: ignore[union-attr]
        else:
            modified_str = str(modified) if modified is not None else ""

        return {
            ".tag": tag,
            "name": getattr(entry, "name", "") or "",
            "path_display": getattr(entry, "path_display", "") or "",
            "server_modified": modified_str,
        }

    def _list_folder(self, dbx: Dropbox, path: str) -> list[Any]:
        entries: list[Any] = []

        try:
            result = dbx.files_list_folder(path)
        except Exception:
            return entries

        entries.extend(result.entries)

        while result.has_more:
            time.sleep(0.1)
            try:
                result = dbx.files_list_folder_continue(result.cursor)
                entries.extend(result.entries)
            except Exception:
                break

        return entries

    def _download_file(self, dbx: Dropbox, path: str) -> bytes:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            dbx.files_download_to_file(tmp_path, path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
